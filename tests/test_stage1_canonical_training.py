import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from stage1_canonical_data import (
    TARGET_DEFINITION,
    _independent_measurements,
    _measurement_weight,
    _safe_int,
    _truthy,
    build_stage1_training_sites,
    canonical_structure_key,
    load_stage1_training_sites,
    scaffold_group_key,
)
from stage1_network_inference import load_stage1_bundle
from reference_residual_model import ReferenceResidualRegressor


def _network_row(molecule_id, smiles, values, family="carboxyl", label="carboxylic_acid"):
    measurements = [
        {
            "experimental_pka": value,
            "transition_row_id": f"transition_{index}",
            "source_file": "experimental.sdf",
            "record_index": index,
            "site_assignment_confidence": "high",
        }
        for index, value in enumerate(values)
    ]
    site = {
        "site_id": "site_1",
        "label": label,
        "family": family,
        "center_maps": [2],
        "experimental_anchor_pka": float(pd.Series(values).median()),
        "experimental_measurements": measurements,
    }
    return {
        "molecule_id": molecule_id,
        "representative_input_smiles": smiles,
        "representative_atom_mapped_smiles": "[CH3:1][C:2](=[O:3])[OH:4]",
        "detected_site_count": 1,
        "protonation_state_enumeration_truncated": False,
        "incomplete_site_ids_json": "[]",
        "sites_json": json.dumps([site]),
        "marvin_values_used": False,
        "epik_values_used": False,
    }


def test_stage1_training_aggregates_replicates_without_target_leakage_features():
    training, quarantine, funnel = build_stage1_training_sites(
        pd.DataFrame([_network_row("mol_1", "CC(=O)O", [4.0, 4.4, 4.8])])
    )
    assert quarantine.empty
    assert funnel["accepted_training_sites"] == 1
    row = training.iloc[0]
    assert row["target_definition"] == TARGET_DEFINITION
    assert row["pka_value"] == pytest.approx(4.4)
    assert row["experimental_measurement_count"] == 3
    assert row["experimental_measurement_range"] == pytest.approx(0.8)
    assert row["measurement_method"] == "experimental"
    assert row["atom_index_raw"] == 1
    assert row["sample_weight"] == pytest.approx(1.0)


def test_stage1_accepts_measured_edge_on_one_serial_physical_coordinate():
    row = _network_row(
        "mol_pyrazole", "c1cn[nH]c1", [2.49],
        family="pyrazole_basicity", label="pyrazole_basicity",
    )
    basicity = json.loads(row["sites_json"])[0]
    basicity["coupling_group_id"] = "group_1"
    acidity = {
        "site_id": "site_2",
        "label": "pyrazole_acidity",
        "family": "pyrazole_acidity",
        "center_maps": [4],
        "coupling_group_id": "group_1",
        "experimental_measurements": [],
    }
    row["detected_site_count"] = 2
    row["sites_json"] = json.dumps([basicity, acidity])

    training, quarantine, funnel = build_stage1_training_sites(pd.DataFrame([row]))

    assert quarantine.empty
    assert list(training["candidate_label"]) == ["pyrazole_basicity"]
    assert training.iloc[0]["pka_value"] == pytest.approx(2.49)
    assert funnel["single_coordinate_molecules"] == 1
    assert funnel["single_coordinate_transition_candidates"] == 1


def test_stage1_training_quarantines_measurements_that_are_not_replicates():
    rows = [
        _network_row("mol_good", "CC(=O)O", [4.5, 4.7]),
        _network_row("mol_bad", "CCC(=O)O", [2.0, 8.0]),
    ]
    training, quarantine, funnel = build_stage1_training_sites(
        pd.DataFrame(rows), max_replicate_range=2.0
    )
    assert list(training["molecule_id"]) == ["mol_good"]
    assert list(quarantine["molecule_id"]) == ["mol_bad"]
    assert quarantine.iloc[0]["quarantine_reason"] == "replicate_range_exceeds_threshold"
    assert funnel["quarantined_training_sites"] == 1


def test_stage1_source_rejects_proprietary_labels():
    row = _network_row("mol_1", "CC(=O)O", [4.7])
    row["marvin_values_used"] = True
    with pytest.raises(ValueError, match="Marvin"):
        build_stage1_training_sites(pd.DataFrame([row]))


def test_stage1_training_quarantines_targets_outside_supported_scale():
    row = _network_row("mol_1", "CC(=O)O", [30.0])
    training, quarantine, _ = build_stage1_training_sites(pd.DataFrame([row]))
    assert training.empty
    assert quarantine.iloc[0]["quarantine_reason"] == "target_outside_supported_pka_range"


def test_stage1_quarantines_apparently_single_site_molecule_with_unresolved_context():
    row = _network_row("mol_1", "CNC(=S)NCCCCc1cc[nH]c1", [7.5], label="pyrrole", family="amine")
    row["unresolved_ionizable_context_count"] = 1
    row["unresolved_ionizable_contexts_json"] = '[{"label":"thioamide"}]'
    training, quarantine, _ = build_stage1_training_sites(pd.DataFrame([row]))
    assert training.empty
    assert quarantine.iloc[0]["quarantine_reason"] == "competing_unresolved_ionizable_context"


def test_stage1_quarantines_single_site_incomplete_microstate_network():
    row = _network_row("mol_1", "c1nnn[nH]1", [4.9], label="tetrazole", family="tetrazole_tetrazolate")
    row["incomplete_site_ids_json"] = json.dumps(["site_1"])
    training, quarantine, _ = build_stage1_training_sites(pd.DataFrame([row]))
    assert training.empty
    assert quarantine.iloc[0]["quarantine_reason"] == "incomplete_or_truncated_microstate_network"


def test_scaffold_groups_hold_ring_scaffolds_together_but_not_all_acyclics():
    assert scaffold_group_key("Cc1ccccc1") == scaffold_group_key("Oc1ccccc1")
    assert scaffold_group_key("CC(=O)O") != scaffold_group_key("CCC(=O)O")


def test_current_canonical_bundle_requires_full_deployment_refit_provenance(tmp_path):
    path = tmp_path / "bad_stage1.pkl"
    with path.open("wb") as handle:
        pickle.dump({
            "model": None, "feature_columns": [], "fp_bits": 32, "fp_radius": 2,
            "training_measurement_method": "experimental",
            "stage1_schema_version": "2.2.0",
            "training_rows": 10, "deployment_fit_rows": 9,
            "trained_on_all_eligible_rows": False,
        }, handle)
    with pytest.raises(ValueError, match="not fit on all eligible rows"):
        load_stage1_bundle(str(path))


def test_reference_residual_bundle_has_import_stable_pickle_identity(tmp_path):
    features = pd.DataFrame({
        "stage1_reference_prior_present": [1.0, 1.0, 0.0],
        "stage1_reference_pka": [4.0, 10.0, 0.0],
        "feature": [0.0, 1.0, 2.0],
    })
    model = ReferenceResidualRegressor(DecisionTreeRegressor(max_depth=1, random_state=1))
    model.fit(features, [4.2, 9.8, 7.0])
    assert type(model).__module__ in {
        "reference_residual_model", "scripts.reference_residual_model"
    }
    path = tmp_path / "roundtrip.pkl"
    with path.open("wb") as handle:
        pickle.dump(model, handle)
    with path.open("rb") as handle:
        restored = pickle.load(handle)
    assert restored.predict(features).shape == (3,)


def test_stage1_scalar_helpers_and_invalid_structure_keys_are_stable():
    assert canonical_structure_key("bad") == canonical_structure_key("bad")
    assert canonical_structure_key("bad").startswith("invalid::")
    assert scaffold_group_key("bad").startswith("invalid::")
    assert _truthy(" YES ") is True
    assert _truthy("no") is False
    assert _truthy(None, default=True) is True
    assert _truthy(float("nan"), default=True) is True
    assert _safe_int("4") == 4
    assert _safe_int("bad", 7) == 7
    assert _safe_int(np.nan, 8) == 8
    assert 0.1 <= _measurement_weight(1000, 0.0, -1.0, "gold") <= 1.5
    assert _measurement_weight(1, 100.0, 5.0, "unknown") == pytest.approx(0.1)


def test_independent_measurements_collapses_copies_and_prefers_higher_tier():
    measurements = [
        {"experimental_pka": 4.2, "source_document_id": "doc", "source_assay_id": "a", "evidence_tier": "silver"},
        {"experimental_pka": 4.2, "source_document_id": "doc", "source_assay_id": "a", "evidence_tier": "gold"},
        {"experimental_pka": 4.2, "source_file": "copied-one.sdf", "evidence_tier": "silver"},
        {"experimental_pka": 4.2, "source_file": "copied-two.sdf", "evidence_tier": "gold"},
    ]
    retained = _independent_measurements(measurements)
    assert len(retained) == 2
    assert all(row["evidence_tier"] == "gold" for row in retained)


def test_stage1_rejects_missing_columns_and_epik_provenance():
    with pytest.raises(ValueError, match="missing Stage 1 source"):
        build_stage1_training_sites(pd.DataFrame({"molecule_id": ["x"]}))
    row = _network_row("mol", "CC(=O)O", [4.7])
    row["epik_values_used"] = True
    with pytest.raises(ValueError, match="Epik"):
        build_stage1_training_sites(pd.DataFrame([row]))


def test_stage1_skips_malformed_and_multicoordinate_network_rows():
    malformed = _network_row("bad-json", "CCO", [7.0])
    malformed["sites_json"] = "not-json"
    multi = _network_row("multi", "NCC(=O)O", [4.0])
    first = json.loads(multi["sites_json"])[0]
    second = {**first, "site_id": "site_2", "coupling_group_id": "group_2"}
    first["coupling_group_id"] = "group_1"
    multi["sites_json"] = json.dumps([first, second])
    training, quarantine, funnel = build_stage1_training_sites(pd.DataFrame([malformed, multi]))
    assert training.empty and quarantine.empty
    assert funnel["single_coordinate_molecules"] == 0


def test_stage1_quarantines_unreadable_completeness_and_no_eligible_measurement():
    incomplete = _network_row("incomplete", "CC(=O)O", [4.7])
    incomplete["incomplete_site_ids_json"] = "not-json"
    ineligible = _network_row("ineligible", "CCC(=O)O", [4.8])
    site = json.loads(ineligible["sites_json"])[0]
    site["experimental_measurements"][0]["stage1_local_supervision_eligible"] = False
    ineligible["sites_json"] = json.dumps([site])
    training, quarantine, _ = build_stage1_training_sites(pd.DataFrame([incomplete, ineligible]))
    assert training.empty
    assert set(quarantine["quarantine_reason"]) == {
        "incomplete_or_truncated_microstate_network",
        "no_finite_experimental_measurement",
    }


def test_stage1_accepts_silver_without_atom_map_and_records_medium_confidence():
    row = _network_row("silver", "CC(=O)O", [4.6])
    site = json.loads(row["sites_json"])[0]
    site["center_maps"] = []
    measurement = site["experimental_measurements"][0]
    measurement.update({
        "evidence_tier": "silver",
        "structural_site_assignment_confidence": "medium",
        "source_document_id": "doc",
        "source_assay_id": "assay",
        "source_site_evidence": "manual",
        "source_site_evidence_confidence": "medium",
    })
    row["sites_json"] = json.dumps([site])
    training, quarantine, _ = build_stage1_training_sites(pd.DataFrame([row]))
    assert quarantine.empty
    result = training.iloc[0]
    assert result["experimental_evidence_tier"] == "silver"
    assert result["experimental_medium_confidence_fraction"] == pytest.approx(1.0)
    assert pd.isna(result["atom_index_raw"])


def test_load_stage1_training_sites_enforces_declared_provenance(tmp_path):
    row = _network_row("mol", "CC(=O)O", [4.7])
    row["stage1_training_measurement_method"] = "experimental"
    path = tmp_path / "network.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    training, quarantine, funnel = load_stage1_training_sites(str(path))
    assert len(training) == 1 and quarantine.empty
    assert funnel["accepted_training_sites"] == 1

    row["stage1_training_measurement_method"] = "marvin"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="not experimental-only"):
        load_stage1_training_sites(str(path))
