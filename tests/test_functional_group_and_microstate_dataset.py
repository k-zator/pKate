"""Integration tests for functional-group assignment and microstate datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

import build_molecule_microstate_network_dataset as molecule_builder
import functional_group_pka_analysis as groups
from build_release_microstate_dataset import build_dataset as build_release_dataset
from substructure_match import (
    FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION,
    find_sites_with_metadata,
    resolve_overlapping_sites,
)


def _write_glycine_measurements(path: Path) -> Path:
    """Write two physical transitions plus one exact source duplicate."""
    records = [
        ("NCC(=O)O", 2.34, "acidic", "acid measurement"),
        ("[NH3+]CC(=O)[O-]", 9.60, "basic", "base measurement"),
        ("NCC(=O)O", 2.34, "acidic", "duplicate acid measurement"),
    ]
    writer = Chem.SDWriter(str(path))
    for smiles, pka, pka_type, name in records:
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        molecule.SetProp("SMILES", smiles)
        molecule.SetProp("pKa", str(pka))
        molecule.SetProp("type", pka_type)
        molecule.SetProp("name", name)
        molecule.SetProp("assay_web_id", f"assay-{pka_type}")
        molecule.SetProp("document_web_id", "document-1")
        molecule.SetProp("chembl_id", "synthetic-glycine")
        molecule.SetProp("temp", "298.15")
        writer.write(molecule)
    writer.close()
    return path


@pytest.fixture(scope="module")
def glycine_pipeline(tmp_path_factory):
    root = tmp_path_factory.mktemp("functional-microstate")
    raw_dir = root / "raw"
    raw_dir.mkdir()
    sdf_path = _write_glycine_measurements(raw_dir / "glycine.sdf")

    assignments = groups.build_functional_group_table(
        raw_dir=str(raw_dir),
        include_files=[sdf_path.name],
        allow_epik=False,
    )
    assignments_path = root / "assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    release_path = root / "transitions.csv"
    release_quarantine_path = root / "transition-quarantine.csv"
    release, release_quarantine, clean_overrides, override_quarantine = (
        build_release_dataset(
            raw_dir=str(raw_dir),
            assignments_path=str(assignments_path),
            override_paths=[],
            output_path=str(release_path),
            quarantine_path=str(release_quarantine_path),
            clean_overrides_path=str(root / "clean-overrides.csv"),
            override_quarantine_path=str(root / "override-quarantine.csv"),
            max_tautomers=2,
            measurement_exclusions_path=None,
            outlier_review_decisions_path=None,
            structure_review_decisions_path=None,
        )
    )
    return {
        "root": root,
        "raw_dir": raw_dir,
        "assignments": assignments,
        "assignments_path": assignments_path,
        "release": release,
        "release_path": release_path,
        "release_quarantine": release_quarantine,
        "clean_overrides": clean_overrides,
        "override_quarantine": override_quarantine,
    }


def _install_stage1_stubs(monkeypatch, *, omit_labels=()):
    bundle = {
        "feature_columns": [
            "group_primary_amine",
            "group_primary_ammonium",
            "group_carboxylic_acid",
            "group_carboxylate",
        ],
        "training_measurement_method": "experimental",
        "training_rows": 20,
        "training_measurement_count": 20,
        "trained_on_all_eligible_rows": True,
        "stage1_schema_version": "test-stage1",
        "target_definition": "synthetic intrinsic pKa",
        "error_calibration_global": {
            "rows": 20,
            "mae": 0.5,
            "p90_abs_error": 0.8,
            "p95_abs_error": 1.0,
        },
    }
    monkeypatch.setattr(
        molecule_builder, "load_stage1_bundle", lambda *_args, **_kwargs: bundle
    )

    def predict(tasks, _path, _workers):
        result = {}
        for task in tasks:
            label = str(task["candidate_label"])
            if label in omit_labels:
                continue
            result[task["task_id"]] = (
                4.5 if "carbox" in label else 9.5
            )
        return result

    monkeypatch.setattr(molecule_builder, "predict_stage1_tasks", predict)
    monkeypatch.setattr(
        molecule_builder,
        "stage1_applicability",
        lambda *_args, **_kwargs: {
            "stage1_exact_label_training_rows": 10,
            "stage1_nearest_same_label_tanimoto": 1.0,
            "stage1_applicability_domain": "interpolation_high_similarity",
        },
    )
    return bundle


def _build_molecule_rows(
    pipeline,
    monkeypatch,
    *,
    max_protonation_states=64,
    omit_labels=(),
):
    root = pipeline["root"]
    model_path = root / f"stage1-{max_protonation_states}-{'missing' if omit_labels else 'full'}.pkl"
    model_path.write_bytes(b"synthetic-stage1-bundle")
    metrics_path = root / "stage1-metrics.json"
    metrics_path.write_text(json.dumps({"scaffold": {"mae": 0.75}}))
    _install_stage1_stubs(monkeypatch, omit_labels=omit_labels)
    return molecule_builder.build_molecule_dataset(
        transitions_path=str(pipeline["release_path"]),
        stage1_model_path=str(model_path),
        stage1_metrics_path=str(metrics_path),
        output_path=str(root / f"molecules-{max_protonation_states}.csv"),
        quarantine_path=str(root / f"molecule-quarantine-{max_protonation_states}.csv"),
        max_protonation_states=max_protonation_states,
        max_tautomers_per_state=2,
        stage1_workers=1,
    )


def test_functional_group_assignment_selects_the_reported_transition_endpoint(
    glycine_pipeline,
):
    assignments = glycine_pipeline["assignments"]

    assert len(assignments) == 2
    assert set(assignments["pka_source_method"]) == {"experimental"}
    assert assignments["source_exact_duplicate_count"].tolist() == [2, 1]
    acid = assignments.loc[assignments["pka_type_canonical"] == "acidic"].iloc[0]
    base = assignments.loc[assignments["pka_type_canonical"] == "basic"].iloc[0]

    assert acid["selected_group_label"] == "carboxylic_acid"
    assert acid["selected_group_family"] == "carboxyl"
    assert acid["pair_member_form"] == "acid_form"
    assert acid["source_exact_duplicate_record_indices"] == "0|2"
    assert base["selected_group_label"] == "primary_ammonium"
    assert base["selected_group_family"] == "amine"
    assert base["pair_member_form"] == "acid_form"
    assert bool(base["site_override_applied"]) is False
    assert assignments["resolved_group_count"].ge(2).all()
    assert assignments["assignment_status"].eq("multi_group").all()


def test_release_dataset_preserves_atom_mapped_conjugates_and_provenance(
    glycine_pipeline,
):
    release = glycine_pipeline["release"]

    assert len(release) == 2
    assert glycine_pipeline["release_quarantine"].empty
    assert glycine_pipeline["clean_overrides"].empty
    assert glycine_pipeline["override_quarantine"].empty
    assert set(release["measurement_method"]) == {"experimental"}
    assert set(release["site_family"]) == {"amine", "carboxyl"}
    assert release["site_atom_maps_json"].map(json.loads).map(bool).all()
    assert release["reference_acid_atom_mapped_smiles"].str.contains(":").all()
    assert release["reference_base_atom_mapped_smiles"].str.contains(":").all()
    assert release["acid_microstate_count"].ge(1).all()
    assert release["base_microstate_count"].ge(1).all()
    assert release["source_assay_id"].str.startswith("assay-").all()
    assert release["source_document_id"].eq("document-1").all()
    assert release["marvin_values_used"].eq(False).all()
    assert release["epik_values_used"].eq(False).all()


def test_molecule_network_attaches_both_experiments_to_exact_sites(
    glycine_pipeline, monkeypatch
):
    dataset, quarantine = _build_molecule_rows(glycine_pipeline, monkeypatch)

    assert quarantine.empty
    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert row["experimental_transition_count"] == 2
    assert row["detected_site_count"] == 2
    assert row["experimentally_anchored_site_count"] == 2
    assert row["stage1_prior_only_site_count"] == 0
    assert row["unattached_experimental_measurement_count"] == 0
    assert row["network_confidence"] == "experimental_anchors_complete_for_detected_sites"
    assert row["marvin_values_used"] == False
    assert row["epik_values_used"] == False

    sites = json.loads(row["sites_json"])
    by_family = {site["family"]: site for site in sites}
    assert by_family["carboxyl"]["experimental_anchor_pka"] == pytest.approx(2.34)
    assert by_family["amine"]["experimental_anchor_pka"] == pytest.approx(9.60)
    assert all(site["local_pka_provenance"] for site in sites)
    assert all(site["experimental_measurements"] for site in sites)

    nodes = json.loads(row["microstate_nodes_json"])
    assert sum(node["network_population_at_ph"] for node in nodes) == pytest.approx(1.0)
    macro_steps = json.loads(row["independent_site_macro_pka_steps_json"])
    assert len(macro_steps) == 2
    assert [step["protonation_step"] for step in macro_steps] == [1, 2]


def test_truncated_network_is_retained_but_explicitly_limited(
    glycine_pipeline, monkeypatch
):
    dataset, quarantine = _build_molecule_rows(
        glycine_pipeline, monkeypatch, max_protonation_states=2
    )

    assert quarantine.empty
    row = dataset.iloc[0]
    assert bool(row["protonation_state_enumeration_truncated"])
    assert row["network_confidence"] == "limited"
    assert json.loads(row["independent_site_macro_pka_steps_json"]) == []


def test_missing_stage1_prediction_cannot_receive_mixed_confidence(
    glycine_pipeline, monkeypatch
):
    dataset, quarantine = _build_molecule_rows(
        glycine_pipeline, monkeypatch, omit_labels={"carboxylic_acid", "carboxylate"}
    )

    assert quarantine.empty
    row = dataset.iloc[0]
    assert row["network_confidence"] == "limited"
    assert json.loads(row["independent_site_macro_pka_steps_json"]) == []
    sites = json.loads(row["sites_json"])
    missing = [site for site in sites if site["local_pka_used"] is None]
    assert missing
    assert all(site["local_pka_provenance"] == "missing_stage1_prediction" for site in missing)


def test_all_unreadable_molecule_groups_produce_quarantine_instead_of_crashing(
    tmp_path, monkeypatch
):
    transitions = pd.DataFrame(
        [
            {
                "dataset_row_id": "bad-1",
                "measurement_method": "experimental",
                "input_smiles": "not-a-smiles",
                "site_family": "amine",
                "source_file": "bad.sdf",
                "record_index": 0,
                "experimental_pka": 7.0,
                "site_label": "primary_amine",
                "site_assignment_confidence": "none",
                "input_atom_mapped_smiles": "not-a-smiles",
                "site_atom_maps_json": "[]",
            }
        ]
    )
    transitions_path = tmp_path / "bad-transitions.csv"
    transitions.to_csv(transitions_path, index=False)
    model_path = tmp_path / "stage1.pkl"
    model_path.write_bytes(b"model")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"scaffold": {"mae": 1.0}}))
    monkeypatch.setattr(
        molecule_builder,
        "load_stage1_bundle",
        lambda *_args, **_kwargs: {
            "feature_columns": ["feature"],
            "training_measurement_method": "experimental",
        },
    )
    monkeypatch.setattr(
        molecule_builder,
        "predict_stage1_tasks",
        lambda tasks, *_args: pytest.fail("quarantined molecules must not be scored"),
    )

    dataset, quarantine = molecule_builder.build_molecule_dataset(
        transitions_path=str(transitions_path),
        stage1_model_path=str(model_path),
        stage1_metrics_path=str(metrics_path),
        output_path=str(tmp_path / "molecules.csv"),
        quarantine_path=str(tmp_path / "quarantine.csv"),
    )

    assert dataset.empty
    assert len(quarantine) == 1
    assert quarantine.iloc[0]["reason"] == "no_readable_representative"
    assert (tmp_path / "molecules.csv").is_file()
    assert (tmp_path / "quarantine.csv").is_file()


def test_site_override_recomputes_the_complete_assignment_contract(
    glycine_pipeline, tmp_path
):
    assignments = glycine_pipeline["assignments"]
    acid = assignments.loc[assignments["pka_type_canonical"] == "acidic"].iloc[0]
    override = {column: acid[column] for column in groups.SITE_OVERRIDE_KEY_COLUMNS}
    override.update(
        {
            "override_group_label": "primary_amine",
            "note": "reviewed alternative site",
        }
    )
    path = tmp_path / "overrides.csv"
    pd.DataFrame([override]).to_csv(path, index=False)

    updated = groups.apply_site_resolution_overrides(
        assignments.copy(), overrides_path=str(path)
    )
    changed = updated.loc[updated["pka_type_canonical"] == "acidic"].iloc[0]

    assert bool(changed["site_override_applied"])
    assert changed["auto_selected_group_label"] == "carboxylic_acid"
    assert changed["selected_group_label"] == "primary_amine"
    assert changed["final_group_family"] == "amine"
    assert changed["pair_member_form"] == "base_form"
    assert changed["conjugate_check_flag"] == "basic_family_vs_acidic_type"
    assert changed["site_override_note"] == "reviewed alternative site"


def test_invalid_override_is_rejected_without_changing_automatic_assignment(
    glycine_pipeline, tmp_path
):
    assignments = glycine_pipeline["assignments"]
    acid = assignments.loc[assignments["pka_type_canonical"] == "acidic"].iloc[0]
    override = {column: acid[column] for column in groups.SITE_OVERRIDE_KEY_COLUMNS}
    override["override_group_label"] = "not_a_detected_group"
    path = tmp_path / "bad-overrides.csv"
    pd.DataFrame([override]).to_csv(path, index=False)

    updated = groups.apply_site_resolution_overrides(
        assignments.copy(), overrides_path=str(path)
    )
    unchanged = updated.loc[updated["pka_type_canonical"] == "acidic"].iloc[0]

    assert not bool(unchanged["site_override_applied"])
    assert unchanged["selected_group_label"] == "carboxylic_acid"
    assert "not a current SMARTS candidate" in unchanged["site_override_note"]


def test_measurement_extraction_keeps_experimental_default_and_epik_opt_in():
    molecule = Chem.MolFromSmiles("CN")
    assert molecule is not None
    molecule.SetProp("pKa", "10.6")
    molecule.SetProp("type", "basic")
    molecule.SetProp("r_epik_pKa_1", "9.8")
    molecule.SetProp("i_epik_pKa_atom_1", "2")
    molecule.SetProp("s_epik_pKa_identifier_1", "basic site")

    experimental = groups.extract_measurements(molecule, allow_epik=False)
    epik = groups.extract_measurements(molecule, allow_epik=True)

    assert experimental == [
        {
            "pka_value": 10.6,
            "pka_source_method": "experimental",
            "pka_type_raw": "basic",
            "atom_index_raw": None,
            "site_identifier_raw": None,
        }
    ]
    assert epik[0]["pka_source_method"] == "epik"
    assert epik[0]["atom_index_raw"] == 2
    assert epik[0]["pka_type_raw"] == "basic"


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "bad", None])
def test_functional_group_numeric_parser_rejects_nonfinite_values(value):
    assert groups._safe_float(value) is None


def test_atom_index_matching_supports_zero_and_one_based_source_conventions():
    molecule = Chem.MolFromSmiles("CN")
    assert molecule is not None
    resolved, _ = resolve_overlapping_sites(find_sites_with_metadata(molecule))

    zero_based, zero_meta = groups._sites_matching_atom_index(resolved, 1)
    one_based, one_meta = groups._sites_matching_atom_index(resolved, 2)

    assert zero_based and zero_meta["atom_index_mode"] == "zero_based"
    assert one_based and one_meta["atom_index_mode"] == "one_based_minus1"
    assert zero_based[0]["label"] == one_based[0]["label"] == "primary_amine"


def test_reported_type_filters_unannotated_sites_but_does_not_erase_atom_evidence():
    molecule = Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")
    assert molecule is not None
    resolved, _ = resolve_overlapping_sites(find_sites_with_metadata(molecule))

    typed_site, _ = groups.select_pair_first_site(
        resolved, preferred_type="basic", pka_value=9.6
    )
    carboxyl = next(site for site in resolved if site["label"] == "carboxylate")
    atom_anchored_site, _ = groups.select_pair_first_site(
        [carboxyl], preferred_type="basic", pka_value=9.6
    )

    assert typed_site["pair_family"] == "amine"
    assert atom_anchored_site["family"] == "carboxyl"


def test_serial_azole_reference_selects_transition_without_inventing_a_site():
    molecule = Chem.MolFromSmiles("c1ncc[nH]1")
    assert molecule is not None
    resolved, _ = resolve_overlapping_sites(find_sites_with_metadata(molecule))
    imidazole = next(site for site in resolved if site["label"] == "imidazole")
    candidates = groups.expand_pair_site_transitions(imidazole)

    basic, basic_reason = groups.select_serial_azole_transition_by_reference(
        candidates, 7.0
    )
    acidic, acidic_reason = groups.select_serial_azole_transition_by_reference(
        candidates, 14.5
    )
    unsupported, unsupported_reason = groups.select_serial_azole_transition_by_reference(
        candidates, 30.0
    )

    assert basic["label"] == "imidazole_basicity"
    assert acidic["label"] == "imidazole_acidity"
    assert "disambiguated" in basic_reason and "disambiguated" in acidic_reason
    assert unsupported is None
    assert unsupported_reason == "serial_azole_pka_conflicts_with_supported_transitions"


def test_assignment_summaries_preserve_repeated_group_instances():
    frame = pd.DataFrame(
        [
            {
                "assignment_status": "multi_group",
                "pka_value": 9.5,
                "resolved_group_count": 2,
                "resolved_groups": "primary_amine",
                "resolved_group_instances": "primary_amine|primary_amine",
            },
            {
                "assignment_status": "single_group",
                "pka_value": 4.5,
                "resolved_group_count": 1,
                "resolved_groups": "carboxylic_acid",
                "resolved_group_instances": "carboxylic_acid",
            },
        ]
    )

    strictness = groups.build_assignment_strictness_summary(frame)
    multi = groups.build_multi_group_subset(frame)
    group_summary = groups.build_multi_group_group_summary(multi)
    combo_summary = groups.build_multi_group_combo_summary(multi)

    assert strictness["n_records"].sum() == 2
    assert strictness["fraction"].sum() == pytest.approx(1.0)
    assert group_summary.iloc[0]["group_label"] == "primary_amine"
    assert group_summary.iloc[0]["count"] == 2
    assert combo_summary.iloc[0]["group_combo"] == "primary_amine|primary_amine"


def test_network_builder_rejects_nonexperimental_or_detector_mismatched_inputs(
    glycine_pipeline, tmp_path, monkeypatch
):
    release = glycine_pipeline["release"].copy()
    release.loc[release.index[0], "measurement_method"] = "epik"
    mixed_path = tmp_path / "mixed.csv"
    release.to_csv(mixed_path, index=False)
    with pytest.raises(ValueError, match="experimental-only"):
        molecule_builder.build_molecule_dataset(
            str(mixed_path), "unused", "unused", "unused", "unused"
        )

    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"model")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"scaffold": {"mae": 1.0}}))
    monkeypatch.setattr(
        molecule_builder,
        "load_stage1_bundle",
        lambda *_args, **_kwargs: {
            "feature_columns": ["feature"],
            "functional_group_detector_schema_versions": ["incompatible"],
        },
    )
    with pytest.raises(ValueError, match="detector provenance"):
        molecule_builder.build_molecule_dataset(
            str(glycine_pipeline["release_path"]),
            str(model_path),
            str(metrics_path),
            str(tmp_path / "out.csv"),
            str(tmp_path / "q.csv"),
        )

    assert set(glycine_pipeline["release"]["functional_group_detector_schema_version"]) == {
        FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION
    }
