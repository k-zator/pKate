import hashlib
import json
import pickle

import pandas as pd
import pytest

from stage3_apply_microstate_inference import (
    apply_stage3,
    _contextual_edge_pka_for_background,
    _protonated_probability,
    _uncertainty_aware_state_confidence,
)


def test_pka_interval_crossing_ph_has_zero_robust_state_confidence():
    low = _protonated_probability(6.5, 7.4)
    high = _protonated_probability(8.5, 7.4)
    assert low < 0.5 < high
    assert _uncertainty_aware_state_confidence(low, high) == 0.0


def test_pka_interval_far_above_ph_retains_robust_protonated_call():
    low = _protonated_probability(10.0, 7.4)
    high = _protonated_probability(12.0, 7.4)
    confidence = _uncertainty_aware_state_confidence(low, high)
    assert low > 0.99
    assert confidence > 0.98


def test_henderson_hasselbalch_probability_is_half_at_pka():
    assert _protonated_probability(7.4, 7.4) == pytest.approx(0.5)


def test_contextual_edge_pka_selects_the_requested_other_site_background():
    edges = [
        {
            "site_id": "site_1",
            "other_site_forms": {"group_2": 0},
            "stage3_contextual_edge_pka": 4.2,
        },
        {
            "site_id": "site_1",
            "other_site_forms": {"group_2": 1},
            "stage3_contextual_edge_pka": 5.1,
        },
    ]
    assert _contextual_edge_pka_for_background(
        edges, "site_1", {"group_1": 0, "group_2": 1}
    ) == pytest.approx(5.1)


def test_incomplete_network_retains_every_site_as_unavailable(tmp_path):
    model_path = tmp_path / "stage2.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({
            "stage2_schema_version": "2.2.0",
            "stage1_model_sha256": "stage1-hash",
            "stage2_free_energy_schema_version": "1.0.0",
        }, handle)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    site = {
        "site_id": "site_1",
        "label": "tetrazole",
        "family": "tetrazole_tetrazolate",
        "input_member_form": "acid_form",
        "atom_maps": [1, 2, 3, 4],
        "center_maps": [1, 2, 3, 4],
        "stage1_intrinsic_pka": 4.9,
        "stage1_predicted_pka_ci_low": 2.9,
        "stage1_predicted_pka_ci_high": 6.9,
        "local_pka_confidence": "low",
        "stage2_applied": False,
    }
    input_path = tmp_path / "stage2.csv"
    pd.DataFrame([{
        "molecule_id": "mol_limited",
        "representative_input_smiles": "c1nnn[nH]1",
        "representative_atom_mapped_smiles": "[cH:1]1[n:2][n:3][n:4][nH:5]1",
        "network_confidence": "limited",
        "sites_json": json.dumps([site]),
        "microstate_nodes_json": "[]",
        "microstate_edges_json": "[]",
        "stage2_macro_pka_steps_json": "[]",
        "stage2_free_energy_model_json": json.dumps({
            "schema_version": "1.0.0",
            "one_body_terms": [],
            "pair_couplings": [],
            "diagnostics": {"status": "unavailable_incomplete_or_truncated_network"},
        }),
        "stage2_free_energy_schema_version": "1.0.0",
        "stage2_free_energy_method": "minimum_norm_pairwise_closure_of_stage2_predicted_macro_ladder",
        "protonation_state_enumeration_truncated": False,
        "incomplete_site_ids_json": json.dumps(["site_1"]),
        "stage1_model_sha256": "stage1-hash",
        "stage2_model_sha256": model_hash,
        "stage2_schema_version": "2.2.0",
        "marvin_values_used": False,
        "epik_values_used": False,
    }]).to_csv(input_path, index=False)

    validation_rows = pd.DataFrame({
        "experimental_anchor_pka": [4.0, 5.0],
        "pred_intrinsic_pka": [4.1, 4.9],
    })
    stage1_oof = tmp_path / "stage1_oof.csv"
    validation_rows.to_csv(stage1_oof, index=False)
    stage2_eval = tmp_path / "stage2_eval.csv"
    pd.DataFrame({
        "experimental_anchor_pka": [4.0, 5.0],
        "stage2_supported_projected_macro_pka": [4.2, 4.8],
    }).to_csv(stage2_eval, index=False)
    out_dir = tmp_path / "stage3"

    result, report = apply_stage3(
        input_path=str(input_path),
        stage2_model_path=str(model_path),
        output_path=str(tmp_path / "stage3.csv"),
        out_dir=str(out_dir),
        stage1_oof_path=str(stage1_oof),
        stage2_eval_path=str(stage2_eval),
        stage2_quarantine_path=str(tmp_path / "missing_quarantine.csv"),
    )

    site_rows = pd.read_csv(out_dir / "stage3_site_predictions.csv")
    assert len(result) == 1
    assert len(site_rows) == 1
    assert report["unavailable_incomplete_network_molecules"] == 1
    assert site_rows.iloc[0]["population_status"] == "unavailable_incomplete_or_truncated_network"
    assert site_rows.iloc[0]["stage3_predicted_site_form_at_ph"] == "unavailable"
    assert pd.isna(site_rows.iloc[0]["stage2_predicted_macro_pka"])
    assert site_rows.iloc[0]["stage3_effective_local_pka_semantics"] == (
        "deprecated_compatibility_alias_of_stage3_one_body_pka"
    )
    assert site_rows.iloc[0]["stage3_effective_local_pka"] == pytest.approx(
        site_rows.iloc[0]["stage3_one_body_pka"]
    )
    assert (out_dir / "stage3_output_schema.json").is_file()
    assert (out_dir / "stage3_ranked_tautomers.csv").is_file()
    assert (out_dir / "stage3_empirical_call_calibration.json").is_file()
    assert pd.isna(site_rows.iloc[0]["stage3_empirical_pka_side_call_confidence"])
