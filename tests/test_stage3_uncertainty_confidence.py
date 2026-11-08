import hashlib
import json
import math
import pickle

import pandas as pd
import pytest

from stage3_apply_microstate_inference import (
    CONFIDENCE_TIER_THRESHOLDS,
    PROTONATED_STATE_DECISION_BOUNDARY,
    apply_stage3,
    _confidence_tier,
    _contextual_edge_pka_for_background,
    _predicted_site_form,
    _protonated_probability,
    _site_state_confidence,
    _uncertainty_aware_state_confidence,
)
from stage2_free_energy_coupling import (
    STAGE2_FREE_ENERGY_METHOD,
    STAGE2_FREE_ENERGY_SCHEMA_VERSION,
    contextual_edge_pkas,
    macro_pka_values_from_free_energy,
    pair_coupling_map,
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


@pytest.mark.parametrize(
    ("ph_minus_pka", "expected_probability", "expected_form"),
    [
        (-1.0, 10.0 / 11.0, "acid_form"),
        (-1e-9, 0.5, "acid_form"),
        (0.0, 0.5, "acid_form"),
        (1e-9, 0.5, "base_form"),
        (1.0, 1.0 / 11.0, "base_form"),
    ],
)
def test_henderson_hasselbalch_boundary_and_site_call_are_consistent(
    ph_minus_pka,
    expected_probability,
    expected_form,
):
    probability = _protonated_probability(7.4, 7.4 + ph_minus_pka)
    assert probability == pytest.approx(expected_probability, abs=1e-9)
    assert _predicted_site_form(probability) == expected_form


def test_henderson_hasselbalch_probability_is_symmetric_around_pka():
    for offset in (0.1, 0.5, 1.0, 3.0):
        below = _protonated_probability(7.4, 7.4 - offset)
        above = _protonated_probability(7.4, 7.4 + offset)
        assert below + above == pytest.approx(1.0)


def test_extreme_ph_values_remain_finite_and_saturate_safely():
    assert _protonated_probability(7.4, -1e6) == pytest.approx(1.0)
    assert _protonated_probability(7.4, 1e6) == pytest.approx(0.0, abs=1e-299)


def test_missing_probability_produces_an_unavailable_site_call():
    assert _predicted_site_form(None) == "unavailable"


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan")])
def test_site_call_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError, match="probability"):
        _predicted_site_form(probability)


def test_interval_touching_boundary_has_zero_robustness():
    boundary = PROTONATED_STATE_DECISION_BOUNDARY
    assert _uncertainty_aware_state_confidence(boundary, 0.9) == 0.0
    assert _uncertainty_aware_state_confidence(0.1, boundary) == 0.0


def test_overall_confidence_is_zero_when_pka_interval_crosses_boundary():
    score, tier, components = _site_state_confidence(
        probability=0.99,
        pka_interval_robustness=0.0,
        empirical_call_confidence=1.0,
        local_pka_confidence="high_empirical_support",
        inverse_gap=0.0,
        network_complete=True,
        pair_coupling_identifiability="no_pair_parameter",
    )
    assert score == 0.0
    assert tier == "very_low"
    assert components["conditional_state_decisiveness"] == pytest.approx(0.98)
    assert components["pka_interval_robustness"] == 0.0


@pytest.mark.parametrize(
    ("support", "expected_factor", "expected_tier"),
    [
        ("high_empirical_support", 1.0, "high"),
        ("medium_empirical_support", 0.80, "high"),
        ("low_empirical_support", 0.55, "medium"),
        ("none", 0.30, "low"),
    ],
)
def test_applicability_factor_scales_overall_confidence(
    support,
    expected_factor,
    expected_tier,
):
    score, tier, components = _site_state_confidence(
        probability=1.0,
        pka_interval_robustness=1.0,
        empirical_call_confidence=1.0,
        local_pka_confidence=support,
        inverse_gap=0.0,
        network_complete=True,
        pair_coupling_identifiability="no_pair_parameter",
    )
    assert score == pytest.approx(expected_factor)
    assert tier == expected_tier
    assert components["stage1_applicability_factor"] == expected_factor


def test_reconstruction_and_identifiability_penalties_are_auditable():
    score, tier, components = _site_state_confidence(
        probability=1.0,
        pka_interval_robustness=1.0,
        empirical_call_confidence=1.0,
        local_pka_confidence="high",
        inverse_gap=1.0,
        network_complete=True,
        pair_coupling_identifiability="regularized_prior_dependent_predicted_macro_ladder",
    )
    assert components["free_energy_reconstruction_factor"] == pytest.approx(
        math.exp(-1.0)
    )
    assert components["pair_coupling_identifiability_factor"] == 0.65
    assert score == pytest.approx(math.exp(-1.0) * 0.65)
    assert tier == "low"


def test_incomplete_network_forces_zero_overall_confidence():
    score, tier, components = _site_state_confidence(
        probability=1.0,
        pka_interval_robustness=1.0,
        empirical_call_confidence=1.0,
        local_pka_confidence="high",
        inverse_gap=0.0,
        network_complete=False,
        pair_coupling_identifiability="unknown",
    )
    assert score == 0.0
    assert tier == "very_low"
    assert components["free_energy_reconstruction_factor"] == 0.0
    assert components["network_complete"] is False


def test_empirical_confidence_at_or_below_chance_contributes_no_evidence():
    for empirical_confidence in (0.0, 0.25, 0.5):
        score, tier, components = _site_state_confidence(
            probability=1.0,
            pka_interval_robustness=1.0,
            empirical_call_confidence=empirical_confidence,
            local_pka_confidence="high",
            inverse_gap=0.0,
            network_complete=True,
            pair_coupling_identifiability="no_pair_parameter",
        )
        assert score == 0.0
        assert tier == "very_low"
        assert components["empirical_pka_side_centered_evidence"] == 0.0


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"probability": float("nan")}, "probability"),
        ({"pka_interval_robustness": -0.01}, "robustness"),
        ({"empirical_call_confidence": 1.01}, "Empirical"),
        ({"inverse_gap": -0.01}, "reconstruction gap"),
    ],
)
def test_confidence_composition_rejects_invalid_components(changes, message):
    values = {
        "probability": 0.9,
        "pka_interval_robustness": 0.8,
        "empirical_call_confidence": 0.9,
        "local_pka_confidence": "high",
        "inverse_gap": 0.0,
        "network_complete": True,
        "pair_coupling_identifiability": "no_pair_parameter",
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        _site_state_confidence(**values)


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [
        (0.0, "very_low"),
        (0.1 - 1e-12, "very_low"),
        (0.1, "low"),
        (0.4 - 1e-12, "low"),
        (0.4, "medium"),
        (0.7 - 1e-12, "medium"),
        (0.7, "high"),
        (1.0, "high"),
    ],
)
def test_confidence_tier_boundaries_are_inclusive(confidence, expected):
    assert _confidence_tier(confidence) == expected


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
def test_confidence_tier_rejects_values_outside_unit_interval(confidence):
    with pytest.raises(ValueError, match="Overall confidence"):
        _confidence_tier(confidence)


def test_confidence_thresholds_remain_ordered_and_publicly_traceable():
    assert CONFIDENCE_TIER_THRESHOLDS == (
        (0.70, "high"),
        (0.40, "medium"),
        (0.10, "low"),
    )


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


def _write_complete_one_site_fixture(tmp_path):
    """Write a minimal complete Stage 2 network for pH-boundary integration tests."""
    model_path = tmp_path / "stage2.pkl"
    with model_path.open("wb") as handle:
        pickle.dump({
            "stage2_schema_version": "2.2.0",
            "stage1_model_sha256": "stage1-hash",
            "stage2_free_energy_schema_version": STAGE2_FREE_ENERGY_SCHEMA_VERSION,
        }, handle)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    site = {
        "site_id": "site_1",
        "label": "primary_amine",
        "family": "amine",
        "input_member_form": "base_form",
        "atom_maps": [2],
        "center_maps": [2],
        "coupling_group_id": "group_1",
        "acid_level": 1,
        "stage1_intrinsic_pka": 7.4,
        "stage1_predicted_pka_ci_low": 6.9,
        "stage1_predicted_pka_ci_high": 7.9,
        "stage2_predicted_macro_pka_ci_low": 6.9,
        "stage2_predicted_macro_pka_ci_high": 7.9,
        "local_pka_confidence": "high_empirical_support",
        "stage2_applied": False,
    }
    nodes = [
        {
            "node_id": "base",
            "protonated_site_count": 0,
            "formal_charge": 0,
            "group_levels": {"group_1": 0},
            "site_forms": {"site_1": "base_form"},
            "reference_atom_mapped_smiles": "[CH3:1][NH2:2]",
            "tautomer_enumeration_truncated": False,
        },
        {
            "node_id": "acid",
            "protonated_site_count": 1,
            "formal_charge": 1,
            "group_levels": {"group_1": 1},
            "site_forms": {"site_1": "acid_form"},
            "reference_atom_mapped_smiles": "[CH3:1][NH3+:2]",
            "tautomer_enumeration_truncated": False,
        },
    ]
    edges = [{
        "edge_id": "edge_1",
        "site_id": "site_1",
        "acid_node_id": "acid",
        "base_node_id": "base",
        "other_site_forms": {},
        "stage2_contextual_edge_pka": 7.4,
    }]
    stage2_steps = [{
        "protonation_step": 1,
        "rank_associated_site_id": "site_1",
        "stage2_predicted_macro_pka": 7.4,
    }]
    free_energy_model = {
        "schema_version": STAGE2_FREE_ENERGY_SCHEMA_VERSION,
        "one_body_terms": [{"site_id": "site_1", "one_body_pka": 7.4}],
        "pair_couplings": [],
        "diagnostics": {
            "status": "optimized",
            "optimizer_success": True,
            "pair_parameter_count": 0,
            "macro_reconstruction_mae": 0.0,
            "macro_reconstruction_max_abs_error": 0.0,
            "pair_coupling_identifiability": "no_pair_parameter",
        },
    }
    input_path = tmp_path / "stage2.csv"
    pd.DataFrame([{
        "molecule_id": "methylamine_boundary",
        "representative_input_smiles": "CN",
        "representative_atom_mapped_smiles": "[CH3:1][NH2:2]",
        "network_confidence": "complete",
        "sites_json": json.dumps([site]),
        "microstate_nodes_json": json.dumps(nodes),
        "microstate_edges_json": json.dumps(edges),
        "stage2_macro_pka_steps_json": json.dumps(stage2_steps),
        "stage2_free_energy_model_json": json.dumps(free_energy_model),
        "stage2_free_energy_schema_version": STAGE2_FREE_ENERGY_SCHEMA_VERSION,
        "stage2_free_energy_method": STAGE2_FREE_ENERGY_METHOD,
        "protonation_state_enumeration_truncated": False,
        "incomplete_site_ids_json": "[]",
        "stage1_model_sha256": "stage1-hash",
        "stage2_model_sha256": model_hash,
        "stage2_schema_version": "2.2.0",
        "marvin_values_used": False,
        "epik_values_used": False,
    }]).to_csv(input_path, index=False)

    residual_fixture = {
        "experimental_anchor_pka": [7.4, 7.4, 7.4, 7.4],
        "site_family": ["amine"] * 4,
    }
    stage1_oof = tmp_path / "stage1_oof.csv"
    pd.DataFrame({
        **residual_fixture,
        "pred_intrinsic_pka": [7.4] * 4,
        "final_group_label": ["primary_amine"] * 4,
    }).to_csv(stage1_oof, index=False)
    stage2_eval = tmp_path / "stage2_eval.csv"
    pd.DataFrame({
        **residual_fixture,
        "stage2_supported_projected_macro_pka": [7.4] * 4,
        "site_label": ["primary_amine"] * 4,
    }).to_csv(stage2_eval, index=False)
    return model_path, input_path, stage1_oof, stage2_eval


def _apply_one_site_at_ph(paths, tmp_path, ph):
    model_path, input_path, stage1_oof, stage2_eval = paths
    suffix = str(ph).replace("-", "minus_").replace(".", "_")
    out_dir = tmp_path / f"stage3_{suffix}"
    _, report = apply_stage3(
        input_path=str(input_path),
        stage2_model_path=str(model_path),
        output_path=str(tmp_path / f"stage3_{suffix}.csv"),
        out_dir=str(out_dir),
        ph=float(ph),
        stage1_oof_path=str(stage1_oof),
        stage2_eval_path=str(stage2_eval),
        stage2_quarantine_path=str(tmp_path / "missing_quarantine.csv"),
    )
    return pd.read_csv(out_dir / "stage3_site_predictions.csv").iloc[0], report


def test_complete_stage3_switches_state_and_confidence_across_pka(tmp_path):
    paths = _write_complete_one_site_fixture(tmp_path)
    below, below_report = _apply_one_site_at_ph(paths, tmp_path, 5.4)
    boundary, boundary_report = _apply_one_site_at_ph(paths, tmp_path, 7.4)
    above, above_report = _apply_one_site_at_ph(paths, tmp_path, 9.4)

    assert below["stage3_predicted_site_form_at_ph"] == "acid_form"
    assert boundary["stage3_predicted_site_form_at_ph"] == "acid_form"
    assert above["stage3_predicted_site_form_at_ph"] == "base_form"
    assert below["stage3_protonated_probability_at_ph"] > 0.99
    assert boundary["stage3_protonated_probability_at_ph"] == pytest.approx(0.5)
    assert above["stage3_protonated_probability_at_ph"] < 0.01
    assert (
        below["stage3_protonated_probability_at_ph"]
        + above["stage3_protonated_probability_at_ph"]
    ) == pytest.approx(1.0)

    assert boundary["stage3_site_state_decision_confidence"] == pytest.approx(0.0)
    assert bool(boundary["stage3_uncertainty_crosses_state_boundary"]) is True
    assert boundary["stage3_uncertainty_aware_site_state_confidence"] == 0.0
    assert boundary["stage3_overall_site_state_confidence"] == 0.0
    assert boundary["stage3_overall_site_state_confidence_tier"] == "very_low"
    assert below["stage3_overall_site_state_confidence"] > 0.70
    assert above["stage3_overall_site_state_confidence"] == pytest.approx(
        below["stage3_overall_site_state_confidence"]
    )
    assert below["stage3_overall_site_state_confidence_tier"] == "high"
    assert above["stage3_overall_site_state_confidence_tier"] == "high"

    assert below_report["robust_site_state_calls_pka_interval_does_not_cross_half"] == 1
    assert above_report["robust_site_state_calls_pka_interval_does_not_cross_half"] == 1
    assert boundary_report["site_state_intervals_crossing_half"] == 1


def test_protein_environment_transfer_scales_final_site_confidence(tmp_path):
    paths = _write_complete_one_site_fixture(tmp_path)
    baseline, _ = _apply_one_site_at_ph(paths, tmp_path, 5.4)
    baseline_confidence = float(baseline["stage3_overall_site_state_confidence"])

    input_path = paths[1]
    table = pd.read_csv(input_path)
    site = json.loads(table.loc[0, "sites_json"])[0]
    site.update({
        "protein_environment_applied": True,
        "protein_environment_chemistry": "amine",
        "protein_environment_delta_pka": 0.4,
        "protein_environment_delta_pka_p90_error": 2.0,
        "protein_environment_confidence_score": 0.4,
        "protein_environment_confidence_tier": "medium",
        "protein_environment_transfer_basis": "lysine_amine_analogue",
        "protein_environment_transfer_shrinkage": 0.65,
        "protein_environment_descriptors": {"protein_heavy_atoms_4a": 8.0},
        "aqueous_stage2_one_body_pka": 7.0,
        "aqueous_stage2_predicted_macro_pka": 7.0,
    })
    table.loc[0, "sites_json"] = json.dumps([site])
    table.to_csv(input_path, index=False)

    corrected, _ = _apply_one_site_at_ph(paths, tmp_path, 5.4)
    assert corrected["stage3_overall_site_state_confidence"] == pytest.approx(
        baseline_confidence * 0.4
    )
    assert bool(corrected["protein_environment_applied"]) is True
    assert corrected["protein_environment_delta_pka"] == pytest.approx(0.4)
    assert json.loads(corrected["protein_environment_descriptors_json"])[
        "protein_heavy_atoms_4a"
    ] == 8.0


def test_stage3_jointly_populates_ligand_and_explicit_protein_site(tmp_path):
    paths = _write_complete_one_site_fixture(tmp_path)
    table = pd.read_csv(paths[1])
    row = table.iloc[0]
    ligand_site = json.loads(row["sites_json"])[0]
    protein_site = {
        "site_id": "protein_A_ASP_42",
        "component_type": "protein_residue",
        "family": "protein_residue",
        "label": "protein_asp",
        "input_member_form": "unknown",
        "atom_maps": [],
        "center_maps": [],
        "coupling_group_id": "protein_A_ASP_42",
        "acid_level": 1,
        "base_level": 0,
        "protein_chain": "A",
        "protein_residue_name": "ASP",
        "protein_residue_id": "42",
        "protein_insertion_code": "",
        "protein_nearest_ligand_site_distance_angstrom": 4.0,
        "stage1_intrinsic_pka": 3.9,
        "stage1_predicted_pka_ci_low": 2.9,
        "stage1_predicted_pka_ci_high": 4.9,
        "stage2_predicted_macro_pka": 3.9,
        "stage2_predicted_macro_pka_ci_low": 2.9,
        "stage2_predicted_macro_pka_ci_high": 4.9,
        "stage2_applied": True,
        "stage2_prediction_confidence": "medium",
        "local_pka_confidence": "medium_protein_residue_calibration",
        "protein_environment_applied": True,
        "protein_environment_evaluated": True,
        "protein_environment_chemistry": "carboxyl",
        "protein_environment_delta_pka": 0.0,
        "protein_environment_delta_pka_p90_error": 1.0,
        "protein_environment_confidence_score": 0.8,
        "protein_environment_confidence_tier": "medium",
        "protein_environment_descriptors": {},
    }
    original_nodes = json.loads(row["microstate_nodes_json"])
    nodes = []
    node_ids = {}
    for protein_level in (0, 1):
        for original in original_nodes:
            node = dict(original)
            origin = original["node_id"]
            node_id = f"{origin}__protein_{protein_level}"
            node_ids[(origin, protein_level)] = node_id
            node["node_id"] = node_id
            node["ligand_origin_node_id"] = origin
            node["site_forms"] = {
                **original["site_forms"],
                "protein_A_ASP_42": (
                    "acid_form" if protein_level else "base_form"
                ),
            }
            node["group_levels"] = {
                **original["group_levels"],
                "protein_A_ASP_42": protein_level,
            }
            node["protonated_site_count"] = (
                original["protonated_site_count"] + protein_level
            )
            nodes.append(node)
    edges = []
    for protein_level in (0, 1):
        edges.append({
            "edge_id": f"ligand_edge_{protein_level}",
            "site_id": "site_1",
            "acid_node_id": node_ids[("acid", protein_level)],
            "base_node_id": node_ids[("base", protein_level)],
            "other_site_forms": {"protein_A_ASP_42": protein_level},
        })
    for ligand_origin in ("base", "acid"):
        edges.append({
            "edge_id": f"protein_edge_{ligand_origin}",
            "site_id": "protein_A_ASP_42",
            "acid_node_id": node_ids[(ligand_origin, 1)],
            "base_node_id": node_ids[(ligand_origin, 0)],
            "other_site_forms": {
                "site_1": int(ligand_origin == "acid")
            },
        })
    sites = [ligand_site, protein_site]
    one_body = {"site_1": 7.4, "protein_A_ASP_42": 3.9}
    pair_terms = [{
        "site_i": "site_1",
        "site_j": "protein_A_ASP_42",
        "coupling_log10_units": -0.2,
        "source": "explicit_nearby_binary_residue_nodes_with_centered_screened_coulomb_coupling",
    }]
    pair_map = pair_coupling_map(pair_terms)
    annotations, _ = contextual_edge_pkas(
        nodes, edges, sites, one_body, pair_map
    )
    for edge in edges:
        edge["stage2_contextual_edge_pka"] = annotations[
            edge["edge_id"]
        ]["contextual_edge_pka"]
    macros = macro_pka_values_from_free_energy(nodes, sites, one_body, pair_map)
    steps = [
        {
            "protonation_step": index,
            "rank_associated_site_id": site_id,
            "stage2_predicted_macro_pka": macro,
        }
        for index, (site_id, macro) in enumerate(
            zip(("site_1", "protein_A_ASP_42"), macros), start=1
        )
    ]
    free_energy = {
        "schema_version": STAGE2_FREE_ENERGY_SCHEMA_VERSION,
        "one_body_terms": [
            {"site_id": site_id, "one_body_pka": pka}
            for site_id, pka in one_body.items()
        ],
        "pair_couplings": pair_terms,
        "diagnostics": {
            "status": "optimized",
            "optimizer_success": True,
            "pair_parameter_count": 1,
            "macro_reconstruction_mae": 0.0,
            "macro_reconstruction_max_abs_error": 0.0,
            "pair_coupling_identifiability": (
                "heuristic_centered_screened_coulomb_not_experimentally_identified"
            ),
        },
    }
    table.loc[0, "sites_json"] = json.dumps(sites)
    table.loc[0, "microstate_nodes_json"] = json.dumps(nodes)
    table.loc[0, "microstate_edges_json"] = json.dumps(edges)
    table.loc[0, "stage2_macro_pka_steps_json"] = json.dumps(steps)
    table.loc[0, "stage2_free_energy_model_json"] = json.dumps(free_energy)
    table.to_csv(paths[1], index=False)

    model_path, input_path, stage1_oof, stage2_eval = paths
    out_dir = tmp_path / "joint_stage3"
    result, report = apply_stage3(
        input_path=str(input_path),
        stage2_model_path=str(model_path),
        output_path=str(tmp_path / "joint_stage3.csv"),
        out_dir=str(out_dir),
        ph=7.4,
        stage1_oof_path=str(stage1_oof),
        stage2_eval_path=str(stage2_eval),
        stage2_quarantine_path=str(tmp_path / "missing_quarantine.csv"),
    )
    site_rows = pd.read_csv(out_dir / "stage3_site_predictions.csv")
    protein_row = site_rows[
        site_rows["component_type"] == "protein_residue"
    ].iloc[0]

    assert len(site_rows) == 2
    assert protein_row["protein_residue_name"] == "ASP"
    assert protein_row["stage3_empirical_calibration_scope"] == (
        "direct_standard_protein_residue"
    )
    components = json.loads(protein_row["stage3_confidence_components_json"])
    assert components["pair_coupling_identifiability_factor"] == 0.55
    assert report["joint_protein_site_rows"] == 1
    assert result.iloc[0]["stage3_joint_protein_site_count"] == 1
    assert set(json.loads(
        result.iloc[0]["stage3_dominant_protein_site_forms_json"]
    )) == {"protein_A_ASP_42"}


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
