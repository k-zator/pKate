import json

import pandas as pd
import pytest

from visualize_canonical_stage3_failures import (
    _deployment_comparison_table_html,
    _experimental_pka_implied_form,
    _mapped_molecule_and_highlights,
    build_pka_failure_table,
    render_gallery,
)


def _stage3_context() -> pd.DataFrame:
    rows = []
    for molecule_id, site_id in (("single", "site_1"), ("multi", "site_2")):
        rows.append({
            "molecule_id": molecule_id,
            "site_id": site_id,
            "input_smiles": "CN",
            "atom_mapped_smiles": "[CH3:1][NH2:2]",
            "input_member_form": "base_form",
            "site_atom_maps_json": "[2]",
            "site_center_maps_json": "[2]",
            "population_ph": 7.4,
            "stage1_intrinsic_pka": 8.5,
            "stage2_predicted_macro_pka": 8.2,
            "stage3_reconstructed_macro_pka": 8.2,
            "stage3_one_body_pka": 8.0,
            "stage3_one_body_pka_interval_low": 7.0,
            "stage3_one_body_pka_interval_high": 9.0,
            "stage3_effective_local_pka": 8.0,
            "stage3_effective_local_pka_ci_low": 7.0,
            "stage3_effective_local_pka_ci_high": 9.0,
            "stage3_contextual_edge_pka_at_dominant_background": 7.8,
            "stage3_contextual_edge_pka_min": 7.5,
            "stage3_contextual_edge_pka_max": 8.4,
            "stage3_protonated_probability_at_ph": 0.8,
            "stage3_predicted_site_form_at_ph": "acid_form",
            "stage3_site_state_decision_confidence": 0.6,
            "population_status": "complete_pairwise_coupled_population",
            "dominant_atom_mapped_smiles": "[CH3:1][NH3+:2]",
            "dominant_configuration_population": 0.8,
        })
    return pd.DataFrame(rows)


def test_combines_stage1_and_stage2_validation_errors_with_stage3_context():
    stage1 = pd.DataFrame([{
        "molecule_id": "single",
        "site_id": "site_1",
        "candidate_label": "amine",
        "site_family": "amine",
        "experimental_anchor_pka": 10.0,
        "pred_intrinsic_pka": 8.5,
    }])
    stage2 = pd.DataFrame([{
        "molecule_id": "multi",
        "site_id": "site_2",
        "site_label": "amine",
        "site_family": "amine",
        "experimental_anchor_pka": 6.0,
        "stage2_supported_projected_macro_pka": 9.0,
    }])

    result = build_pka_failure_table(stage1, stage2, _stage3_context())

    assert list(result["validation_scope"]) == [
        "stage1_scaffold_oof_single_site",
        "stage2_scaffold_validation_multisite",
    ]
    assert list(result["signed_error"]) == [-1.5, 3.0]
    assert list(result["abs_error"]) == [1.5, 3.0]
    assert list(result["deployed_assigned_macro_pka"]) == [8.2, 8.2]
    assert list(result["deployed_assigned_local_pka"]) == [8.0, 8.0]
    assert list(result["deployed_assigned_one_body_pka"]) == [8.0, 8.0]
    assert list(result["deployed_macro_signed_difference_from_experimental"]) == pytest.approx([-1.8, 2.2])
    assert result["atom_mapped_smiles"].notna().all()
    assert list(result["experimental_pka_implied_dominant_site_form"]) == [
        "acid_form", "base_form",
    ]
    assert list(result["stage3_form_vs_experimental_pka_check"]) == [
        "agrees", "disagrees",
    ]
    assert not result["input_drawing_is_experimental_state_observation"].any()
    table = _deployment_comparison_table_html(result)
    assert "deployed macro pKa" in table
    assert "Stage3 one-body pKa" in table
    assert "dominant-background edge pKa" in table
    assert "8.20" in table
    assert "source drawing" in table


def test_experimental_pka_form_check_uses_protonated_acid_member():
    probability, form = _experimental_pka_implied_form(10.0, 7.4)
    assert probability == pytest.approx(0.997494, abs=1e-6)
    assert form == "acid_form"

    probability, form = _experimental_pka_implied_form(4.0, 7.4)
    assert probability == pytest.approx(0.000398, abs=1e-6)
    assert form == "base_form"


def test_atom_map_highlights_prefer_center_color():
    mol, highlighted, colors = _mapped_molecule_and_highlights(
        "[CH3:1][NH3+:2]",
        "[1, 2]",
        "[2]",
    )

    assert mol is not None
    assert highlighted == [0, 1]
    assert colors[0] == (0.95, 0.75, 0.25)
    assert colors[1] == (0.90, 0.20, 0.20)


def test_large_multisite_error_gets_non_destructive_secondary_transition_guess():
    stage1 = pd.DataFrame(columns=[
        "molecule_id", "site_id", "candidate_label", "site_family",
        "experimental_anchor_pka", "pred_intrinsic_pka",
    ])
    network_sites = [
        {
            "site_id": "site_primary",
            "label": "pyridine",
            "family": "amine",
            "coupling_group": "amine",
            "rank_associated_macro_pka": 4.0,
            "atom_maps": [1, 2, 3, 4, 5, 6],
            "center_maps": [4],
        },
        {
            "site_id": "site_secondary",
            "label": "imidazole_acidity",
            "family": "imidazole_acidity",
            "coupling_group": "imidazole",
            "rank_associated_macro_pka": 9.8,
            "atom_maps": [7, 8, 9, 10, 11],
            "center_maps": [11],
        },
    ]
    stage2 = pd.DataFrame([{
        "molecule_id": "multi",
        "site_id": "site_primary",
        "site_label": "pyridine",
        "site_family": "amine",
        "experimental_anchor_pka": 10.0,
        "stage1_network_macro_pka": 4.0,
        "stage2_supported_projected_macro_pka": 5.0,
        "network_sites_json": json.dumps(network_sites),
    }])
    context = _stage3_context().query("molecule_id == 'multi'").copy()
    context["site_id"] = "site_primary"

    result = build_pka_failure_table(stage1, stage2, context)

    assert result.loc[0, "site_label"] == "pyridine"
    assert result.loc[0, "secondary_guess_site_label"] == "imidazole_acidity"
    assert result.loc[0, "secondary_guess_stage1_macro_pka"] == pytest.approx(9.8)
    assert result.loc[0, "secondary_guess_abs_error"] == pytest.approx(0.2)
    assert bool(result.loc[0, "secondary_guess_review_recommended"])
    assert bool(result.loc[0, "secondary_guess_is_diagnostic_not_training_label"])


def test_secondary_guess_detects_reciprocal_swap_without_relabeling_primary():
    stage1 = pd.DataFrame(columns=[
        "molecule_id", "site_id", "candidate_label", "site_family",
        "experimental_anchor_pka", "pred_intrinsic_pka",
    ])
    network_sites = [
        {
            "site_id": "amine",
            "label": "secondary_amine",
            "family": "amine",
            "rank_associated_macro_pka": 9.0,
            "experimental_anchor_pka": 5.0,
            "atom_maps": [1],
            "center_maps": [1],
        },
        {
            "site_id": "phenol",
            "label": "phenol",
            "family": "phenol_phenolate",
            "rank_associated_macro_pka": 6.0,
            "experimental_anchor_pka": 10.0,
            "atom_maps": [2],
            "center_maps": [2],
        },
    ]
    stage2 = pd.DataFrame([{
        "molecule_id": "multi",
        "site_id": "amine",
        "site_label": "secondary_amine",
        "site_family": "amine",
        "experimental_anchor_pka": 5.0,
        "stage1_network_macro_pka": 9.0,
        "stage2_supported_projected_macro_pka": 9.0,
        "network_sites_json": json.dumps(network_sites),
    }])
    context = _stage3_context().query("molecule_id == 'multi'").copy()
    context["site_id"] = "amine"

    result = build_pka_failure_table(stage1, stage2, context)

    assert result.loc[0, "site_label"] == "secondary_amine"
    assert result.loc[0, "secondary_guess_site_label"] == "phenol"
    assert bool(result.loc[0, "secondary_guess_requires_joint_swap"])
    assert result.loc[0, "secondary_guess_assignment_mode"] == "joint_swap_with_existing_anchor"
    assert result.loc[0, "secondary_guess_joint_swap_current_total_abs_error"] == pytest.approx(8.0)
    assert result.loc[0, "secondary_guess_joint_swap_proposed_total_abs_error"] == pytest.approx(2.0)
    assert result.loc[0, "secondary_guess_joint_swap_improvement"] == pytest.approx(6.0)
    assert bool(result.loc[0, "secondary_guess_review_recommended"])


def test_empty_gallery_removes_stale_png_and_writes_empty_csv(tmp_path):
    png = tmp_path / "stale.png"
    csv_path = tmp_path / "empty.csv"
    png.write_bytes(b"old image")

    rendered = render_gallery(
        pd.DataFrame(columns=["atom_mapped_smiles"]),
        "secondary_transition_assignment_guesses",
        png,
        csv_path,
        max_examples=10,
    )

    assert rendered == 0
    assert not png.exists()
    assert csv_path.exists()
