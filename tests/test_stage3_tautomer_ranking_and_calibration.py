import pytest

from stage3_empirical_calibration import calibrate_site_call
from stage3_tautomer_ranking import (
    rank_tautomers_within_configurations,
    select_top_tautomer_in_configuration,
)


def test_tautomer_weights_are_normalized_over_equivalent_configuration_nodes():
    nodes = [
        {
            "node_id": "neutral_a",
            "group_levels": {"g": 0},
            "stage3_configuration_population_at_ph": 0.8,
            "tautomers": ["Oc1ccccc1"],
            "tautomer_enumeration_truncated": False,
        },
        {
            "node_id": "neutral_b",
            "group_levels": {"g": 0},
            "stage3_configuration_population_at_ph": 0.8,
            "tautomers": ["O=C1C=CC=CC1"],
            "tautomer_enumeration_truncated": False,
        },
        {
            "node_id": "protonated",
            "group_levels": {"g": 1},
            "stage3_configuration_population_at_ph": 0.2,
            "tautomers": ["[OH2+]c1ccccc1"],
            "tautomer_enumeration_truncated": False,
        },
    ]

    records, summary, dominant = rank_tautomers_within_configurations(
        nodes, molecule_id="phenol"
    )

    neutral = [record for record in records if record["configuration_tautomer_count"] == 2]
    assert sum(row["tautomer_heuristic_conditional_weight"] for row in neutral) == pytest.approx(1.0)
    assert sum(row["complete_microstate_heuristic_population_at_ph"] for row in records) == pytest.approx(1.0)
    assert summary["complete_microstate_heuristic_population_sum"] == pytest.approx(1.0)
    assert dominant is not None
    assert dominant["atom_mapped_smiles"] == "Oc1ccccc1"
    assert all("stage3_top_ranked_tautomer" in node for node in nodes)


def test_final_selection_cannot_leave_preselected_protonation_configuration():
    records = [
        {"configuration_id": "physical_winner", "tautomer_rank_within_configuration": 1,
         "atom_mapped_smiles": "a", "complete_microstate_heuristic_population_at_ph": 0.3},
        {"configuration_id": "physical_winner", "tautomer_rank_within_configuration": 2,
         "atom_mapped_smiles": "b", "complete_microstate_heuristic_population_at_ph": 0.3},
        {"configuration_id": "heuristic_global_winner", "tautomer_rank_within_configuration": 1,
         "atom_mapped_smiles": "c", "complete_microstate_heuristic_population_at_ph": 0.4},
    ]
    selected = select_top_tautomer_in_configuration(records, "physical_winner")
    assert selected["atom_mapped_smiles"] == "a"


def test_configuration_reenumeration_replaces_stale_source_truncation_flag():
    nodes = [{
        "node_id": "n",
        "site_forms": {"s": "base_form"},
        "stage3_configuration_population_at_ph": 1.0,
        "tautomers": ["c1ccncc1", "c1cccnc1"],
        "tautomer_enumeration_truncated": True,
    }]
    records, summary, _ = rank_tautomers_within_configurations(
        nodes, molecule_id="pyridine"
    )
    assert summary["truncated_configuration_count"] == 0
    assert summary["source_truncated_configuration_count"] == 1
    assert records[0]["source_node_enumeration_truncated"] is True
    assert nodes[0]["stage3_tautomer_ranking_limited_by_truncation"] is False


def test_configuration_wide_limit_is_applied_after_equivalent_seed_deduplication():
    nodes = [
        {
            "node_id": "keto",
            "group_levels": {"g": 0},
            "stage3_configuration_population_at_ph": 1.0,
            "reference_atom_mapped_smiles": "CC=O",
            "tautomer_enumeration_truncated": False,
        },
        {
            "node_id": "enol",
            "group_levels": {"g": 0},
            "stage3_configuration_population_at_ph": 1.0,
            "reference_atom_mapped_smiles": "C=CO",
            "tautomer_enumeration_truncated": False,
        },
    ]
    records, summary, _ = rank_tautomers_within_configurations(
        nodes,
        molecule_id="acetaldehyde",
        max_tautomers_per_configuration=16,
    )
    assert {row["atom_mapped_smiles"] for row in records} == {"CC=O", "C=CO"}
    assert len(records) == 2
    assert summary["complete_configuration_count"] == 1
    assert summary["truncated_configuration_count"] == 0
    assert all(row["configuration_enumeration_seed_node_count"] == 2 for row in records)


def test_configuration_wide_limit_reports_rdkit_truncation():
    nodes = [{
        "node_id": "keto",
        "group_levels": {"g": 0},
        "stage3_configuration_population_at_ph": 1.0,
        "reference_atom_mapped_smiles": "CC=O",
        "tautomer_enumeration_truncated": False,
    }]
    records, summary, _ = rank_tautomers_within_configurations(
        nodes,
        molecule_id="acetaldehyde",
        max_tautomers_per_configuration=1,
    )
    assert len(records) == 1
    assert summary["truncated_configuration_count"] == 1
    assert records[0]["configuration_rdkit_truncated_seed_count"] == 1
    assert records[0]["configuration_enumeration_complete"] is False


def _calibration_fixture():
    source = {
        "rows": 4,
        "global_residuals": [-0.3, -0.1, 0.1, 0.3],
        "family_residuals": {"amine": [-0.3, -0.1, 0.1, 0.3]},
        "label_residuals": {"primary_amine": [-0.1, 0.1]},
    }
    return {"sources": {"stage1_single_site": source}}


def test_empirical_calibration_reports_probability_for_selected_pka_side():
    acid = calibrate_site_call(
        _calibration_fixture(),
        source_name="stage1_single_site",
        site_label="primary_amine",
        site_family="amine",
        predicted_pka=10.0,
        ph=7.4,
        predicted_site_form="acid_form",
    )
    base = calibrate_site_call(
        _calibration_fixture(),
        source_name="stage1_single_site",
        site_label="primary_amine",
        site_family="amine",
        predicted_pka=4.0,
        ph=7.4,
        predicted_site_form="base_form",
    )
    assert acid["predicted_site_form_pka_side_confidence"] > 0.75
    assert base["predicted_site_form_pka_side_confidence"] > 0.75
    assert acid["label_rows"] == 2
    assert acid["family_rows"] == 4
    assert acid["empirical_macro_pka_interval_90_low"] < 10.0
    assert acid["empirical_macro_pka_interval_90_high"] > 10.0
