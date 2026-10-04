import json

import pandas as pd
import pytest

from stage2_network_context import (
    build_stage2_feature_matrix,
    build_stage2_site_table,
    build_stage2_weak_training_sites,
    prepare_stage2_training_sites,
    project_macro_pkas_nonincreasing,
    project_macro_pkas_nonincreasing_with_fixed,
)
from stage2_train_network_context import supported_families_for_counts


def _network_row():
    sites = [
        {
            "site_id": "site_1",
            "label": "carboxylic_acid",
            "family": "carboxyl",
            "atom_maps": [2, 3, 4],
            "center_maps": [4],
            "stage1_intrinsic_pka": 4.0,
            "local_pka_used": 4.0,
            "rank_associated_macro_step": 1,
            "rank_associated_macro_pka": 4.3,
            "stage1_group_feature_present": True,
            "experimental_anchor_pka": 4.8,
            "experimental_measurements": [{"experimental_pka": 4.8}],
        },
        {
            "site_id": "site_2",
            "label": "primary_amine",
            "family": "amine",
            "atom_maps": [1, 2],
            "center_maps": [1],
            "stage1_intrinsic_pka": 9.5,
            "local_pka_used": 9.5,
            "rank_associated_macro_step": 2,
            "rank_associated_macro_pka": 9.2,
            "stage1_group_feature_present": True,
            "experimental_anchor_pka": None,
            "experimental_measurements": [],
        },
    ]
    return {
        "molecule_id": "mol_test",
        "representative_input_smiles": "NCC(=O)O",
        "representative_atom_mapped_smiles": "[NH2:1][CH2:2][C:3](=[O:4])[OH:5]",
        "detected_site_count": 2,
        "protonation_state_count": 4,
        "protonation_edge_count": 4,
        "protonation_state_enumeration_truncated": False,
        "incomplete_site_ids_json": "[]",
        "sites_json": json.dumps(sites),
    }


def test_stage2_target_is_residual_from_blind_network_baseline():
    table = build_stage2_site_table(pd.DataFrame([_network_row()]), anchored_only=True)
    assert len(table) == 1
    assert table.iloc[0]["experimental_anchor_pka"] == pytest.approx(4.8)
    assert table.iloc[0]["stage1_network_macro_pka"] == pytest.approx(4.3)
    assert table.iloc[0]["stage2_delta_target"] == pytest.approx(0.5)


def test_stage2_features_exclude_experimental_target_columns():
    table = build_stage2_site_table(pd.DataFrame([_network_row()]), anchored_only=True)
    features = build_stage2_feature_matrix(table, fp_bits=32)
    forbidden = {
        "experimental_anchor_pka",
        "experimental_measurement_count",
        "experimental_measurement_std",
        "stage2_delta_target",
    }
    assert not (forbidden & set(features.columns))
    assert not any(column.startswith("experimental_") for column in features.columns)
    assert "stage1_intrinsic_pka" in features.columns
    assert "stage1_network_macro_pka" in features.columns
    assert "other_center_graph_distance_min" in features.columns


def test_stage2_site_table_contains_unmeasured_sites_for_application():
    table = build_stage2_site_table(pd.DataFrame([_network_row()]), anchored_only=False)
    assert set(table["site_id"]) == {"site_1", "site_2"}
    assert table.loc[table.site_id == "site_2", "experimental_anchor_pka"].isna().all()


def test_macro_projection_enforces_thermodynamic_order():
    projected = project_macro_pkas_nonincreasing([8.0, 9.0, 6.0])
    assert projected == pytest.approx([8.5, 8.5, 6.0])
    assert all(projected[idx] >= projected[idx + 1] for idx in range(len(projected) - 1))


def test_macro_projection_does_not_move_unsupported_fallback_sites():
    projected = project_macro_pkas_nonincreasing_with_fixed(
        [8.0, 9.0, 6.0, 7.0],
        [False, True, False, True],
    )
    assert projected == pytest.approx([8.0, 8.0, 6.0, 6.0])
    assert projected[0] == 8.0
    assert projected[2] == 6.0


def test_stage2_quarantines_conflicting_replicates():
    table = build_stage2_site_table(pd.DataFrame([_network_row()]), anchored_only=True)
    conflicting = table.iloc[0].copy()
    conflicting["molecule_key"] = "mol_conflict"
    conflicting["molecule_id"] = "mol_conflict"
    conflicting["experimental_anchor_pka"] = 8.0
    conflicting["experimental_measurement_count"] = 2.0
    conflicting["experimental_measurement_min"] = 2.0
    conflicting["experimental_measurement_max"] = 14.0
    conflicting["experimental_measurement_range"] = 12.0
    conflicting["experimental_measurement_mad"] = 6.0
    conflicting["experimental_measurement_robust_sigma"] = 8.8956
    conflicting["experimental_values_json"] = "[2.0,14.0]"
    accepted, quarantine, funnel = prepare_stage2_training_sites(
        pd.DataFrame([table.iloc[0], conflicting])
    )
    assert len(accepted) == 1
    assert len(quarantine) == 1
    assert quarantine.iloc[0]["quarantine_reason"] == "replicate_range_exceeds_threshold"
    assert funnel["accepted_training_sites"] == 1


def test_priority_families_bypass_only_the_general_count_gate():
    supported = supported_families_for_counts({
        "amine": 100,
        "guanidine_like": 49,
        "phenol_phenolate": 46,
        "imine_iminium": 49,
        "alcohol_alkoxide": 0,
    })
    assert supported == {"amine", "guanidine_like", "phenol_phenolate"}


def test_complete_weak_molecule_label_is_marginalized_without_full_exact_weight():
    row = _network_row()
    row["weak_experimental_measurements_json"] = json.dumps([{
        "experimental_pka": 8.0,
        "transition_row_id": "weak_1",
        "source_document_id": "doc_1",
        "source_file": "toy.sdf",
        "record_index": 1,
    }])
    row["unresolved_ionizable_context_count"] = 0
    full = build_stage2_site_table(pd.DataFrame([row]), anchored_only=False)
    weak, blocked = build_stage2_weak_training_sites(pd.DataFrame([row]), full)
    assert blocked.empty
    assert len(weak) == 2
    assert weak["weak_candidate_probability"].sum() == pytest.approx(1.0)
    assert weak["sample_weight"].sum() == pytest.approx(0.20)
    assert set(weak["supervision_scope"]) == {"weak_molecule_macro_marginalized"}


def test_weak_label_is_blocked_when_there_is_an_unenumerated_candidate_context():
    row = _network_row()
    row["weak_experimental_measurements_json"] = json.dumps([{
        "experimental_pka": 8.0, "transition_row_id": "weak_1"
    }])
    row["unresolved_ionizable_context_count"] = 1
    row["unresolved_ionizable_contexts_json"] = '[{"label":"thioamide"}]'
    full = build_stage2_site_table(pd.DataFrame([row]), anchored_only=False)
    weak, blocked = build_stage2_weak_training_sites(pd.DataFrame([row]), full)
    assert weak.empty
    assert len(blocked) == 1
    assert blocked.iloc[0]["blocked_reason"] == "incomplete_candidate_space_unresolved_ionizable_context"


def test_incomplete_network_is_excluded_from_exact_and_weak_stage2_training():
    row = _network_row()
    row["incomplete_site_ids_json"] = json.dumps(["site_2"])
    row["weak_experimental_measurements_json"] = json.dumps([{
        "experimental_pka": 8.0,
        "transition_row_id": "weak_1",
    }])
    row["unresolved_ionizable_context_count"] = 0

    full = build_stage2_site_table(pd.DataFrame([row]), anchored_only=False)
    weak, blocked = build_stage2_weak_training_sites(pd.DataFrame([row]), full)

    assert full.empty
    assert weak.empty
    assert len(blocked) == 1
    assert blocked.iloc[0]["blocked_reason"] == "incomplete_or_missing_network"
