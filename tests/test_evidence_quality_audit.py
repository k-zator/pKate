import json

import pandas as pd

from audit_pka_evidence_quality import run_audit


def test_evidence_audit_reports_tiers_lineages_and_stage3_site_confidence(tmp_path):
    transitions = pd.DataFrame([
        {
            "dataset_row_id": "r1", "source_file": "a.sdf", "record_index": 0,
            "input_smiles": "CCO", "experimental_pka": 16.0,
            "site_label": "primary_alcohol", "site_family": "alcohol_alkoxide",
            "evidence_tier": "silver", "reference_plausibility_status": "consistent",
            "stage1_local_supervision_eligible": True,
            "source_structure_charge_changed": False,
            "unresolved_ionizable_context_count": 0,
        },
        {
            "dataset_row_id": "r2", "source_file": "copy.sdf", "record_index": 1,
            "input_smiles": "OCC", "experimental_pka": 16.0,
            "site_label": "primary_alcohol", "site_family": "alcohol_alkoxide",
            "evidence_tier": "silver", "reference_plausibility_status": "consistent",
            "stage1_local_supervision_eligible": True,
            "source_structure_charge_changed": False,
            "unresolved_ionizable_context_count": 0,
        },
        {
            "dataset_row_id": "r3", "source_file": "b.sdf", "record_index": 2,
            "input_smiles": "c1cc[nH]c1", "experimental_pka": 7.5,
            "site_label": "pyrrole", "site_family": "amine",
            "evidence_tier": "ambiguous", "reference_plausibility_status": "conflict",
            "stage1_local_supervision_eligible": False,
            "source_structure_charge_changed": False,
            "unresolved_ionizable_context_count": 1,
            "reference_z_distance": 4.0,
        },
    ])
    transition_path = tmp_path / "transitions.csv"
    transitions.to_csv(transition_path, index=False)
    network_path = tmp_path / "stage3.csv"
    pd.DataFrame([{
        "sites_json": json.dumps([
            {
                "stage3_overall_site_state_confidence_tier": "high",
                "stage3_empirical_pka_side_call_confidence": 0.9,
                "stage3_empirical_calibration_source": "stage1_single_site",
                "stage3_empirical_calibration_scope": "label_family_global_hierarchical_shrinkage",
            },
            {
                "stage3_overall_site_state_confidence_tier": "very_low",
                "stage3_empirical_pka_side_call_confidence": 0.4,
                "stage3_empirical_calibration_source": "stage2_multisite",
                "stage3_empirical_calibration_scope": "family_global_hierarchical_shrinkage",
            },
        ])
    }]).to_csv(network_path, index=False)
    stage1_path = tmp_path / "stage1.csv"
    pd.DataFrame([{
        "candidate_label": "primary_alcohol",
        "experimental_evidence_tier": "silver",
        "experimental_measurement_count": 2,
        "experimental_independent_lineage_count": 1,
    }]).to_csv(stage1_path, index=False)

    summary = run_audit(
        str(transition_path), str(network_path), str(stage1_path), str(tmp_path / "audit"),
        str(tmp_path / "missing_weak.csv"), str(tmp_path / "missing_blocked.csv"),
    )
    assert summary["transition_dataset"]["evidence_tiers"] == {
        "silver": 2, "ambiguous": 1
    }
    assert summary["source_lineage"]["duplicate_groups"] == 1
    assert summary["stage1_training"]["independent_measurement_lineages"] == 1
    assert summary["stage3_network"]["overall_site_state_confidence_tiers"] == {
        "high": 1, "very_low": 1
    }
    calibration = summary["stage3_network"]["empirical_pka_side_calibration"]
    assert calibration["available_site_calls"] == 2
    assert calibration["median"] == 0.65
    assert calibration["sources"] == {
        "stage1_single_site": 1, "stage2_multisite": 1
    }
