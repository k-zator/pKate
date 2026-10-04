import json

import pandas as pd
import pytest
from rdkit import Chem

from stage1_outlier_review import (
    DECISION_COLUMNS,
    _decision_for_action,
    _web_review_page,
    build_detector_expansion_review_queue,
    reopen_reviews_for_detector_change,
)


def _queue_row():
    candidate = {
        "candidate_index": 0,
        "label": "primary_amine",
        "family": "amine",
        "site_type": "basic",
        "atom_indices": [0, 1],
        "pair_supported": True,
        "is_current": True,
    }
    return pd.Series({
        "source_file": "toy.sdf",
        "record_index": 0,
        "smiles": "CN",
        "pka_value": 10.0,
        "pka_source_method": "experimental",
        "pka_type_canonical": "basic",
        "atom_index_raw": None,
        "review_id": "outlier_test",
        "review_model_sha256": "abc",
        "abs_error": 4.0,
        "predicted_pka": 6.0,
        "source_record_pka": 10.0,
        "outlier_rank": 1,
        "current_site_label": "primary_amine",
        "current_site_family": "amine",
        "current_site_atom_maps_json": "[1,2]",
        "current_assignment_basis": "structurally_unique_pair_site",
        "source_site_evidence": "unannotated_molecule_level_pka",
        "source_site_evidence_confidence": "none",
        "source_assay_id": "",
        "source_document_id": "",
        "source_molecule_id": "",
        "stage1_structure_key": "CN",
        "candidate_sites_json": json.dumps([candidate]),
    })


def test_first_web_decision_can_be_saved_to_empty_table():
    decisions = pd.DataFrame(columns=DECISION_COLUMNS)
    updated = _decision_for_action(
        decisions,
        _queue_row(),
        "confirm_assignment",
        {"note": ["chemically checked"]},
    )
    assert len(updated) == 1
    decision = updated.iloc[0]
    assert decision["review_decision"] == "confirm_assignment"
    assert decision["reviewed_site_label"] == "primary_amine"
    assert json.loads(decision["reviewed_site_atom_indices_json"]) == [0, 1]
    assert decision["review_note"] == "chemically checked"


def test_confirm_with_a_different_selected_candidate_reassigns_that_candidate():
    row = _queue_row()
    candidates = json.loads(row["candidate_sites_json"])
    candidates.append({
        "candidate_index": 1,
        "label": "carboxylic_acid",
        "family": "carboxyl",
        "site_type": "acidic",
        "atom_indices": [2, 3, 4],
        "pair_supported": True,
        "is_current": False,
    })
    row["candidate_sites_json"] = json.dumps(candidates)
    updated = _decision_for_action(
        pd.DataFrame(columns=DECISION_COLUMNS), row, "confirm_assignment",
        {"candidate_index": ["1"], "note": ["selected alternative"]},
    )
    decision = updated.iloc[0]
    assert decision["review_decision"] == "reassign_site"
    assert decision["reviewed_site_label"] == "carboxylic_acid"
    assert json.loads(decision["reviewed_site_atom_indices_json"]) == [2, 3, 4]


def test_ambiguous_evidence_confirmation_requires_explicit_attestations():
    row = _queue_row()
    row["review_trigger"] = "evidence_quality_ambiguous"
    with pytest.raises(ValueError, match="source-site and transition verification"):
        _decision_for_action(
            pd.DataFrame(columns=DECISION_COLUMNS), row, "confirm_assignment", {}
        )
    updated = _decision_for_action(
        pd.DataFrame(columns=DECISION_COLUMNS), row, "confirm_assignment",
        {
            "verify_site_attribution": ["on"],
            "verify_transition_identity": ["on"],
        },
    )
    assert bool(updated.iloc[0]["review_verified_site_attribution"])
    assert bool(updated.iloc[0]["review_verified_transition_identity"])


def test_web_completion_message_distinguishes_deferred_records():
    row = _queue_row()
    queue = pd.DataFrame([row])
    deferred = _web_review_page(queue, None, {"outlier_test": "defer"})
    completed = _web_review_page(queue, None, {"outlier_test": "confirm_assignment"})
    assert "This review pass is complete" in deferred
    assert "do not rebuild the final dataset yet" in deferred
    assert "Stage 1 review complete" in completed
    assert "corrected release can now be rebuilt" in completed


def test_web_progress_ignores_decisions_not_present_in_current_queue():
    queue = pd.DataFrame([_queue_row()])
    page = _web_review_page(
        queue, queue.iloc[0],
        {"outlier_test": "defer", "old_no_longer_queued": "confirm_assignment"},
    )
    assert "0/1 complete" in page
    assert "1 remaining" in page
    assert "1 deferred" in page


def test_detector_change_reopens_new_phosphate_site_and_preserves_history(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    writer = Chem.SDWriter(str(raw_dir / "toy.sdf"))
    writer.write(Chem.MolFromSmiles("COP(=O)(O)O"))
    writer.close()
    decision = {column: "" for column in DECISION_COLUMNS}
    decision.update({
        "source_file": "toy.sdf",
        "record_index": 0,
        "smiles": "COP(=O)(O)O",
        "pka_value": 2.0,
        "pka_source_method": "experimental",
        "review_id": "outlier_phosphate",
        "review_decision": "exclude_wrong_site_unknown",
        "review_note": "no phosphorus candidate was shown",
        "reviewed_at": "2026-09-01T00:00:00Z",
    })
    decisions_path = tmp_path / "decisions.csv"
    history_path = tmp_path / "history.csv"
    report_path = tmp_path / "report.csv"
    pd.DataFrame([decision]).to_csv(decisions_path, index=False)

    report = reopen_reviews_for_detector_change(
        str(decisions_path), str(raw_dir), str(history_path), str(report_path)
    )
    assert len(report) == 1
    assert "phosphoric_acid" in report.iloc[0]["new_candidate_sites_json"]
    current = pd.read_csv(decisions_path)
    assert current.iloc[0]["review_decision"] == "defer"
    assert "prior_decision=exclude_wrong_site_unknown" in current.iloc[0]["review_note"]
    history = pd.read_csv(history_path)
    assert history.iloc[0]["review_decision"] == "exclude_wrong_site_unknown"
    assert history.iloc[0]["replacement_detector_schema_version"] == "2.5.0"

    # Idempotent: an already reopened decision is not appended twice.
    assert reopen_reviews_for_detector_change(
        str(decisions_path), str(raw_dir), str(history_path), str(report_path)
    ).empty
    assert len(pd.read_csv(history_path)) == 1


def test_detector_expansion_queue_includes_unreviewed_unique_alcohol(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    writer = Chem.SDWriter(str(raw_dir / "toy.sdf"))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    assignment = {
        "source_file": "toy.sdf", "record_index": 0, "smiles": "CCO",
        "pka_value": 16.0, "pka_source_method": "experimental",
        "pka_type_canonical": "acidic", "atom_index_raw": None,
    }
    assignments_path = tmp_path / "assignments.csv"
    pd.DataFrame([assignment]).to_csv(assignments_path, index=False)
    release_path = tmp_path / "candidate_release.csv"
    pd.DataFrame([{**assignment,
        "experimental_pka": 16.0, "input_smiles": "CCO",
        "dataset_row_id": "new_alcohol", "site_label": "primary_alcohol",
        "site_family": "alcohol_alkoxide", "site_atom_maps_json": "[2,3]",
        "protonation_center_atom_maps_json": "[3]",
        "site_assignment_basis": "structurally_unique_pair_site",
        "site_assignment_confidence": "high", "override_applied": False,
        "source_site_evidence": "unannotated_molecule_level_pka",
        "source_site_evidence_confidence": "none",
    }]).to_csv(release_path, index=False)

    queue = build_detector_expansion_review_queue(
        str(release_path), assignments_path=str(assignments_path),
        decisions_path=str(tmp_path / "missing_decisions.csv"), raw_dir=str(raw_dir),
    )
    assert len(queue) == 1
    assert queue.iloc[0]["review_trigger"] == "detector_v2_new_family_stage1_candidate"
    assert queue.iloc[0]["current_site_family"] == "alcohol_alkoxide"
    assert "primary_alcohol" in queue.iloc[0]["candidate_labels"]
