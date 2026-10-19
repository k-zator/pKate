"""Detailed workflow tests for the persistent manual pKa quality-control tools."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import pytest
from rdkit import Chem

import pka_structure_consistency_review as structure_review
import stage1_outlier_review as outlier_review
import visualize_canonical_stage3_failures as failure_viz


def _write_sdf(path: Path, smiles: str) -> None:
    writer = Chem.SDWriter(str(path))
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    writer.write(molecule)
    writer.close()


def _capture_handler(monkeypatch, module, target, *args):
    """Capture a reviewer's real HTTP handler without opening a network socket."""
    captured = {}

    class FakeServer:
        def __init__(self, address, handler):
            captured["address"] = address
            captured["handler"] = handler

        def serve_forever(self):
            return None

        def server_close(self):
            return None

    monkeypatch.setattr(module, "ThreadingHTTPServer", FakeServer)
    target(*args)
    return captured["handler"]


def _handler_request(handler_class, method: str, path: str, fields=None):
    """Invoke a BaseHTTPRequestHandler method against in-memory byte streams."""
    raw = b"" if fields is None else urlencode(fields).encode("utf-8")
    handler = object.__new__(handler_class)
    handler.path = path
    handler.headers = {"Content-Length": str(len(raw))}
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    status = []
    headers = {}
    handler.send_response = status.append
    handler.send_header = lambda key, value: headers.__setitem__(key, value)
    handler.end_headers = lambda: None
    getattr(handler, f"do_{method}")()
    return status[-1], headers, handler.wfile.getvalue()


def _assignment(smiles: str = "CN", pka: float = 10.0) -> dict[str, object]:
    return {
        "source_file": "toy.sdf",
        "record_index": 0,
        "smiles": smiles,
        "pka_value": pka,
        "pka_source_method": "experimental",
        "pka_type_canonical": "basic" if "N" in smiles else "acidic",
        "pka_type_raw": "basic" if "N" in smiles else "acidic",
        "atom_index_raw": None,
    }


def _transition(smiles: str = "CN", pka: float = 10.0) -> dict[str, object]:
    return {
        "dataset_row_id": "transition-1",
        "source_file": "toy.sdf",
        "record_index": 0,
        "experimental_pka": pka,
        "input_smiles": smiles,
        "site_label": "primary_amine",
        "site_family": "amine",
        "site_atom_maps_json": "[1,2]",
        "protonation_center_atom_maps_json": "[2]",
        "site_assignment_basis": "structurally_unique_pair_site",
        "site_assignment_confidence": "high",
        "structural_site_assignment_confidence": "high",
        "source_site_evidence": "source_atom_index",
        "source_site_evidence_confidence": "high",
        "source_assay_id": "assay-1",
        "source_document_id": "document-1",
        "source_molecule_id": "molecule-1",
    }


def _outlier_inputs(tmp_path: Path) -> dict[str, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_sdf(raw / "toy.sdf", "CN")
    assignment = tmp_path / "assignments.csv"
    pd.DataFrame([_assignment()]).to_csv(assignment, index=False)
    transitions = tmp_path / "transitions.csv"
    pd.DataFrame([_transition()]).to_csv(transitions, index=False)
    oof = tmp_path / "oof.csv"
    pd.DataFrame([
        {
            "molecule_id": "mol-1",
            "site_id": "site-1",
            "structure_key": "CN",
            "pka_value": 10.0,
            "pred_intrinsic_pka": 5.0,
            "abs_error": 5.0,
            "experimental_transition_ids_json": '["transition-1"]',
        },
        {
            "molecule_id": "below-threshold",
            "site_id": "site-2",
            "structure_key": "CN",
            "pka_value": 10.0,
            "pred_intrinsic_pka": 8.0,
            "abs_error": 2.0,
            "experimental_transition_ids_json": '["transition-1"]',
        },
        {
            "molecule_id": "missing-transition",
            "site_id": "site-3",
            "structure_key": "CN",
            "pka_value": 10.0,
            "pred_intrinsic_pka": 0.0,
            "abs_error": 10.0,
            "experimental_transition_ids_json": '["not-present"]',
        },
    ]).to_csv(oof, index=False)
    return {"raw": raw, "assignment": assignment, "transitions": transitions, "oof": oof}


def _queue_row() -> pd.Series:
    candidate = {
        "candidate_index": 0,
        "label": "primary_amine",
        "family": "amine",
        "site_type": "basic",
        "atom_indices": [1],
        "atom_text": "1",
        "pair_supported": True,
        "is_current": True,
        "is_auto": True,
        "atom_index_hit": True,
    }
    alternative = {
        "candidate_index": 1,
        "label": "unsupported_nitrogen",
        "family": "unknown",
        "site_type": "basic",
        "atom_indices": [1],
        "atom_text": "1",
        "pair_supported": False,
        "is_current": False,
        "is_auto": False,
        "atom_index_hit": False,
    }
    return pd.Series({
        **_assignment(),
        "review_id": "outlier-test",
        "review_trigger": "stage1_scaffold_oof_abs_error_gt_threshold",
        "review_status": "pending",
        "review_note": "",
        "reviewed_at": "",
        "review_model_sha256": "sha",
        "outlier_rank": 1,
        "stage1_structure_key": "CN",
        "source_record_pka": 10.0,
        "predicted_pka": 5.0,
        "abs_error": 5.0,
        "current_site_label": "primary_amine",
        "current_site_family": "amine",
        "current_site_atom_maps_json": "[2]",
        "current_protonation_center_maps_json": "[2]",
        "current_assignment_basis": "structurally_unique_pair_site",
        "structural_site_assignment_confidence": "high",
        "source_site_evidence": "source_atom_index",
        "source_site_evidence_confidence": "high",
        "source_assay_id": "assay-1",
        "source_document_id": "document-1",
        "source_molecule_id": "molecule-1",
        "candidate_sites_json": json.dumps([candidate, alternative]),
    })


def test_stage1_outlier_queue_traces_model_failure_back_to_exact_raw_record(tmp_path: Path):
    inputs = _outlier_inputs(tmp_path)
    queue = outlier_review.build_outlier_review_queue(
        str(inputs["oof"]), str(inputs["transitions"]), str(inputs["assignment"]),
        str(tmp_path / "missing-decisions.csv"), str(inputs["raw"]), threshold=3.0,
    )
    assert len(queue) == 1
    row = queue.iloc[0]
    assert row["dataset_row_id"] == "transition-1"
    assert row["review_status"] == "pending"
    assert row["abs_error"] == pytest.approx(5.0)
    assert row["review_model_sha256"] == outlier_review._sha256(str(inputs["oof"]))
    assert "primary_amine" in row["candidate_labels"]
    candidates = json.loads(row["candidate_sites_json"])
    assert any(candidate["is_current"] and candidate["pair_supported"] for candidate in candidates)


def test_stage1_queue_uses_persisted_record_specific_decision(tmp_path: Path):
    inputs = _outlier_inputs(tmp_path)
    first = outlier_review.build_outlier_review_queue(
        str(inputs["oof"]), str(inputs["transitions"]), str(inputs["assignment"]),
        str(tmp_path / "decisions.csv"), str(inputs["raw"]),
    )
    decision = outlier_review._upsert_decision(
        pd.DataFrame(columns=outlier_review.DECISION_COLUMNS), first.iloc[0],
        "exclude_implausible_pka", note="source checked",
    )
    outlier_review._save_decisions(str(tmp_path / "decisions.csv"), decision)
    rebuilt = outlier_review.build_outlier_review_queue(
        str(inputs["oof"]), str(inputs["transitions"]), str(inputs["assignment"]),
        str(tmp_path / "decisions.csv"), str(inputs["raw"]),
    )
    assert rebuilt.iloc[0]["review_status"] == "exclude_implausible_pka"
    assert rebuilt.iloc[0]["review_note"] == "source checked"


def test_stage1_decision_loader_repairs_partial_schema_and_save_is_last_write_wins(tmp_path: Path):
    path = tmp_path / "decisions.csv"
    pd.DataFrame({"review_id": ["one"], "review_decision": [None]}).to_csv(path, index=False)
    loaded = outlier_review._load_decisions(str(path))
    assert list(loaded.columns) == outlier_review.DECISION_COLUMNS
    assert loaded.iloc[0]["review_decision"] == ""

    row = _queue_row()
    first = outlier_review._upsert_decision(loaded.iloc[:0], row, "defer", note="first")
    second = outlier_review._upsert_decision(first, row, "exclude_wrong_endpoint", note="second")
    outlier_review._save_decisions(str(path), pd.concat([first, second], ignore_index=True))
    persisted = outlier_review._load_decisions(str(path))
    assert len(persisted) == 1
    assert persisted.iloc[0]["review_decision"] == "exclude_wrong_endpoint"
    assert persisted.iloc[0]["review_note"] == "second"


def test_evidence_quality_queue_requires_independent_attestation_for_structural_confirmations(tmp_path: Path):
    inputs = _outlier_inputs(tmp_path)
    release = pd.DataFrame([{
        **_transition(),
        "evidence_tier": "ambiguous",
        "evidence_tier_reason": "molecule-level scalar with two ionizable contexts",
        "unresolved_ionizable_context_count": 2,
        "unresolved_ionizable_contexts_json": '["site-2"]',
        "reference_z_distance": 1.2,
        "reference_pka": 9.0,
        "reference_transition_definition": "BH+ -> B",
        "source_original_smiles": "CN",
        "evidence_policy_schema_version": "test-policy",
    }])
    release_path = tmp_path / "ambiguous.csv"
    release.to_csv(release_path, index=False)
    decisions_path = tmp_path / "decisions.csv"
    queue = outlier_review.build_evidence_quality_review_queue(
        str(release_path), str(inputs["assignment"]), str(decisions_path), str(inputs["raw"])
    )
    assert len(queue) == 1
    assert queue.iloc[0]["review_status"] == "pending"
    assert queue.iloc[0]["review_trigger"] == "evidence_quality_ambiguous"
    assert queue.iloc[0]["abs_error"] == pytest.approx(1.0)

    unverified = outlier_review._upsert_decision(
        pd.DataFrame(columns=outlier_review.DECISION_COLUMNS), queue.iloc[0],
        "confirm_assignment", "primary_amine", [1],
    )
    outlier_review._save_decisions(str(decisions_path), unverified)
    assert outlier_review.build_evidence_quality_review_queue(
        str(release_path), str(inputs["assignment"]), str(decisions_path), str(inputs["raw"])
    ).iloc[0]["review_status"] == "pending"

    verified = outlier_review._decision_for_action(
        unverified, queue.iloc[0], "confirm_assignment",
        {"verify_site_attribution": ["on"], "verify_transition_identity": ["on"]},
    )
    outlier_review._save_decisions(str(decisions_path), verified)
    assert outlier_review.build_evidence_quality_review_queue(
        str(release_path), str(inputs["assignment"]), str(decisions_path), str(inputs["raw"])
    ).iloc[0]["review_status"] == "confirm_assignment"


def test_evidence_queue_absent_schema_is_empty(tmp_path: Path):
    path = tmp_path / "release.csv"
    pd.DataFrame({"dataset_row_id": ["x"]}).to_csv(path, index=False)
    assert outlier_review.build_evidence_quality_review_queue(release_dataset_path=str(path)).empty


def test_detector_reopen_report_becomes_clickable_queue(tmp_path: Path):
    inputs = _outlier_inputs(tmp_path)
    report = tmp_path / "report.csv"
    pd.DataFrame([{
        "source_file": "toy.sdf", "record_index": 0, "pka_value": 10.0,
    }]).to_csv(report, index=False)
    queue = outlier_review.build_detector_reopen_review_queue(
        str(report), str(inputs["assignment"]), str(tmp_path / "decisions.csv"), str(inputs["raw"])
    )
    assert len(queue) == 1
    assert queue.iloc[0]["review_trigger"] == "detector_schema_change_reopened_decision"
    assert queue.iloc[0]["current_site_label"] == "primary_amine"
    assert queue.iloc[0]["source_site_evidence"] == "source_ionization_type_only_no_site"
    assert outlier_review.build_detector_reopen_review_queue(str(tmp_path / "absent.csv")).empty


def test_review_status_refresh_keeps_unattested_evidence_confirmation_pending(tmp_path: Path):
    row = _queue_row()
    row["review_trigger"] = "evidence_quality_ambiguous"
    queue = pd.DataFrame([row])
    decision = outlier_review._upsert_decision(
        pd.DataFrame(columns=outlier_review.DECISION_COLUMNS), row,
        "confirm_assignment", "primary_amine", [1],
    )
    path = tmp_path / "decisions.csv"
    outlier_review._save_decisions(str(path), decision)
    refreshed = outlier_review._refresh_review_status(queue, str(path))
    assert refreshed.iloc[0]["review_status"] == "pending"

    decision.loc[:, "review_verified_site_attribution"] = True
    decision.loc[:, "review_verified_transition_identity"] = True
    outlier_review._save_decisions(str(path), decision)
    refreshed = outlier_review._refresh_review_status(queue, str(path))
    assert refreshed.iloc[0]["review_status"] == "confirm_assignment"


@pytest.mark.parametrize(
    ("action", "form", "decision"),
    [
        ("reassign_site", {"candidate_index": ["1"]}, "unsupported_site"),
        ("unsupported_site", {"unsupported_label": ["new_site"], "unsupported_atom": ["3"]}, "unsupported_site"),
        ("exclude_wrong_site_unknown", {"note": ["unknown"]}, "exclude_wrong_site_unknown"),
        ("exclude_wrong_endpoint", {}, "exclude_wrong_endpoint"),
        ("exclude_implausible_pka", {}, "exclude_implausible_pka"),
        ("defer", {}, "defer"),
    ],
)
def test_stage1_web_actions_persist_each_supported_outcome(action, form, decision):
    updated = outlier_review._decision_for_action(
        pd.DataFrame(columns=outlier_review.DECISION_COLUMNS), _queue_row(), action, form
    )
    assert updated.iloc[0]["review_decision"] == decision


@pytest.mark.parametrize(
    ("action", "form", "message"),
    [
        ("confirm_assignment", {"candidate_index": ["bad"]}, "selected candidate index"),
        ("confirm_assignment", {"candidate_index": ["99"]}, "not part"),
        ("reassign_site", {}, "Select a candidate"),
        ("reassign_site", {"candidate_index": ["99"]}, "not part"),
        ("unsupported_site", {}, "require a label"),
        ("unsupported_site", {"unsupported_label": ["x"], "unsupported_atom": ["bad"]}, "must be an integer"),
        ("not-an-action", {}, "Unsupported review action"),
    ],
)
def test_stage1_web_actions_reject_incomplete_or_stale_submissions(action, form, message):
    with pytest.raises(ValueError, match=message):
        outlier_review._decision_for_action(
            pd.DataFrame(columns=outlier_review.DECISION_COLUMNS), _queue_row(), action, form
        )


def test_stage1_review_pack_renders_html_and_honors_pending_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    queue = pd.DataFrame([
        _queue_row().to_dict(),
        {**_queue_row().to_dict(), "review_id": "second", "review_status": "confirm_assignment"},
    ])
    rendered: list[str] = []

    def fake_render(row, _cache, image_dir):
        path = Path(image_dir) / f"{row['review_id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        rendered.append(str(row["review_id"]))
        return str(path)

    monkeypatch.setattr(outlier_review, "_render_row", fake_render)
    pending = outlier_review.review_outliers(
        queue, str(tmp_path / "decisions.csv"), str(tmp_path), str(tmp_path / "images"),
        str(tmp_path / "review.html"), include_reviewed=False, max_records=1,
        export_only=True, render_all=True, open_images=False,
    )
    assert pending == 1
    assert rendered == ["outlier-test", "second"]
    page = (tmp_path / "review.html").read_text()
    assert "Stage 1 outlier review pack" in page
    assert "primary_amine" in page


def test_stage1_actual_record_renderer_and_console_summary_create_human_audit_artifacts(tmp_path: Path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_sdf(raw / "toy.sdf", "CN")
    row = _queue_row()
    image = outlier_review._render_row(
        row, outlier_review.RawMolCache(str(raw)), str(tmp_path / "images")
    )
    assert Path(image).stat().st_size > 0
    candidates = outlier_review._runtime_candidates(row)
    outlier_review._print_review(row, candidates, image)
    printed = capsys.readouterr().out
    assert "experimental=10.000" in printed
    assert "primary_amine" in printed
    assert "commands:" in printed
    assert outlier_review._optional_number(None) == "not yet modeled"
    assert outlier_review._optional_number(float("nan")) == "not yet modeled"
    assert outlier_review._queue_rank_label(row) == "#1"
    row["review_trigger"] = "detector_schema_change_reopened_decision"
    assert outlier_review._queue_rank_label(row) == "reopened after detector change"
    row["review_trigger"] = "detector_v2_new_family_stage1_candidate"
    assert outlier_review._queue_rank_label(row) == "new detector family"


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("c manually checked", "confirm_assignment"),
        ("r 1 unsupported detector", "unsupported_site"),
        ("w unknown site", "exclude_wrong_site_unknown"),
        ("e endpoint mismatch", "exclude_wrong_endpoint"),
        ("v impossible value", "exclude_implausible_pka"),
        ("d revisit later", "defer"),
        ("u custom_site 0", "unsupported_site"),
    ],
)
def test_stage1_terminal_console_reviewer_persists_every_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, expected: str
):
    monkeypatch.setattr(outlier_review, "_render_row", lambda *_args: str(tmp_path / "image.png"))
    monkeypatch.setattr(outlier_review, "_print_review", lambda *_args: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: command)
    path = tmp_path / "decisions.csv"
    reviewed = outlier_review.review_outliers(
        pd.DataFrame([_queue_row().to_dict()]), str(path), str(tmp_path),
        str(tmp_path / "images"), str(tmp_path / "index.html"), open_images=False,
    )
    assert reviewed == 1
    saved = outlier_review._load_decisions(str(path)).iloc[0]
    assert saved["review_decision"] == expected
    if command.startswith("u "):
        assert saved["reviewed_site_label"] == "custom_site"


def test_stage1_console_reviewer_recovers_from_invalid_commands_then_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    monkeypatch.setattr(outlier_review, "_render_row", lambda *_args: str(tmp_path / "image.png"))
    monkeypatch.setattr(outlier_review, "_print_review", lambda *_args: None)
    commands = iter(["", "r nope", "r 99", "u", "u label bad", "unknown", "s"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    path = tmp_path / "decisions.csv"
    assert outlier_review.review_outliers(
        pd.DataFrame([_queue_row().to_dict()]), str(path), str(tmp_path),
        str(tmp_path / "images"), str(tmp_path / "index.html"), open_images=False,
    ) == 1
    assert not path.exists()
    output = capsys.readouterr().out
    assert "invalid candidate index" in output
    assert "atom index must be an integer" in output
    assert "unrecognized command" in output


def test_stage1_console_quit_saves_prior_decisions_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rows = [_queue_row().to_dict(), {**_queue_row().to_dict(), "review_id": "second"}]
    commands = iter(["c first", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    monkeypatch.setattr(outlier_review, "_render_row", lambda *_args: str(tmp_path / "image.png"))
    monkeypatch.setattr(outlier_review, "_print_review", lambda *_args: None)
    path = tmp_path / "decisions.csv"
    assert outlier_review.review_outliers(
        pd.DataFrame(rows), str(path), str(tmp_path), str(tmp_path / "images"),
        str(tmp_path / "index.html"), open_images=False,
    ) == 2
    assert len(outlier_review._load_decisions(str(path))) == 1


def test_stage1_main_orchestrates_and_prioritizes_evidence_queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    base = pd.DataFrame([_queue_row()])
    evidence = base.copy()
    evidence.loc[:, "review_trigger"] = "evidence_quality_ambiguous"
    expansion = base.copy()
    expansion.loc[:, "review_trigger"] = "detector_v2_new_family_stage1_candidate"
    monkeypatch.setattr(outlier_review, "build_outlier_review_queue", lambda **_kwargs: base.copy())
    monkeypatch.setattr(outlier_review, "build_detector_expansion_review_queue", lambda **_kwargs: expansion.copy())
    monkeypatch.setattr(outlier_review, "build_evidence_quality_review_queue", lambda **_kwargs: evidence.copy())
    monkeypatch.setattr(outlier_review, "build_detector_reopen_review_queue", lambda **_kwargs: pd.DataFrame())
    monkeypatch.setattr(outlier_review, "_refresh_review_status", lambda queue, _path: queue)
    captured: dict[str, object] = {}

    def fake_review(queue, **kwargs):
        captured["queue"] = queue.copy()
        captured["kwargs"] = kwargs
        return len(queue)

    monkeypatch.setattr(outlier_review, "review_outliers", fake_review)
    queue_path = tmp_path / "queue.csv"
    monkeypatch.setattr(sys, "argv", [
        "stage1_outlier_review.py", "--oof", "oof.csv", "--transitions", "transitions.csv",
        "--assignments", "assignments.csv", "--raw-dir", str(tmp_path),
        "--decisions", str(tmp_path / "decisions.csv"), "--queue", str(queue_path),
        "--detector-expansion-release", "release.csv",
        "--detector-expansion-queue", str(tmp_path / "expansion.csv"),
        "--evidence-quality-queue", str(tmp_path / "evidence.csv"), "--export-only",
    ])
    outlier_review.main()
    combined = captured["queue"]
    assert len(combined) == 1
    assert combined.iloc[0]["review_trigger"] == "evidence_quality_ambiguous"
    assert queue_path.exists()
    assert Path(tmp_path / "expansion.csv").exists()
    assert Path(tmp_path / "evidence.csv").exists()


def test_stage1_main_reopen_and_web_paths_render_missing_images_before_serving(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    queue = pd.DataFrame([_queue_row().to_dict()])
    expansion_path = tmp_path / "existing-expansion.csv"
    queue.to_csv(expansion_path, index=False)
    reopen_report = tmp_path / "reopen.csv"
    captured: dict[str, object] = {}
    monkeypatch.setattr(outlier_review, "reopen_reviews_for_detector_change", lambda **_kwargs: pd.DataFrame([{"review_id": "reopened"}]))
    monkeypatch.setattr(outlier_review, "build_outlier_review_queue", lambda **_kwargs: pd.DataFrame())
    monkeypatch.setattr(outlier_review, "build_evidence_quality_review_queue", lambda **_kwargs: pd.DataFrame())
    monkeypatch.setattr(outlier_review, "build_detector_reopen_review_queue", lambda **_kwargs: queue.copy())
    monkeypatch.setattr(outlier_review, "_refresh_review_status", lambda frame, _path: frame)

    def fake_render(row, _cache, image_dir):
        path = Path(image_dir) / f"{row['review_id']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
        return str(path)

    monkeypatch.setattr(outlier_review, "_render_row", fake_render)
    monkeypatch.setattr(outlier_review, "serve_web_reviewer", lambda frame, **kwargs: captured.update({"queue": frame.copy(), "kwargs": kwargs}))
    monkeypatch.setattr(sys, "argv", [
        "stage1_outlier_review.py", "--web", "--reopen-detector-gaps",
        "--oof", "oof.csv", "--transitions", "transitions.csv", "--assignments", "assignments.csv",
        "--raw-dir", str(tmp_path), "--decisions", str(tmp_path / "decisions.csv"),
        "--decision-history", str(tmp_path / "history.csv"), "--reopen-report", str(reopen_report),
        "--queue", str(tmp_path / "queue.csv"), "--detector-expansion-queue", str(expansion_path),
        "--evidence-quality-queue", str(tmp_path / "evidence.csv"), "--image-dir", str(tmp_path / "images"),
        "--host", "127.0.0.1", "--port", "9999",
    ])
    outlier_review.main()
    assert len(captured["queue"]) == 1
    assert captured["kwargs"]["port"] == 9999
    assert (tmp_path / "images/outlier-test.png").exists()


def test_stage1_web_server_serves_images_validates_routes_and_persists_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    queue = pd.DataFrame([_queue_row().to_dict()])
    decisions = tmp_path / "decisions.csv"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    handler = _capture_handler(
        monkeypatch, outlier_review, outlier_review.serve_web_reviewer,
        queue, str(decisions), str(image_dir), "127.0.0.1", 0,
    )
    status, _, body = _handler_request(handler, "GET", "/")
    assert status == 200 and b"Stage 1 pKa review" in body
    assert _handler_request(handler, "GET", "/not-a-route")[0] == 404
    assert _handler_request(handler, "GET", "/image/unknown.png")[0] == 404
    assert _handler_request(handler, "GET", "/image/outlier-test.png")[0] == 404
    (image_dir / "outlier-test.png").write_bytes(b"test-png")
    image_response = _handler_request(handler, "GET", "/image/outlier-test.png")
    assert image_response[0] == 200 and image_response[2] == b"test-png"
    assert _handler_request(handler, "POST", "/wrong", {})[0] == 404
    assert _handler_request(handler, "POST", "/decision", {"review_id": "unknown", "action": "defer"})[0] == 400
    saved = _handler_request(handler, "POST", "/decision", {
        "review_id": "outlier-test", "action": "defer", "note": "web checked",
    })
    assert saved[0] == 303 and saved[1]["Location"] == "/"
    persisted = outlier_review._load_decisions(str(decisions)).iloc[0]
    assert persisted["review_decision"] == "defer"
    assert persisted["review_note"] == "web checked"
    complete = _handler_request(handler, "GET", "/")[2]
    assert b"This review pass is complete" in complete


def _structure_review_inputs(tmp_path: Path) -> dict[str, Path]:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_sdf(raw / "toy.sdf", "CN")
    comparison = tmp_path / "comparison.csv"
    pd.DataFrame([
        {
            "molecule_id": "mol-1", "site_id": "site-1", "site_label": "primary_amine",
            "site_family": "amine", "validation_scope": "stage1_scaffold_oof_single_site",
            "population_ph": 7.4, "stage3_predicted_site_form_at_ph": "base_form",
            "stage3_form_vs_experimental_pka_check": "disagrees",
            "atom_mapped_smiles": "[CH3:1][NH2:2]", "input_member_form": "base_form",
            "deployed_assigned_macro_pka": 8.0, "deployed_assigned_local_pka": 8.2,
            "deployed_assigned_one_body_pka": 8.2, "stage3_reconstructed_macro_pka": 8.0,
            "stage3_contextual_edge_pka_at_dominant_background": 7.8,
            "stage3_contextual_edge_pka_min": 7.5, "stage3_contextual_edge_pka_max": 8.5,
            "stage3_protonated_probability_at_ph": 0.2,
            "dominant_atom_mapped_smiles": "[CH3:1][NH2:2]",
            "site_atom_maps_json": "[2]", "site_center_maps_json": "[2]",
        },
        {
            "molecule_id": "not-in-network", "site_id": "x", "site_label": "x", "site_family": "x",
            "validation_scope": "x", "population_ph": 7.4,
            "stage3_predicted_site_form_at_ph": "base_form",
            "stage3_form_vs_experimental_pka_check": "disagrees",
        },
    ]).to_csv(comparison, index=False)
    measurement = {
        "experimental_pka": 10.0, "source_file": "toy.sdf", "record_index": 0,
        "source_assay_id": "A", "source_document_id": "D", "source_molecule_id": "M",
        "source_site_evidence": "source_atom_index", "source_site_evidence_confidence": "high",
        "transition_identity_confidence": "high", "network_mapping_basis": "exact",
    }
    sites = [{"site_id": "site-1", "experimental_measurements": [measurement]}]
    nodes = [
        {"site_forms": {"site-1": "acid_form"}, "stage3_configuration_population_at_ph": 0.7,
         "reference_atom_mapped_smiles": "[CH3:1][NH3+:2]"},
        {"site_forms": {"site-1": "base_form"}, "stage3_configuration_population_at_ph": 0.3,
         "reference_atom_mapped_smiles": "[CH3:1][NH2:2]"},
    ]
    networks = tmp_path / "networks.csv"
    pd.DataFrame([{
        "molecule_id": "mol-1", "sites_json": json.dumps(sites),
        "stage3_microstate_nodes_json": json.dumps(nodes),
    }]).to_csv(networks, index=False)
    return {"raw": raw, "comparison": comparison, "networks": networks}


def test_structure_consistency_queue_proposes_pka_compatible_microstate_with_provenance(tmp_path: Path):
    inputs = _structure_review_inputs(tmp_path)
    queue = structure_review.build_review_queue(
        str(inputs["comparison"]), str(inputs["networks"]), str(inputs["raw"]),
        str(tmp_path / "decisions.csv"),
    )
    assert len(queue) == 1
    row = queue.iloc[0]
    assert row["experimental_pka_implied_form"] == "acid_form"
    assert row["proposed_pka_consistent_smiles"] == "C[NH3+]"
    assert row["original_smiles"] == "CN"
    assert row["source_assay_id"] == "A"
    assert row["absolute_experimental_vs_deployed_macro_gap"] == pytest.approx(2.0)


def test_structure_review_helpers_cover_boundaries_and_unavailable_nodes():
    assert structure_review._json_list("bad") == []
    assert structure_review._json_list('{"not":"a list"}') == []
    assert structure_review._canonical_unmapped_smiles("[CH3:1][NH3+:2]") == "C[NH3+]"
    assert structure_review._canonical_unmapped_smiles("bad smiles") == ""
    assert structure_review._implied_form(7.4, 7.4) == (0.5, "equal_acid_and_base")
    assert structure_review._implied_form(1000, 7.4)[1] == "acid_form"
    assert structure_review._implied_form(-1000, 7.4)[1] == "base_form"
    assert structure_review._conditional_node([], "site", "acid_form") is None
    assert structure_review._optional_float(None) is None
    assert structure_review._optional_float(float("nan")) is None
    assert structure_review._optional_float("2.5") == pytest.approx(2.5)


def test_replacement_validation_rejects_parse_and_connectivity_changes():
    with pytest.raises(ValueError, match="original SMILES"):
        structure_review.validate_replacement_smiles("bad", "CC")
    with pytest.raises(ValueError, match="Replacement SMILES"):
        structure_review.validate_replacement_smiles("CC", "bad")
    with pytest.raises(ValueError, match="connectivity"):
        structure_review.validate_replacement_smiles("C1CC1", "CCC")


@pytest.mark.parametrize(
    ("action", "replacement", "mode"),
    [
        ("keep_current_structure_and_transition", "ignored", "ignored"),
        ("exclude_measurement_structure_pair", "ignored", "ignored"),
        ("exclude_entire_source_record", "ignored", "ignored"),
        ("correct_transition_definition", "ignored", "basic_protonation"),
        ("defer", "ignored", "ignored"),
    ],
)
def test_structure_decision_actions_are_upserted_with_only_relevant_fields(tmp_path: Path, action, replacement, mode):
    path = tmp_path / "decisions.csv"
    row = pd.Series({
        "review_id": "review-1", "source_file": "toy.sdf", "record_index": 0,
        "pka_value": 10.0, "molecule_id": "mol-1", "site_id": "site-1", "original_smiles": "CN",
    })
    structure_review._save_decision(str(path), row, action, "checked", replacement, mode)
    saved = structure_review._load_decisions(str(path)).iloc[0]
    assert saved["review_decision"] == action
    assert pd.isna(saved["replacement_smiles"]) or saved["replacement_smiles"] == ""
    if action == "correct_transition_definition":
        assert saved["replacement_transition_mode"] == mode
    else:
        assert pd.isna(saved["replacement_transition_mode"]) or saved["replacement_transition_mode"] == ""
    with pytest.raises(ValueError, match="Unsupported action"):
        structure_review._save_decision(str(path), row, "invalid", "", "", "")


def test_structure_review_page_and_images_render_complete_audit_artifacts(tmp_path: Path):
    inputs = _structure_review_inputs(tmp_path)
    queue = structure_review.build_review_queue(
        str(inputs["comparison"]), str(inputs["networks"]), str(inputs["raw"]),
        str(tmp_path / "decisions.csv"),
    )
    image_dir = tmp_path / "images"
    structure_review.render_review_images(queue, str(image_dir))
    assert (image_dir / f"{queue.iloc[0]['review_id']}.png").stat().st_size > 0
    page = structure_review._page(queue, queue.iloc[0], {}, "test error")
    assert "Transition warning" in page and "test error" in page
    complete = structure_review._page(queue, None, {queue.iloc[0]["review_id"]: "defer"})
    assert "review complete" in complete and "1 deferred" in complete


def test_structure_review_rendering_skips_unparseable_panels_without_leaving_false_artifacts(tmp_path: Path):
    row = pd.Series({
        "review_id": "bad", "representative_atom_mapped_smiles": "bad",
        "proposed_pka_consistent_atom_mapped_smiles": "bad",
        "stage3_dominant_atom_mapped_smiles": "bad", "input_member_form": "base_form",
        "pka_value": 7.0, "population_ph": 7.4,
        "experimental_pka_implied_form": "base_form",
        "experimental_pka_implied_protonated_probability": 0.2,
        "stage3_predicted_form": "base_form", "stage3_protonated_probability": 0.2,
        "stage2_predicted_macro_pka": 7.0, "stage3_one_body_pka": 7.0,
        "stage3_contextual_edge_pka_at_dominant_background": None,
        "site_atom_maps_json": "[]", "site_center_maps_json": "[]",
    })
    structure_review.render_review_images(pd.DataFrame([row]), str(tmp_path / "images"))
    assert not list((tmp_path / "images").glob("*.png"))
    assert structure_review._drawing("bad", "[]", "[]") == (None, [], {})


def test_structure_review_main_build_only_writes_queue_decisions_and_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    inputs = _structure_review_inputs(tmp_path)
    queue_path = tmp_path / "audit/queue.csv"
    decisions = tmp_path / "curation/decisions.csv"
    images = tmp_path / "audit/images"
    monkeypatch.setattr(sys, "argv", [
        "pka_structure_consistency_review.py", "--comparison", str(inputs["comparison"]),
        "--networks", str(inputs["networks"]), "--raw-dir", str(inputs["raw"]),
        "--decisions", str(decisions), "--queue", str(queue_path),
        "--image-dir", str(images), "--build-only",
    ])
    structure_review.main()
    assert len(pd.read_csv(queue_path)) == 1
    assert list(pd.read_csv(decisions).columns) == structure_review.DECISION_COLUMNS
    assert list(images.glob("*.png"))


def test_structure_review_web_server_handles_navigation_validation_and_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    inputs = _structure_review_inputs(tmp_path)
    decisions = tmp_path / "decisions.csv"
    queue = structure_review.build_review_queue(
        str(inputs["comparison"]), str(inputs["networks"]), str(inputs["raw"]), str(decisions)
    )
    image_dir = tmp_path / "images"
    structure_review.render_review_images(queue, str(image_dir))
    review_id = str(queue.iloc[0]["review_id"])
    handler = _capture_handler(
        monkeypatch, structure_review, structure_review.serve_web_reviewer,
        queue, str(decisions), str(image_dir), "127.0.0.1", 0,
    )
    root = _handler_request(handler, "GET", "/")
    assert root[0] == 200 and review_id.encode() in root[2]
    requested = _handler_request(handler, "GET", f"/?review_id={review_id}")
    assert requested[0] == 200 and b"Transition warning" in requested[2]
    assert _handler_request(handler, "GET", "/image/missing.png")[0] == 404
    image = _handler_request(handler, "GET", f"/image/{review_id}.png")
    assert image[0] == 200 and image[2].startswith(b"\x89PNG")
    assert _handler_request(handler, "POST", "/", {"review_id": "unknown", "action": "defer"})[0] == 400
    invalid = _handler_request(handler, "POST", "/", {
        "review_id": review_id, "action": "replace_input_smiles",
        "replacement_smiles": "not smiles",
    })
    assert invalid[0] == 400 and b"not parseable" in invalid[2]
    saved = _handler_request(handler, "POST", "/", {
        "review_id": review_id, "action": "defer", "note": "web deferred",
    })
    assert saved[0] == 303 and saved[1]["Location"] == "/"
    persisted = structure_review._load_decisions(str(decisions)).iloc[0]
    assert persisted["review_decision"] == "defer"
    assert persisted["review_note"] == "web deferred"
    assert b"review complete" in _handler_request(handler, "GET", "/")[2]


def _failure_row() -> pd.Series:
    return pd.Series({
        "molecule_id": "mol-1", "site_id": "site-1", "site_label": "primary_amine",
        "site_family": "amine", "validation_scope": "stage2_scaffold_validation_multisite",
        "validation_regime": "scaffold", "experimental_anchor_pka": 10.0,
        "validation_predicted_pka": 5.0, "signed_error": -5.0, "abs_error": 5.0,
        "input_smiles": "CN", "atom_mapped_smiles": "[CH3:1][NH2:2]",
        "dominant_atom_mapped_smiles": "[CH3:1][NH3+:2]", "input_member_form": "base_form",
        "site_atom_maps_json": "[2]", "site_center_maps_json": "[2]", "population_ph": 7.4,
        "stage2_predicted_macro_pka": 8.0, "deployed_assigned_macro_pka": 8.0,
        "stage3_reconstructed_macro_pka": 8.1, "stage3_macro_reconstruction_abs_error": 0.1,
        "stage3_one_body_pka": 8.2, "deployed_assigned_one_body_pka": 8.2,
        "stage3_effective_local_pka": 8.2, "stage3_effective_local_pka_ci_low": 7.0,
        "stage3_effective_local_pka_ci_high": 9.0,
        "stage3_contextual_edge_pka_at_dominant_background": 8.0,
        "stage3_contextual_edge_pka_min": 7.5, "stage3_contextual_edge_pka_max": 8.5,
        "stage3_protonated_probability_at_ph": 0.8,
        "stage3_protonated_probability_ci_low": 0.6, "stage3_protonated_probability_ci_high": 0.9,
        "stage3_predicted_site_form_at_ph": "acid_form",
        "stage3_overall_site_state_confidence": 0.8,
        "stage3_empirical_pka_side_call_confidence": 0.9,
        "experimental_pka_implied_protonated_probability_at_population_ph": 0.997,
        "experimental_pka_implied_dominant_site_form": "acid_form",
        "stage3_form_vs_experimental_pka_check": "agrees", "population_status": "complete_test",
        "secondary_guess_review_recommended": False,
    })


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"validation_scope": "stage1_scaffold_oof_single_site"}, "not_a_multisite_stage2_validation_row"),
        ({"experimental_anchor_pka": None}, "missing_experimental_or_validation_value"),
        ({"network_sites_json": "bad"}, "no_alternative_enumerated_transition"),
        ({"network_sites_json": "[]"}, "no_alternative_enumerated_transition"),
        ({"network_sites_json": '[{"site_id":"site-1"},{"site_id":"site-2","rank_associated_macro_pka":8}]', "stage1_network_macro_pka": None}, "primary_stage1_network_pka_unavailable"),
        ({"network_sites_json": '[{"site_id":"site-1","rank_associated_macro_pka":5},{"site_id":"site-2"}]', "stage1_network_macro_pka": 5}, "alternative_stage1_network_pka_unavailable"),
    ],
)
def test_secondary_transition_guess_reports_why_a_suggestion_is_unavailable(updates, reason):
    row = _failure_row()
    row["network_sites_json"] = "[]"
    for key, value in updates.items():
        row[key] = value
    result = failure_viz._secondary_transition_guess(row)
    assert result["secondary_guess_reason"] == reason
    assert not result["secondary_guess_available"]


def test_secondary_transition_guess_low_confidence_reasons_do_not_mutate_assignment():
    row = _failure_row()
    row["stage1_network_macro_pka"] = 5.0
    row["network_sites_json"] = json.dumps([
        {"site_id": "site-1", "rank_associated_macro_pka": 5.0},
        {"site_id": "site-2", "label": "phenol", "family": "phenol_phenolate", "rank_associated_macro_pka": 8.0},
    ])
    below = failure_viz._secondary_transition_guess(row, error_threshold=6.0)
    assert below["secondary_guess_reason"] == "primary_heldout_error_below_review_threshold"
    weak = failure_viz._secondary_transition_guess(row, min_improvement=4.0)
    assert weak["secondary_guess_reason"] == "alternative_does_not_improve_enough"
    far = failure_viz._secondary_transition_guess(row, max_secondary_abs_error=1.0)
    assert far["secondary_guess_reason"] == "alternative_remains_too_far_from_experimental_pka"
    assert row["site_id"] == "site-1"


def test_failure_visualization_helpers_cover_boundaries_and_legends():
    assert failure_viz._finite_float("bad") is None
    assert failure_viz._finite_float(float("nan")) is None
    assert failure_viz._finite_float("2.5") == pytest.approx(2.5)
    assert failure_viz._experimental_pka_implied_form(None, 7.4)[1] == "unavailable"
    assert failure_viz._experimental_pka_implied_form(1000, 7.4) == (1.0, "acid_form")
    assert failure_viz._experimental_pka_implied_form(-1000, 7.4) == (0.0, "base_form")
    assert failure_viz._experimental_pka_implied_form(7.4, 7.4)[1] == "equal_acid_and_base"
    assert failure_viz._form_agreement_check(pd.Series()) == "unavailable_or_boundary"
    assert failure_viz._parse_maps("bad") == []
    assert failure_viz._fmt(None) == "NA"
    row = _failure_row()
    categories = [
        "heldout_pka_errors", "secondary_transition_assignment_guesses",
        "low_confidence_state_calls_at_ph_7_4", "empirical_pka_side_conflicts",
        "stage2_free_energy_reconstruction_gaps", "pair_coupling_changed_site_calls",
        "incomplete_microstate_networks",
    ]
    for category in categories:
        assert isinstance(failure_viz._legend(row, category), str)
    assert "INPUT DRAWING" in failure_viz._comparison_legend(row, "input")
    assert "DOMINANT MICROSTATE" in failure_viz._comparison_legend(row, "stage3")


def test_failure_galleries_render_valid_molecules_skip_invalid_and_keep_full_csv(tmp_path: Path):
    valid = _failure_row()
    invalid = _failure_row().copy()
    invalid["molecule_id"] = "invalid"
    invalid["atom_mapped_smiles"] = "bad smiles"
    rows = pd.DataFrame([valid, invalid])
    png = tmp_path / "gallery.png"
    csv = tmp_path / "gallery.csv"
    assert failure_viz.render_gallery(rows, "heldout_pka_errors", png, csv, 10) == 1
    assert png.stat().st_size > 0
    assert len(pd.read_csv(csv)) == 2

    comparison_png = tmp_path / "comparison.png"
    comparison_csv = tmp_path / "comparison.csv"
    assert failure_viz.render_deployment_comparison_gallery(
        rows, comparison_png, comparison_csv, 10
    ) == 1
    assert comparison_png.stat().st_size > 0
    assert len(pd.read_csv(comparison_csv)) == 2


def test_stage3_failure_visualization_main_writes_all_review_categories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    stage1 = tmp_path / "stage1.csv"
    pd.DataFrame([{
        "molecule_id": "mol-1", "site_id": "site-1", "candidate_label": "primary_amine",
        "site_family": "amine", "experimental_anchor_pka": 10.0, "pred_intrinsic_pka": 5.0,
    }]).to_csv(stage1, index=False)
    stage2 = tmp_path / "stage2.csv"
    pd.DataFrame(columns=[
        "molecule_id", "site_id", "site_label", "site_family", "experimental_anchor_pka",
        "stage2_supported_projected_macro_pka",
    ]).to_csv(stage2, index=False)
    sites_row = _failure_row().to_dict()
    sites_row.update({
        "stage1_intrinsic_pka": 5.0, "stage3_site_state_decision_confidence": 0.1,
        "stage3_overall_site_state_confidence": 0.1,
        "stage3_empirical_pka_side_call_confidence": 0.4,
        "stage3_macro_reconstruction_abs_error": 0.5,
        "stage3_site_call_changed_by_pair_coupling": True,
        "stage3_uncoupled_protonated_probability_at_ph": 0.2,
        "population_status": "limited",
    })
    sites = tmp_path / "sites.csv"
    pd.DataFrame([sites_row]).to_csv(sites, index=False)
    rendered_categories: list[str] = []

    def fake_gallery(rows, category, output_png, output_csv, max_examples):
        rendered_categories.append(category)
        rows.to_csv(output_csv, index=False)
        return len(rows)

    monkeypatch.setattr(failure_viz, "render_gallery", fake_gallery)
    monkeypatch.setattr(
        failure_viz, "render_deployment_comparison_gallery",
        lambda rows, output_png, output_csv, max_examples: (rows.to_csv(output_csv, index=False) or len(rows)),
    )
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "visualize_canonical_stage3_failures.py", "--stage1-oof", str(stage1),
        "--stage2-eval", str(stage2), "--stage3-sites", str(sites),
        "--out-dir", str(out), "--pka-error-threshold", "3",
    ])
    failure_viz.main()
    assert len(rendered_categories) == 7
    summary = json.loads((out / "stage3_failure_summary.json").read_text())
    assert summary["heldout_pka_error_rows"] == 1
    assert summary["low_confidence_molecules"] == 1
    assert summary["incomplete_microstate_network_molecules"] == 1
    assert "Canonical Stage 3 review galleries" in (out / "index.html").read_text()

