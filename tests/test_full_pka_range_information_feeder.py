import csv
import io
import json
import sys

import pytest

import full_pka_range_information_feeder as feeder
from full_pka_range_information_feeder import (
    BASE_COLUMNS,
    FeederState,
    load_rows,
    normalize_answer,
)


def _write_template(path, enabled=False):
    row = {column: "" for column in BASE_COLUMNS}
    row.update({
        "transition_id": "toy_cation_to_neutral",
        "site_family": "toy_basicity",
        "acid_form_label": "toy_acid",
        "base_form_label": "toy_base",
        "acid_charge_relative_to_neutral": "+1",
        "base_charge_relative_to_neutral": "0",
        "measurement_kind": "macro",
        "allowed_protonation_atom_rule": "all ring N",
        "reference_pka": "5.2" if enabled else "",
        "reference_uncertainty": "0.3" if enabled else "",
        "solvent": "water" if enabled else "",
        "temperature_K": "298.15" if enabled else "",
        "source_citation": "DOI:test" if enabled else "",
        "enable": "true" if enabled else "false",
        "notes": "Toy transition",
    })
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def test_unknown_is_a_valid_submitted_answer_and_requests_research(tmp_path):
    template = tmp_path / "template.csv"
    _write_template(template)
    row = load_rows(str(template), str(tmp_path / "missing.csv"))[0]
    result = normalize_answer(row, {
        "support_decision": "want_support",
        "pka_input_mode": "unknown",
        "research_requested": True,
        "submission_status": "submitted",
    })
    assert result["submission_status"] == "submitted"
    assert result["research_requested"] == "true"
    assert result["activation_readiness"] == "research_requested"
    assert result["enable"] == "false"


def test_range_is_preserved_and_converted_to_midpoint_plus_uncertainty(tmp_path):
    template = tmp_path / "template.csv"
    _write_template(template)
    row = load_rows(str(template), str(tmp_path / "missing.csv"))[0]
    result = normalize_answer(row, {
        "support_decision": "want_support",
        "pka_input_mode": "range",
        "reference_pka_low": "4",
        "reference_pka_high": "6",
        "conditions_status": "aqueous_ambient",
        "measurement_kind": "macro",
        "source_citation": "from memory",
    })
    assert result["reference_pka"] == "5"
    assert result["reference_uncertainty"] == "1"
    assert result["solvent"] == "water"
    assert result["temperature_K"] == "298.15"
    assert result["activation_readiness"] == "ready_for_codex_validation"
    assert result["enable"] == "false"


def test_invalid_range_has_a_plain_error(tmp_path):
    template = tmp_path / "template.csv"
    _write_template(template)
    row = load_rows(str(template), str(tmp_path / "missing.csv"))[0]
    with pytest.raises(ValueError, match="low ≤ high"):
        normalize_answer(row, {
            "pka_input_mode": "range",
            "reference_pka_low": "9",
            "reference_pka_high": "3",
        })


def test_state_autosaves_without_mutating_template(tmp_path):
    template = tmp_path / "template.csv"
    responses = tmp_path / "responses.csv"
    _write_template(template)
    state = FeederState(str(template), str(responses))
    saved = state.save({
        "transition_id": "toy_cation_to_neutral",
        "support_decision": "not_needed",
        "submission_status": "submitted",
    })
    assert responses.exists()
    assert saved["activation_readiness"] == "not_requested"
    reloaded = load_rows(str(template), str(responses))[0]
    assert reloaded["support_decision"] == "not_needed"
    assert reloaded["submission_status"] == "submitted"
    assert reloaded["acid_form_label"] == "toy_acid"


def test_validated_template_overlay_replaces_stale_missing_metadata(tmp_path):
    template = tmp_path / "template.csv"
    responses = tmp_path / "responses.csv"
    _write_template(template, enabled=True)
    row = load_rows(str(template), str(responses))[0]
    assert row["enable"] == "true"
    assert row["reference_pka"] == "5.2"
    assert row["activation_readiness"] == "validated_and_enabled"
    assert row["missing_for_activation"] == ""


def test_loading_rejects_empty_and_incomplete_templates(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match="No transition rows"):
        load_rows(str(missing), str(tmp_path / "responses.csv"))

    incomplete = tmp_path / "incomplete.csv"
    incomplete.write_text("transition_id\nexample\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        load_rows(str(incomplete), str(tmp_path / "responses.csv"))


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ({"support_decision": "maybe"}, "Invalid support"),
        ({"pka_input_mode": "guess"}, "Invalid pKa"),
        ({"measurement_kind": "mesoscopic"}, "Measurement kind"),
        ({"pka_input_mode": "exact", "reference_pka": ""}, "Enter the pKa"),
        ({"pka_input_mode": "exact", "reference_pka": "seven"}, "Not a number"),
        ({"temperature_K": "warm"}, "Not a number"),
        ({"ionic_strength_M": "some"}, "Not a number"),
    ],
)
def test_normalize_answer_rejects_invalid_enumerations_and_numbers(
    tmp_path, answer, message
):
    template = tmp_path / "template.csv"
    _write_template(template)
    row = load_rows(str(template), str(tmp_path / "responses.csv"))[0]
    with pytest.raises(ValueError, match=message):
        normalize_answer(row, answer)


def test_normalize_answer_reports_all_missing_metadata_and_summary(tmp_path):
    template = tmp_path / "template.csv"
    _write_template(template)
    row = load_rows(str(template), str(tmp_path / "responses.csv"))[0]
    result = normalize_answer(row, {
        "support_decision": "want_support",
        "pka_input_mode": "unknown",
        "research_requested": False,
    })
    assert result["activation_readiness"] == "needs_research_or_metadata"
    assert set(result["missing_for_activation"].split("; ")) == {
        "pKa value or research", "solvent", "temperature",
        "macro/micro identity", "source",
    }
    unsure = normalize_answer(row, {"support_decision": "unsure"})
    summary = feeder.summarize([result, unsure])
    assert summary["total"] == 2
    assert summary["want_support"] == 1
    assert summary["unsure"] == 1


def test_state_rejects_unknown_transition_and_returns_defensive_rows(tmp_path):
    template = tmp_path / "template.csv"
    responses = tmp_path / "responses.csv"
    _write_template(template)
    state = FeederState(str(template), str(responses))
    public = state.public_rows()
    public[0]["support_decision"] = "tampered"
    assert state.rows[0]["support_decision"] == "unsure"
    with pytest.raises(ValueError, match="Unknown transition_id"):
        state.save({"transition_id": "missing"})


def _invoke_handler(state, method, path, body=None, declared_length=None):
    handler = object.__new__(feeder.make_handler(state))
    encoded = b"" if body is None else (
        body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    )
    handler.path = path
    handler.headers = {
        "Content-Length": str(len(encoded) if declared_length is None else declared_length)
    }
    handler.rfile = io.BytesIO(encoded)
    handler.wfile = io.BytesIO()
    result = {"status": None, "headers": {}}
    handler.send_response = lambda status: result.update(status=status)
    handler.send_header = lambda key, value: result["headers"].__setitem__(key, value)
    handler.end_headers = lambda: None
    getattr(handler, f"do_{method}")()
    return result["status"], result["headers"].get("Content-Type", ""), handler.wfile.getvalue()


def test_http_handler_serves_ui_state_download_and_valid_save(tmp_path):
    template = tmp_path / "template.csv"
    responses = tmp_path / "responses.csv"
    _write_template(template)
    state = FeederState(str(template), str(responses))
    status, content_type, body = _invoke_handler(state, "GET", "/")
    assert status == 200 and content_type.startswith("text/html")
    assert b"Full-range pKa information feeder" in body
    status, _, body = _invoke_handler(state, "GET", "/api/state")
    assert status == 200 and json.loads(body)["summary"]["total"] == 1
    status, content_type, body = _invoke_handler(state, "GET", "/api/download.csv")
    assert status == 200 and content_type.startswith("text/csv")
    assert b"transition_id" in body
    status, _, body = _invoke_handler(state, "GET", "/missing")
    assert status == 404 and json.loads(body)["error"] == "Not found"

    status, _, body = _invoke_handler(state, "POST", "/api/save", {
        "transition_id": "toy_cation_to_neutral",
        "support_decision": "not_needed",
        "submission_status": "submitted",
    })
    saved = json.loads(body)
    assert status == 200
    assert saved["row"]["activation_readiness"] == "not_requested"
    assert saved["summary"]["submitted"] == 1


def test_http_handler_rejects_wrong_routes_sizes_json_and_payload_types(tmp_path):
    template = tmp_path / "template.csv"
    responses = tmp_path / "responses.csv"
    _write_template(template)
    state = FeederState(str(template), str(responses))
    assert _invoke_handler(state, "POST", "/wrong", {})[0] == 404
    assert _invoke_handler(state, "POST", "/api/save", b"")[0] == 400
    assert _invoke_handler(state, "POST", "/api/save", b"not-json")[0] == 400
    assert _invoke_handler(state, "POST", "/api/save", ["not", "an", "object"])[0] == 400
    assert _invoke_handler(state, "POST", "/api/save", {"transition_id": "unknown"})[0] == 400
    assert _invoke_handler(state, "POST", "/api/save", b"{}", declared_length=1_000_001)[0] == 400


def test_feeder_main_opens_browser_and_closes_server(
    tmp_path, monkeypatch, capsys
):
    template = tmp_path / "template.csv"
    responses = tmp_path / "responses.csv"
    _write_template(template)
    events = []

    class FakeServer:
        def __init__(self, address, handler):
            events.append(("init", address, handler))

        def serve_forever(self):
            events.append(("serve",))
            raise KeyboardInterrupt

        def server_close(self):
            events.append(("close",))

    monkeypatch.setattr(feeder, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(feeder.webbrowser, "open", lambda url: events.append(("open", url)))
    monkeypatch.setattr(sys, "argv", [
        "full_pka_range_information_feeder.py",
        "--template", str(template),
        "--responses", str(responses),
        "--host", "127.0.0.2",
        "--port", "9876",
    ])
    feeder.main()
    assert ("open", "http://127.0.0.2:9876") in events
    assert events[-1] == ("close",)
    assert "autosave_file=" in capsys.readouterr().out
