import csv

import pytest

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
