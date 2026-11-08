"""Tests for provenance-safe pKahub transition assignability auditing."""

import json

import pandas as pd
import pytest

from audit_pkahub_transition_assignability import audit


def _row(molid, smiles, method="epikx.match", source="example"):
    return {
        "pka_value": 4.5,
        "data_source": source,
        "charge_state_pre": 0,
        "charge_state_post": -1,
        "microspecies_pre": "",
        "microspecies_post": "",
        "molid": molid,
        "smiles": smiles,
        "rawdataID": molid,
        "temperature": 25.0,
        "ionic_strength": None,
        "assignment_method": method,
        "assignment_error": 0.2,
        "primary_source": "fixture",
    }


def test_audit_separates_structural_certainty_from_teacher_assignments(tmp_path):
    source = tmp_path / "all.tsv"
    pd.DataFrame([
        _row("acid", "CC(=O)O"),
        _row("ampholyte", "NCC(=O)O"),
        _row("none", "c1ccccc1", method="manual"),
        _row("bad", "notasmiles"),
    ]).to_csv(source, sep="\t", index=False)

    summary = audit(source, tmp_path / "audit")
    assignments = pd.read_csv(tmp_path / "audit/molecule_assignability.csv")

    assert summary["input_rows"] == 4
    assert summary["assignment_provenance"]["epik_derived_rows"] == 3
    assert dict(zip(assignments.molid, assignments.status)) == {
        "acid": "clean_single_transition",
        "ampholyte": "multiple_candidate_transitions",
        "none": "no_supported_transition",
        "bad": "invalid_smiles",
    }
    assert json.loads((tmp_path / "audit/summary.json").read_text()) == summary


def test_audit_rejects_conflicting_molid_structures(tmp_path):
    source = tmp_path / "all.tsv"
    pd.DataFrame([
        _row("same", "CC(=O)O"),
        _row("same", "CN"),
    ]).to_csv(source, sep="\t", index=False)

    with pytest.raises(ValueError, match="more than one SMILES"):
        audit(source, tmp_path / "audit")
