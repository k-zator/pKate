"""CASP16 ligand pKa ingestion and supervision-boundary contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import prepare_casp16_training_data as casp16
import acquire_casp16_complex_structures as casp16_structures


def test_casp16_uses_experimental_columns_and_quarantines_ambiguous_sites(tmp_path):
    source = tmp_path / "chymasecatgSI.txt"
    pd.DataFrame([
        {
            "compound id": "acid", "structure": "CC(=O)O", "pdb": "1ABC",
            "series": "toy", "PKA1": "4.75", "PKA1_CLASS": "ACID",
            "PKA2": "nd", "PKA2_CLASS": "",
            "ADMET Predictor PKA": 99.0,
        },
        {
            "compound id": "multi", "structure": "NCC(=O)O", "pdb": "2ABC",
            "series": "toy", "PKA1": "2.3", "PKA1_CLASS": "ACID",
            "PKA2": "9.6", "PKA2_CLASS": "BASE",
            "ADMET Predictor PKA": -99.0,
        },
        {
            "compound id": "censored", "structure": "CN", "pdb": "",
            "series": "toy", "PKA1": ">10", "PKA1_CLASS": "BASE",
            "PKA2": "", "PKA2_CLASS": "",
            "ADMET Predictor PKA": 10.1,
        },
    ]).to_csv(source, sep=";", index=False)
    output = tmp_path / "prepared"
    report = casp16.prepare(
        source, output, tmp_path / "missing_stage1.csv",
        tmp_path / "missing_network.csv",
    )

    stage1 = pd.read_csv(output / "stage1_supplemental_sites.csv")
    macro = pd.read_csv(output / "macro_set_transition_placeholders.csv")
    quarantine = pd.read_csv(output / "stage1_supplemental_quarantine.csv")
    assert report["numeric_experimental_pka_values"] == 3
    assert 99.0 not in set(stage1["pka_value"])
    assert -99.0 not in set(stage1["pka_value"])
    assert stage1["computational_pka_values_used"].eq(False).all()
    assert stage1["computed_pka_columns_ignored"].eq(True).all()
    assert set(stage1["training_cohort"]) == {
        "casp16_experimental_single_transition"
    }
    assert set(macro["source_dataset"]) == {"casp16"}
    assert macro["stage2_exact_site_supervision_eligible"].eq(False).all()
    assert macro["stage2_weak_molecule_supervision_eligible"].eq(True).all()
    assert "non_exact_or_non_numeric_experimental_pka" in set(quarantine["reason"])
    assert "ADMET Predictor PKA" in report["predicted_columns_ignored"]


def test_casp16_macro_rows_have_no_assignment_teacher(tmp_path):
    macro = pd.DataFrame([{
        "molecule_group_key": "glycine",
        "smiles": "NCC(=O)O",
        "experimental_pka_values_json": json.dumps([2.3, 9.6]),
        "casp16_compound_id": "gly",
        "casp16_associated_pdb": "1ABC",
    }])
    rows = casp16._macro_placeholders(macro)
    assert len(rows) == 2
    assert rows["assignment_teacher_used"].eq(False).all()
    assert rows["admet_predictor_values_used"].eq(False).all()
    assert rows["source_measurement_scope"].eq(
        "unordered_molecule_macroscopic_pka_set"
    ).all()


def test_casp16_complex_acquisition_is_reference_only(tmp_path, monkeypatch):
    table = tmp_path / "casp.txt"
    pd.DataFrame([
        {"pdb": "1abc", "PKA1": "4.2", "PKA2": ""},
        {"pdb": "2abc", "PKA1": "nd", "PKA2": ""},
    ]).to_csv(table, sep=";", index=False)

    def fake_acquire(pdb, output):
        return {
            "pdb": pdb, "status": "downloaded_and_verified",
            "path": str(output / f"{pdb}.cif.gz"), "sha256": "abc",
            "bytes": 10, "url": f"https://example.test/{pdb}",
        }

    monkeypatch.setattr(casp16_structures, "_acquire_one", fake_acquire)
    report = casp16_structures.acquire(table, tmp_path / "structures", workers=1)
    assert report["requested_pdb_entries"] == 1
    assert report["structures"][0]["pdb"] == "1ABC"
    assert report["eligible_for_protein_environment_fitting"] is False
    assert "not a bound-state label" in report["label_semantics"]
