"""Synthetic end-to-end tests for the dataset audit command-line workflows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

import audit_functional_group_coverage as coverage_audit
import audit_pair_form_training_provenance as provenance_audit
import audit_raw_dataset_contributions as raw_audit


def _write_sdf(path: Path, records: list[tuple[str, dict[str, str]]]) -> None:
    writer = Chem.SDWriter(str(path))
    try:
        for smiles, properties in records:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            for key, value in properties.items():
                mol.SetProp(key, value)
            writer.write(mol)
    finally:
        writer.close()


def _assignment(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source_file": "source.sdf",
        "record_index": 0,
        "smiles": "CCN",
        "pka_value": 9.5,
        "pka_source_method": "experimental",
        "atom_site_match": False,
        "site_override_applied": False,
        "assignment_status": "single_group",
        "group_mode": "pair_type",
        "pair_member_form": "base_form",
        "pair_assignment_confidence": "high",
        "distribution_ok": True,
        "resolved_groups": "primary_amine",
        "resolved_group_instances": "primary_amine",
    }
    row.update(updates)
    return row


def test_functional_group_scan_inventory_summary_and_gallery(tmp_path: Path) -> None:
    records = [
        {"record_id": "pyridine", "smiles": "c1ccncc1"},
        {"record_id": "phosphate", "smiles": "OP(=O)(O)O"},
        {"record_id": "broken", "smiles": "not-a-smiles"},
    ]
    atoms, molecules = coverage_audit._scan_records(records, "synthetic", 0.5)

    assert set(molecules["parse_status"]) == {"ok", "invalid_smiles"}
    assert {"N", "O", "P"}.issubset(set(atoms["element"]))
    pyridine_n = atoms[(atoms["record_id"] == "pyridine") & (atoms["element"] == "N")]
    assert len(pyridine_n) == 1
    assert bool(pyridine_n.iloc[0]["candidate_covered"])
    assert coverage_audit._environment(Chem.MolFromSmiles("c1ccncc1"), 3)

    summary = coverage_audit._coverage_summary(atoms, molecules)
    assert summary["molecules"] == 2
    assert summary["invalid_smiles"] == 1
    assert summary["ring_nitrogen_atoms"] == 1

    inventory = coverage_audit._pattern_inventory()
    assert {"label", "family", "has_supported_conjugate_pair"}.issubset(inventory.columns)
    assert inventory["has_supported_conjugate_pair"].any()
    assert coverage_audit._pair_supported({"label": "pyridine"})
    assert not coverage_audit._pair_supported({"label": "ketone"})

    gallery_rows = pd.DataFrame([
        {
            "record_id": "pyridine",
            "smiles": "c1ccncc1",
            "atom_index": 3,
            "environment_radius2": "ccncc",
        },
        {
            "record_id": "bad",
            "smiles": "invalid",
            "atom_index": 0,
            "environment_radius2": "N",
        },
    ])
    gallery = tmp_path / "gallery.png"
    assert coverage_audit._render_unmatched_gallery(gallery_rows, gallery) == 1
    assert gallery.exists() and gallery.stat().st_size > 0


def test_functional_group_coverage_cli_writes_complete_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    networks = tmp_path / "networks.csv"
    assignments = tmp_path / "assignments.csv"
    out_dir = tmp_path / "audit"
    pd.DataFrame([
        {"molecule_id": "mol-1", "representative_input_smiles": "c1ccncc1"},
        {"molecule_id": "mol-2", "representative_input_smiles": "CCN"},
    ]).to_csv(networks, index=False)
    pd.DataFrame([
        {
            "source_file": "raw.sdf",
            "record_index": 4,
            "smiles": "Oc1ccccc1",
            "pka_source_method": "experimental",
        },
        {
            "source_file": "ignored.sdf",
            "record_index": 5,
            "smiles": "CCO",
            "pka_source_method": "marvin",
        },
    ]).to_csv(assignments, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_functional_group_coverage.py",
            "--networks", str(networks),
            "--assignments", str(assignments),
            "--out-dir", str(out_dir),
        ],
    )

    coverage_audit.main()

    summary = json.loads((out_dir / "summary.json").read_text())
    assert set(summary["corpora"]) == {
        "canonical_stage3", "raw_experimental_unique_structure"
    }
    assert summary["smarts_inventory"]["labels"] > 0
    assert (out_dir / "index.html").exists()
    assert (out_dir / "all_heteroatom_atom_audit.csv").exists()
    assert "audit_index=" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"pka_source_method": "marvin", "atom_site_match": True}, ("marvin_backed", "marvin_atom_match")),
        ({"pka_source_method": "marvin"}, ("marvin_backed", "marvin_no_atom_match")),
        ({"site_override_applied": True}, ("structure_supported_non_marvin", "manual_site_override")),
        ({"atom_site_match": True}, ("structure_supported_non_marvin", "explicit_atom_match_non_marvin")),
        ({"assignment_status": "single_group"}, ("structure_supported_non_marvin", "single_group_drawn_state")),
        ({"assignment_status": "multi_group"}, ("macro_only_no_site_truth", "multi_group_pair_first")),
    ],
)
def test_pair_provenance_buckets(updates: dict[str, object], expected: tuple[str, str]) -> None:
    assert provenance_audit._provenance_bucket(_assignment(**updates)) == expected


def test_pair_provenance_expansion_summaries_and_report() -> None:
    assignments = pd.DataFrame([
        _assignment(resolved_group_instances="carboxylic_acid|carboxylate"),
        _assignment(
            record_index=1,
            resolved_group_instances="",
            resolved_groups="primary_amine|primary_amine",
            assignment_status="multi_group",
        ),
        _assignment(record_index=2, resolved_group_instances="ketone", group_mode="individual"),
    ])
    pair_slice = provenance_audit._pair_assignment_slice(assignments)
    candidates = provenance_audit._expand_pair_head_rows(assignments)
    assert len(pair_slice) == 2
    assert set(candidates["candidate_label"]) >= {"carboxylic_acid", "carboxylate", "primary_amine"}
    assert len(candidates[candidates["record_index"] == 1]) == 1

    candidates[["provenance_bucket", "provenance_detail"]] = candidates.apply(
        lambda row: pd.Series(provenance_audit._provenance_bucket(row)), axis=1
    )
    bucket = provenance_audit._fraction_summary(candidates, ["provenance_bucket"], "candidate")
    detail = provenance_audit._fraction_summary(
        candidates, ["provenance_bucket", "provenance_detail"], "candidate"
    )
    source = provenance_audit._source_summary(candidates, "candidate")
    family = provenance_audit._family_summary(candidates, "candidate")
    assert bucket["fraction"].sum() == pytest.approx(1.0)
    assert source["rows"].sum() == len(candidates)
    assert family["rows"].sum() == len(candidates)
    assert provenance_audit._format_pct(0.125) == "12.5%"
    assert provenance_audit._top_rows(pd.DataFrame()) == "(empty)"

    report = provenance_audit._build_markdown_report(
        pair_slice.assign(
            provenance_bucket="structure_supported_non_marvin",
            provenance_detail="single_group_drawn_state",
        ),
        bucket,
        detail,
        source,
        candidates,
        bucket,
        detail,
        source,
        family,
    )
    assert "Pair-Form Training Provenance Audit" in report
    assert "Bucket definitions" in report
    assert provenance_audit._fraction_summary(pd.DataFrame(), ["x"], "empty").empty
    assert provenance_audit._source_summary(pd.DataFrame(), "empty").empty
    assert provenance_audit._family_summary(pd.DataFrame(), "empty").empty


def test_pair_provenance_cli_and_no_pair_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assignments = tmp_path / "assignments.csv"
    out_dir = tmp_path / "provenance"
    pd.DataFrame([
        _assignment(pka_source_method="marvin", atom_site_match=True),
        _assignment(
            record_index=1,
            assignment_status="multi_group",
            resolved_group_instances="carboxylic_acid|carboxylate",
        ),
    ]).to_csv(assignments, index=False)
    monkeypatch.setattr(
        provenance_audit,
        "ensure_curated_training_data",
        lambda **_kwargs: (str(assignments), "unused.csv"),
    )
    monkeypatch.setattr(
        sys, "argv", ["audit_pair_form_training_provenance.py", "--out-dir", str(out_dir)]
    )
    provenance_audit.main()
    assert (out_dir / "pair_form_training_provenance.md").exists()
    assert len(list(out_dir.glob("*.csv"))) == 7

    pd.DataFrame([_assignment(group_mode="individual")]).to_csv(assignments, index=False)
    with pytest.raises(RuntimeError, match="No pair-type"):
        provenance_audit.main()


def test_raw_contribution_builder_and_summaries(tmp_path: Path) -> None:
    sdf = tmp_path / "synthetic.sdf"
    _write_sdf(
        sdf,
        [
            ("CCN", {"pKa": "9.5", "pKa_type": "basic"}),
            ("CC(=O)O", {"pKa": "4.7", "pKa_type": "acidic"}),
            ("CCO", {}),
            ("CCO", {"pKa": "99", "pKa_type": "acidic"}),
        ],
    )
    rows = raw_audit.build_rows_for_sdf(str(sdf), 0.75)
    assert len(rows) == 2
    assert rows["molecule_key"].nunique() == 2
    assert set(rows["pka_source_method"]) == {"experimental"}

    overview = raw_audit.summarize_overview(rows)
    groups = raw_audit.summarize_group_contrib(rows)
    families = raw_audit.summarize_family_contrib(rows)
    assert overview.iloc[0]["rows"] == 2
    assert groups["rows"].sum() == 2
    assert families["rows"].sum() == 2
    assert raw_audit._safe_rate(pd.Series(dtype=float)) == 0.0
    assert raw_audit._safe_rate(pd.Series([True, False])) == pytest.approx(0.5)
    assert not raw_audit.summarize_uniques(groups, "final_group_label", "unique").empty

    duplicated = pd.concat([groups, groups.assign(source_file="second.sdf")], ignore_index=True)
    assert raw_audit.summarize_uniques(duplicated, "final_group_label", "unique").empty


def test_raw_contribution_cli_and_input_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_dir = tmp_path / "out"
    _write_sdf(
        raw_dir / "one.sdf",
        [("Oc1ccccc1", {"pKa": "10.0", "ionization_type": "acidic"})],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_raw_dataset_contributions.py", "--raw-dir", str(raw_dir), "--out-dir", str(out_dir)],
    )
    raw_audit.main()
    assert len(list(out_dir.glob("*.csv"))) == 6

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_raw_dataset_contributions.py", "--raw-dir", str(empty_dir), "--out-dir", str(out_dir)],
    )
    with pytest.raises(FileNotFoundError, match="No .sdf"):
        raw_audit.main()

    _write_sdf(empty_dir / "no-pka.sdf", [("CCO", {})])
    with pytest.raises(RuntimeError, match="No pKa-bearing"):
        raw_audit.main()
