"""Tests for documentation figures, assignment galleries, and SDF CLIs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from rdkit import Chem

import render_pipeline_overview as pipeline_figure
import render_prediction_example as prediction_figure
import visualize_complex_group_assignments as complex_viz
import visualize_single_group_classes as class_viz
from sdf_fixer import cli as fixer_cli
from sdf_fixer import fix as fixer_entry
from sdf_fixer.group_identifier import identify_groups_from_smiles
from sdf_fixer.protonation_evaluator import ProtonationAssessment, Verdict
from sdf_fixer.sdf_reader import SDFRecord


def test_pipeline_overview_build_render_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert pipeline_figure._wrapped("alpha beta gamma", 8).count("\n") >= 1
    figure = pipeline_figure.build_figure()
    assert len(figure.axes) == 1
    plt.close(figure)
    paths = pipeline_figure.render(str(tmp_path), "pipeline", ["svg", "png"], 72)
    assert {path.suffix for path in paths} == {".svg", ".png"}
    assert all(path.stat().st_size > 0 for path in paths)
    assert "Canonical pKate pipeline overview" in (tmp_path / "pipeline.svg").read_text()
    with pytest.raises(ValueError, match="Unsupported format"):
        pipeline_figure.render(str(tmp_path), "bad", ["pdf"], 72)
    monkeypatch.setattr(sys, "argv", ["render_pipeline_overview.py", "--output-dir", str(tmp_path), "--basename", "cli", "--formats", "svg", "--dpi", "60"])
    pipeline_figure.main()
    assert str(tmp_path / "cli.svg") in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["render_pipeline_overview.py", "--formats", " , "])
    with pytest.raises(ValueError, match="At least one"):
        pipeline_figure.main()


def test_prediction_example_molecule_build_render_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = prediction_figure._molecule_image("NCCCC(=O)O", site_atoms=[0, 6], size=(300, 180))
    assert image.size == (300, 180)
    with pytest.raises(ValueError, match="Could not parse"):
        prediction_figure._molecule_image("bad-smiles", site_atoms=[0, 1])
    figure = prediction_figure.build_figure()
    assert len(figure.axes) >= 2
    plt.close(figure)
    paths = prediction_figure.render(str(tmp_path), "prediction", ["svg"], 72)
    assert paths == [tmp_path / "prediction.svg"]
    assert paths[0].stat().st_size > 0
    with pytest.raises(ValueError, match="Unsupported format"):
        prediction_figure.render(str(tmp_path), "bad", ["jpg"], 72)
    monkeypatch.setattr(sys, "argv", ["render_prediction_example.py", "--output-dir", str(tmp_path), "--basename", "cli", "--formats", "png", "--dpi", "60"])
    prediction_figure.main()
    assert (tmp_path / "cli.png").exists()
    monkeypatch.setattr(sys, "argv", ["render_prediction_example.py", "--formats", ""])
    with pytest.raises(ValueError, match="At least one"):
        prediction_figure.main()


def test_complex_assignment_helpers_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assignments = tmp_path / "assignments.csv"
    pd.DataFrame([
        {"smiles": "NCC(=O)O", "assignment_status": "multi_group", "resolved_group_count": 2, "rejected_group_count": 1, "resolved_groups": "primary_amine|carboxylic_acid", "source_file": "one.sdf"},
        {"smiles": "CCN", "assignment_status": "single_group", "resolved_group_count": 1, "rejected_group_count": 0, "resolved_groups": "primary_amine", "source_file": "two.sdf"},
    ]).to_csv(assignments, index=False)
    selected = complex_viz._select_complex_smiles(str(assignments), top_n=2)
    assert selected["smiles"].tolist() == ["NCC(=O)O"]
    mol, sites = complex_viz._collect_site_annotations("NCC(=O)O")
    assert mol.GetNumAtoms() == 5 and len(sites) >= 2
    with pytest.raises(ValueError, match="Could not parse"):
        complex_viz._collect_site_annotations("bad")
    atoms, colors, legend = complex_viz._build_highlight_data(sites)
    assert atoms and colors and ":" in legend
    out_dir = tmp_path / "complex"
    monkeypatch.setattr(sys, "argv", ["visualize_complex_group_assignments.py", "--assignments", str(assignments), "--out-dir", str(out_dir), "--top-n", "1"])
    complex_viz.main()
    assert (out_dir / "complex_group_examples.csv").exists()
    assert (out_dir / "complex_group_examples.png").exists()


def test_single_class_helpers_render_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mol, sites = class_viz._collect_sites("Oc1ccccc1")
    assert mol is not None and sites
    assert class_viz._collect_sites("bad") == (None, [])
    chosen = class_viz._pick_site(sites, "phenol", None)
    assert chosen is not None and chosen["label"] == "phenol"
    assert class_viz._pick_site([], "phenol", None) is None
    assert class_viz._pick_site(sites, "missing", "missing") is not None
    assert class_viz._sanitize_label("amine / acid") == "amine___acid"
    frame = pd.DataFrame([
        {"smiles": "Oc1ccccc1", "pka_value": 10.0, "final_group_label": "phenol", "final_group_refined": "phenol", "resolved_groups": "phenol", "source_file": "raw.sdf"},
        {"smiles": "bad", "pka_value": 9.0, "final_group_label": "phenol", "final_group_refined": "phenol", "resolved_groups": "phenol", "source_file": "raw.sdf"},
    ])
    csv = tmp_path / "classes.csv"
    frame.to_csv(csv, index=False)
    out_dir = tmp_path / "classes"
    monkeypatch.setattr(sys, "argv", ["visualize_single_group_classes.py", "--in-csv", str(csv), "--out-dir", str(out_dir), "--classes", "phenol,missing", "--max-examples", "2"])
    class_viz.main()
    index = pd.read_csv(out_dir / "class_visualization_index.csv")
    assert index["class_name"].tolist() == ["phenol"]
    assert (out_dir / "phenol_examples.png").exists()
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"smiles": ["CCO"]}).to_csv(bad_csv, index=False)
    monkeypatch.setattr(sys, "argv", ["visualize_single_group_classes.py", "--in-csv", str(bad_csv), "--out-dir", str(out_dir)])
    with pytest.raises(RuntimeError, match="final_group_label"):
        class_viz.main()


def _record(smiles: str = "CCN") -> SDFRecord:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return SDFRecord(record_index=2, mol=mol, original_smiles=smiles, canonical_smiles=Chem.MolToSmiles(mol), pka_value=9.5, temperature=298.0, name="example", formal_charge=Chem.GetFormalCharge(mol))


def test_sdf_cli_formatting_print_and_main(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert fixer_cli._form_tag("acid_form") == "protonated/acid-form"
    assert fixer_cli._form_tag("base_form") == "deprotonated/base-form"
    assert fixer_cli._form_tag(None) == "?"
    for verdict in Verdict:
        assert fixer_cli._verdict_banner(verdict)
    monkeypatch.setattr(fixer_cli, "_USE_COLOR", True)
    assert fixer_cli._green("ok").startswith("\x1b[32m")
    rec = _record()
    gid = identify_groups_from_smiles(rec.original_smiles)
    assert gid is not None and gid.ionizable_sites
    assessment = ProtonationAssessment(record_index=2, original_smiles=rec.original_smiles, canonical_smiles=rec.canonical_smiles, pka_value=9.5, verdict=Verdict.NEEDS_PROTONATION, assigned_site=gid.ionizable_sites[0], current_form="base_form", expected_form="acid_form", corrected_smiles="C[NH3+]", explanation="synthetic correction")
    fixer_cli.print_evaluation(rec, gid, assessment, verbose=True)
    assert "CORRECTED SMILES" in capsys.readouterr().out
    monkeypatch.setattr(fixer_cli, "read_sdf", lambda _path: [rec])
    monkeypatch.setattr(fixer_cli, "identify_groups", lambda *_args, **_kwargs: gid)
    monkeypatch.setattr(fixer_cli, "evaluate_record", lambda *_args, **_kwargs: assessment)
    monkeypatch.setattr(sys, "argv", ["sdf-fixer", "input.sdf", "--only-fixes", "--max-records", "1", "--ph", "7.0"])
    fixer_cli.main()
    output = capsys.readouterr().out
    assert "Need amendment" in output and "Auto-corrected SMILES" in output


def test_sdf_fix_entrypoint_forwards_all_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}
    monkeypatch.setattr(fixer_entry, "fix_sdf", lambda **kwargs: received.update(kwargs))
    monkeypatch.setattr(sys, "argv", ["sdf-fix", "input.sdf", "--out", str(tmp_path / "out.sdf"), "--log", str(tmp_path / "log.csv"), "--ph", "6.5", "--max-records", "3", "--auto-only"])
    fixer_entry.main()
    assert received == {"sdf_path": "input.sdf", "out_path": str(tmp_path / "out.sdf"), "log_path": str(tmp_path / "log.csv"), "ph": 6.5, "max_records": 3, "auto_only": True}
