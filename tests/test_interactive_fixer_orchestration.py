"""Detailed tests for the stateful manual/automatic SDF fixing workflow."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from rdkit import Chem

from sdf_fixer import interactive_fixer as fixer
from sdf_fixer.group_identifier import identify_groups_from_smiles
from sdf_fixer.protonation_evaluator import ProtonationAssessment, Verdict
from sdf_fixer.sdf_reader import SDFRecord


def _record(index: int, smiles: str, pka: float = 7.0) -> SDFRecord:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return SDFRecord(
        record_index=index,
        mol=mol,
        original_smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(mol),
        pka_value=pka,
        name=f"record-{index}",
        formal_charge=Chem.GetFormalCharge(mol),
    )


def _assessment(rec: SDFRecord, verdict: Verdict, corrected: str | None = None):
    gid = identify_groups_from_smiles(rec.original_smiles)
    assert gid is not None
    return gid, ProtonationAssessment(
        record_index=rec.record_index,
        original_smiles=rec.original_smiles,
        canonical_smiles=rec.canonical_smiles,
        pka_value=rec.pka_value,
        verdict=verdict,
        assigned_site=gid.ionizable_sites[0] if gid.ionizable_sites else None,
        corrected_smiles=corrected,
        explanation="synthetic assessment",
    )


def test_fix_log_color_form_and_image_opener_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = fixer.FixLogEntry(1, None, "CCN", "C[NH3+]", 9.5, "needs", "user", None, None)
    assert entry.as_dict()["name"] == ""
    assert entry.as_dict()["atom_index"] == ""
    assert fixer._form_tag("acid_form") == "protonated/acid-form"
    assert fixer._form_tag("base_form") == "deprotonated/base-form"
    assert fixer._form_tag(None) == "?"
    monkeypatch.setattr(fixer, "_USE_COLOR", True)
    assert fixer._red("x").startswith("\x1b[31m")

    monkeypatch.setattr(fixer.platform, "system", lambda: "Unknown")
    assert fixer._try_open_image("x.png") is False
    monkeypatch.setattr(fixer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(fixer.shutil, "which", lambda cmd: "/bin/opener" if cmd == "xdg-open" else None)
    calls = []
    monkeypatch.setattr(fixer.subprocess, "Popen", lambda args, **kwargs: calls.append((args, kwargs)))
    assert fixer._try_open_image("x.png") is True
    assert calls[0][0] == ["xdg-open", "x.png"]
    monkeypatch.setattr(fixer.subprocess, "Popen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("no gui")))
    assert fixer._try_open_image("x.png") is False


def test_display_helpers_cover_group_ranking_and_saved_or_opened_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rec = _record(0, "NCC(=O)O", 4.0)
    gid = identify_groups_from_smiles(rec.original_smiles)
    assert gid is not None and gid.n_ionizable >= 2
    fixer._print_record_header(rec, gid)
    fixer._print_ionizable_groups(gid, rec.pka_value)
    fixer._show_atom_table(rec.mol, gid)
    output = capsys.readouterr().out
    assert "Plausibility ranking" in output and "Record #0" in output

    benzene = _record(1, "c1ccccc1")
    no_groups = identify_groups_from_smiles(benzene.original_smiles)
    assert no_groups is not None
    fixer._print_ionizable_groups(no_groups, None)
    assert "No ionizable" in capsys.readouterr().out

    image = tmp_path / "mol.png"
    image.write_bytes(b"png")
    monkeypatch.setattr(fixer, "render_for_user", lambda *_args: str(image))
    monkeypatch.setattr(fixer, "_try_open_image", lambda _path: False)
    fixer._show_visualization(rec.mol, gid, rec)
    assert "Image saved" in capsys.readouterr().out
    monkeypatch.setattr(fixer, "_try_open_image", lambda _path: True)
    fixer._show_visualization(rec.mol, gid, rec)
    assert "Image opened" in capsys.readouterr().out


def test_apply_user_edit_supports_every_command(monkeypatch: pytest.MonkeyPatch) -> None:
    mol = Chem.MolFromSmiles("CN")
    assert mol is not None
    assert fixer._apply_user_edit(mol, "")[2] == "invalid"
    assert fixer._apply_user_edit(mol, "skip")[2] == "skip"
    assert fixer._apply_user_edit(mol, "keep")[2] == "kept"
    assert fixer._apply_user_edit(mol, "quit")[2] == "quit"
    assert fixer._apply_user_edit(mol, "p nope")[2] == "invalid"
    assert fixer._apply_user_edit(mol, "d nope")[2] == "invalid"
    assert fixer._apply_user_edit(mol, "mystery")[2] == "invalid"
    monkeypatch.setattr(fixer, "_protonate_at", lambda _mol, idx: f"protonated-{idx}")
    monkeypatch.setattr(fixer, "_deprotonate_at", lambda _mol, idx: f"deprotonated-{idx}")
    assert fixer._apply_user_edit(mol, "protonate 1") == ("protonated-1", 1, "user_protonate")
    assert fixer._apply_user_edit(mol, "deprotonate 1") == ("deprotonated-1", 1, "user_deprotonate")


@pytest.mark.parametrize(
    ("commands", "action", "user_command"),
    [(["s"], "skip", "s"), (["k"], "kept", "k")],
)
def test_ask_user_skip_and_keep(
    commands, action, user_command, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _record(0, "CCN", 9.5)
    gid, assessment = _assessment(rec, Verdict.AMBIGUOUS)
    monkeypatch.setattr(fixer, "_show_atom_table", lambda *_args: None)
    monkeypatch.setattr(fixer, "_show_visualization", lambda *_args: None)
    iterator = iter(commands)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(iterator))
    result = fixer.ask_user(rec, gid, assessment)
    assert result.action == action and result.user_command == user_command


def test_ask_user_retries_invalid_and_failed_then_logs_site(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _record(0, "CCN", 9.5)
    gid, assessment = _assessment(rec, Verdict.AMBIGUOUS)
    monkeypatch.setattr(fixer, "_show_atom_table", lambda *_args: None)
    monkeypatch.setattr(fixer, "_show_visualization", lambda *_args: None)
    edits = iter([
        (None, None, "invalid"),
        (None, 1, "user_protonate"),
        ("C[NH3+]", 2, "user_protonate"),
    ])
    monkeypatch.setattr(fixer, "_apply_user_edit", lambda *_args: next(edits))
    monkeypatch.setattr("builtins.input", lambda _prompt: "command")
    result = fixer.ask_user(rec, gid, assessment)
    assert result.action == "user"
    assert result.fixed_smiles == "C[NH3+]"
    assert result.site_label is not None


def test_ask_user_eof_and_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _record(0, "CCN")
    gid, assessment = _assessment(rec, Verdict.BORDERLINE)
    monkeypatch.setattr(fixer, "_show_atom_table", lambda *_args: None)
    monkeypatch.setattr(fixer, "_show_visualization", lambda *_args: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError()))
    assert fixer.ask_user(rec, gid, assessment).user_command == "EOF"
    monkeypatch.setattr("builtins.input", lambda _prompt: "q")
    with pytest.raises(fixer._QuitSignal):
        fixer.ask_user(rec, gid, assessment)


def _write_sdf(path: Path, smiles: list[str]) -> None:
    writer = Chem.SDWriter(str(path))
    try:
        for index, value in enumerate(smiles):
            mol = Chem.MolFromSmiles(value)
            assert mol is not None
            mol.SetProp("SMILES", value)
            mol.SetProp("tag", str(index))
            writer.write(mol)
    finally:
        writer.close()


def test_fixed_sdf_writer_replaces_valid_and_retains_invalid_edits(tmp_path: Path) -> None:
    source = tmp_path / "source.sdf"
    output = tmp_path / "fixed.sdf"
    _write_sdf(source, ["CCN", "CCO", "c1ccccc1"])
    assert fixer._write_fixed_sdf(
        str(source), str(output), {0: "C[NH3+]", 1: "invalid-smiles"}
    ) == 3
    supplier = Chem.SDMolSupplier(str(output), removeHs=False)
    molecules = [mol for mol in supplier if mol is not None]
    assert len(molecules) == 3
    assert molecules[0].GetProp("SMILES") == "C[NH3+]"
    assert molecules[0].GetProp("SMILES_original") == "CCN"
    assert molecules[1].GetProp("SMILES") == "invalid-smiles"


def test_fix_sdf_orchestrates_all_noninteractive_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        _record(0, "CCN"),
        _record(1, "CCN"),
        _record(2, "c1ccccc1"),
        _record(3, "NCC(=O)O"),
    ]
    assessments = {
        0: _assessment(records[0], Verdict.CORRECT)[1],
        1: _assessment(records[1], Verdict.NEEDS_PROTONATION, "C[NH3+]")[1],
        2: _assessment(records[2], Verdict.NO_IONIZABLE)[1],
        3: _assessment(records[3], Verdict.AMBIGUOUS)[1],
    }
    monkeypatch.setattr(fixer, "read_sdf", lambda _path: records)
    monkeypatch.setattr(fixer, "identify_groups", lambda mol: identify_groups_from_smiles(Chem.MolToSmiles(mol)))
    monkeypatch.setattr(fixer, "evaluate_record", lambda rec, _gid, ph: assessments[rec.record_index])
    written = {}
    monkeypatch.setattr(fixer, "_write_fixed_sdf", lambda _src, _out, fixes: written.update(fixes) or len(records))
    monkeypatch.setattr(fixer, "get_vis_dir", lambda: str(tmp_path / "vis"))
    out = tmp_path / "fixed.sdf"
    log_path = tmp_path / "fixes.csv"
    _, _, entries = fixer.fix_sdf(
        "input.sdf", str(out), str(log_path), auto_only=True
    )
    assert [entry.action for entry in entries] == ["kept", "auto", "kept", "skip"]
    assert written == {1: "C[NH3+]"}
    assert len(list(csv.DictReader(log_path.open()))) == 4


def test_fix_sdf_interactive_edit_and_quit_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [_record(0, "CCN"), _record(1, "CCN")]
    gid, assessment = _assessment(records[0], Verdict.BORDERLINE)
    monkeypatch.setattr(fixer, "read_sdf", lambda _path: records)
    monkeypatch.setattr(fixer, "identify_groups", lambda _mol: gid)
    monkeypatch.setattr(fixer, "evaluate_record", lambda *_args, **_kwargs: assessment)
    calls = 0

    def ask(*_args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise fixer._QuitSignal()
        return fixer.FixLogEntry(0, None, "CCN", "C[NH3+]", 7.0, "borderline", "user", "primary_amine", 1)

    monkeypatch.setattr(fixer, "ask_user", ask)
    captured = {}
    monkeypatch.setattr(fixer, "_write_fixed_sdf", lambda _src, _out, fixes: captured.update(fixes) or 2)
    monkeypatch.setattr(fixer, "get_vis_dir", lambda: str(tmp_path))
    out, log, entries = fixer.fix_sdf(
        str(tmp_path / "input.sdf"), max_records=2
    )
    assert Path(out).name == "FIXED_input.sdf"
    assert Path(log).name == "FIXLOG_input.csv"
    assert len(entries) == 1 and captured == {0: "C[NH3+]"}


def test_write_log_ignores_empty_and_writes_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "log.csv"
    fixer._write_log([], str(path))
    assert not path.exists()
    fixer._write_log([
        fixer.FixLogEntry(1, "x", "CCN", "CCN", 9.0, "correct", "kept", "amine", 1)
    ], str(path))
    assert list(csv.DictReader(path.open()))[0]["site_label"] == "amine"
