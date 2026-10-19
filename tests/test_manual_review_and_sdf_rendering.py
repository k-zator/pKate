"""Data-independent tests for curator review, SDF I/O, and molecule rendering."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

import site_resolution_review as review
from functional_group_pka_analysis import SITE_OVERRIDE_KEY_COLUMNS
from sdf_fixer import interactive_fixer, mol_visualizer
from sdf_fixer.group_identifier import identify_groups
from sdf_fixer.protonation_evaluator import ProtonationAssessment, Verdict
from sdf_fixer.sdf_reader import SDFRecord, _safe_float, iter_sdf, read_sdf


def _write_example_sdf(path: Path) -> Path:
    records = [
        (
            "[NH3+]CC(=O)[O-]",
            {
                "SMILES": "[NH3+]CC(=O)[O-]",
                "pKa": "9.60",
                "temp": "298.15",
                "name": "glycine zwitterion",
                "ionization_type": "base",
                "type": "basic",
                "custom_property": "preserve me",
            },
        ),
        (
            "CCO",
            {
                "SMILES": "CCO",
                "pKa": "not measured",
                "temp": "nan",
                "name": "ethanol",
                "custom_property": "second record",
            },
        ),
    ]
    writer = Chem.SDWriter(str(path))
    for smiles, properties in records:
        molecule = Chem.MolFromSmiles(smiles)
        assert molecule is not None
        for key, value in properties.items():
            molecule.SetProp(key, value)
        writer.write(molecule)
    writer.close()
    return path


def _review_row(record_index: int = 0, **updates) -> dict:
    row = {
        "source_file": "review.sdf",
        "record_index": record_index,
        "smiles": "NCC(=O)O",
        "pka_value": 9.6 + record_index,
        "pka_source_method": "experimental",
        "pka_type_canonical": "basic",
        "atom_index_raw": 0,
        "assignment_status": "multi_group",
        "atom_candidate_count": 2,
        "atom_site_match": True,
        "resolved_group_count": 2,
        "pair_assignment_confidence": "ambiguous",
        "resolved_groups": "primary_amine|carboxylic_acid",
        "selected_group_label": "primary_amine",
        "auto_selected_group_label": "primary_amine",
        "atom_candidate_labels": "primary_amine",
        "site_override_applied": False,
        "review_reason": "multi_group",
    }
    row.update(updates)
    return row


def _record(smiles: str, index: int = 0, pka: float = 9.6) -> SDFRecord:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return SDFRecord(
        record_index=index,
        mol=molecule,
        original_smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(molecule),
        pka_value=pka,
        name=f"record {index}",
        formal_charge=Chem.GetFormalCharge(molecule),
    )


def test_sdf_reader_preserves_properties_charges_and_record_identity(tmp_path):
    sdf_path = _write_example_sdf(tmp_path / "example.sdf")

    records = read_sdf(str(sdf_path))

    assert [record.record_index for record in records] == [0, 1]
    assert records[0].original_smiles == "[NH3+]CC(=O)[O-]"
    assert records[0].pka_value == pytest.approx(9.6)
    assert records[0].temperature == pytest.approx(298.15)
    assert records[0].name == "glycine zwitterion"
    assert records[0].formal_charge == 0
    assert sorted(records[0].atom_charges.values()) == [-1, 1]
    assert records[0].raw_properties["custom_property"] == "preserve me"
    assert records[1].pka_value is None
    assert records[1].temperature is None


def test_lazy_and_eager_sdf_readers_have_the_same_semantics(tmp_path):
    sdf_path = _write_example_sdf(tmp_path / "example.sdf")
    eager = read_sdf(str(sdf_path))
    lazy = list(iter_sdf(str(sdf_path)))

    assert [record.record_index for record in lazy] == [0, 1]
    assert [record.canonical_smiles for record in lazy] == [
        record.canonical_smiles for record in eager
    ]
    assert [record.raw_properties for record in lazy] == [
        record.raw_properties for record in eager
    ]


@pytest.mark.parametrize("value", [None, "", "bad", "nan", "inf", "-inf"])
def test_sdf_numeric_parser_rejects_missing_malformed_and_nonfinite_values(value):
    assert _safe_float(value) is None


def test_sdf_reader_honours_explicit_hydrogen_policy(tmp_path):
    path = tmp_path / "explicit-h.sdf"
    molecule = Chem.AddHs(Chem.MolFromSmiles("CO"))
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()

    preserved = read_sdf(str(path), removeHs=False)[0]
    stripped = read_sdf(str(path), removeHs=True)[0]
    assert preserved.mol.GetNumAtoms() > stripped.mol.GetNumAtoms()


def test_molecule_png_renderer_is_headless_valid_and_non_mutating(tmp_path):
    molecule = Chem.MolFromSmiles("NCC(=O)O")
    assert molecule is not None
    groups = identify_groups(molecule)
    output = tmp_path / "nested" / "review.png"

    rendered = mol_visualizer.render_mol_png(
        molecule,
        groups,
        str(output),
        pka_value=9.6,
        record_index=7,
        width=640,
        height=360,
    )

    payload = output.read_bytes()
    assert rendered == str(output)
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert payload[12:16] == b"IHDR"
    assert struct.unpack(">II", payload[16:24]) == (640, 360)
    assert all(not atom.HasProp("molAtomMapNumber") for atom in molecule.GetAtoms())


def test_renderer_legend_and_atom_table_identify_sites():
    molecule = Chem.MolFromSmiles("NCC(=O)O")
    assert molecule is not None
    groups = identify_groups(molecule)

    legend = mol_visualizer._build_legend_text(groups, 9.6, 4)
    table = mol_visualizer.atom_table_str(molecule, groups)

    assert "Rec #4" in legend
    assert "9.6" in legend
    assert "charge=" in legend
    assert "primary_amine" in legend
    assert "Idx" in table
    assert "primary_amine" in table


def test_render_for_user_reuses_session_directory(tmp_path, monkeypatch):
    molecule = Chem.MolFromSmiles("CC(=O)O")
    assert molecule is not None
    groups = identify_groups(molecule)
    monkeypatch.setattr(mol_visualizer, "_VIS_DIR", str(tmp_path))

    first = mol_visualizer.render_for_user(molecule, groups, 3, 4.7)
    second = mol_visualizer.render_for_user(molecule, groups, 4, 4.7)

    assert Path(first).name == "record_00003.png"
    assert Path(second).name == "record_00004.png"
    assert Path(first).is_file() and Path(second).is_file()


def test_review_reason_generation_tolerates_blank_numeric_fields():
    row = pd.Series(
        _review_row(
            atom_candidate_count=np.nan,
            resolved_group_count=np.nan,
            atom_index_raw=np.nan,
            assignment_status="single_group",
            pair_assignment_confidence="",
            resolved_groups="primary_amine",
        )
    )
    assert review._reason_tags(row) == []


def test_review_reason_generation_handles_csv_booleans_and_all_reason_types():
    row = pd.Series(
        _review_row(
            atom_site_match="False",
            resolved_groups="primary_amine|carboxylic_acid|primary_amine",
        )
    )
    assert review._reason_tags(row) == [
        "multi_group",
        "multiple_atom_matches",
        "atom_index_unmatched",
        "pair_confidence_ambiguous",
        "multiple_resolved_labels",
    ]


def test_review_queue_filters_only_truthy_review_flags(monkeypatch):
    frame = pd.DataFrame(
        [
            _review_row(0, site_override_applied="False"),
            _review_row(1, site_override_applied="True"),
        ]
    )
    monkeypatch.setattr(review, "build_functional_group_table", lambda **_: frame)

    queue = review.build_site_resolution_review_queue(include_reviewed=False)
    all_rows = review.build_site_resolution_review_queue(include_reviewed=True)

    assert queue["record_index"].tolist() == [0]
    assert all_rows["record_index"].tolist() == [0, 1]


def test_raw_molecule_cache_preserves_indices_and_returns_copies(tmp_path):
    _write_example_sdf(tmp_path / "example.sdf")
    cache = review.RawMolCache(str(tmp_path))

    first = cache.get("example.sdf", 0)
    first.GetAtomWithIdx(0).SetFormalCharge(0)
    fresh_copy = cache.get("example.sdf", 0)

    assert fresh_copy.GetAtomWithIdx(0).GetFormalCharge() == 1
    with pytest.raises(IndexError, match="out of range"):
        cache.get("example.sdf", 2)
    with pytest.raises(IndexError, match="out of range"):
        cache.get("example.sdf", -1)


def test_review_png_marks_real_candidates(tmp_path):
    molecule = Chem.MolFromSmiles("NCC(=O)O")
    assert molecule is not None
    row = pd.Series(_review_row())
    candidates = review._ordered_candidates(molecule, row, overlap_threshold=0.75)
    output = tmp_path / "site-review.png"

    review._render_review_png(molecule, row, candidates, str(output))

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert candidates
    assert any(candidate["atom_index_hit"] for candidate in candidates)
    assert any(candidate["is_current"] for candidate in candidates)


def test_override_upsert_is_key_exact_and_save_accepts_bare_filename(
    tmp_path, monkeypatch
):
    first = pd.Series(_review_row(0))
    second = pd.Series(_review_row(1))
    overrides = pd.DataFrame(columns=review._override_columns())
    overrides = review._upsert_override(overrides, first, "primary_amine", "first", 0)
    overrides = review._upsert_override(overrides, second, "carboxylic_acid", "second", 2)
    overrides = review._upsert_override(overrides, first, "secondary_amine", "updated", 0)

    assert len(overrides) == 2
    selected = overrides.loc[overrides["record_index"].astype(int) == 0].iloc[0]
    assert selected["override_group_label"] == "secondary_amine"
    assert selected["note"] == "updated"
    assert str(selected["reviewed_at"]).endswith("+00:00")

    monkeypatch.chdir(tmp_path)
    review._save_overrides("overrides.csv", overrides)
    restored = review._load_overrides("overrides.csv")
    assert len(restored) == 2
    assert set(review._override_columns()) == set(restored.columns)


def test_loading_legacy_override_table_adds_missing_columns(tmp_path):
    path = tmp_path / "legacy.csv"
    pd.DataFrame([{"source_file": "old.sdf", "record_index": 2}]).to_csv(
        path, index=False
    )

    restored = review._load_overrides(str(path))

    assert list(restored.columns) == review._override_columns()
    assert restored.iloc[0]["source_file"] == "old.sdf"
    assert restored.iloc[0]["override_group_label"] == ""


def test_review_decision_is_bound_to_the_currently_displayed_structure(
    tmp_path, monkeypatch
):
    queue = pd.DataFrame([_review_row(0), _review_row(1)])
    molecule = Chem.MolFromSmiles("NCC(=O)O")
    assert molecule is not None

    monkeypatch.setattr(
        review, "build_site_resolution_review_queue", lambda **_: queue
    )
    monkeypatch.setattr(review.RawMolCache, "get", lambda *_: Chem.Mol(molecule))
    monkeypatch.setattr(
        review,
        "_ordered_candidates",
        lambda _mol, row, **_: [
            {
                "candidate_index": 0,
                "label": f"selected_for_record_{int(row['record_index'])}",
                "family": "test",
                "site_type": "basic",
                "atom_set": {int(row["record_index"])},
                "atom_text": str(int(row["record_index"])),
                "atom_index_hit": True,
                "is_auto": False,
                "is_current": False,
            }
        ],
    )
    monkeypatch.setattr(review, "_render_review_png", lambda *args: args[-1])
    commands = iter(["s", "0"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))

    overrides_path = tmp_path / "overrides.csv"
    _, _, reviewed = review.review_site_resolution(
        raw_dir=str(tmp_path),
        overrides_path=str(overrides_path),
        queue_csv=str(tmp_path / "queue.csv"),
        image_dir=str(tmp_path / "images"),
        open_images=False,
    )

    saved = pd.read_csv(overrides_path)
    assert reviewed == 2
    assert saved["record_index"].astype(int).tolist() == [1]
    assert saved.iloc[0]["override_group_label"] == "selected_for_record_1"
    assert int(saved.iloc[0]["override_atom_index_raw"]) == 1


def test_manual_review_quit_preserves_decisions_already_saved(tmp_path, monkeypatch):
    queue = pd.DataFrame([_review_row(0), _review_row(1)])
    molecule = Chem.MolFromSmiles("NCC(=O)O")
    assert molecule is not None
    monkeypatch.setattr(
        review, "build_site_resolution_review_queue", lambda **_: queue
    )
    monkeypatch.setattr(review.RawMolCache, "get", lambda *_: Chem.Mol(molecule))
    monkeypatch.setattr(review, "_ordered_candidates", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(review, "_render_review_png", lambda *args: args[-1])
    commands = iter(["m reviewer_selected_group", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))
    overrides_path = tmp_path / "overrides.csv"

    review.review_site_resolution(
        raw_dir=str(tmp_path),
        overrides_path=str(overrides_path),
        queue_csv=str(tmp_path / "queue.csv"),
        image_dir=str(tmp_path / "images"),
        open_images=False,
    )

    saved = pd.read_csv(overrides_path, keep_default_na=False)
    assert saved["record_index"].astype(int).tolist() == [0]
    assert saved.iloc[0]["override_group_label"] == "reviewer_selected_group"
    assert saved.iloc[0]["note"] == "manual"


def test_keep_removes_only_the_displayed_rows_prior_override(tmp_path, monkeypatch):
    row_zero = pd.Series(_review_row(0))
    row_one = pd.Series(_review_row(1))
    existing = pd.DataFrame(columns=review._override_columns())
    existing = review._upsert_override(existing, row_zero, "old_zero")
    existing = review._upsert_override(existing, row_one, "keep_one")
    overrides_path = tmp_path / "overrides.csv"
    review._save_overrides(str(overrides_path), existing)

    molecule = Chem.MolFromSmiles("NCC(=O)O")
    assert molecule is not None
    monkeypatch.setattr(
        review,
        "build_site_resolution_review_queue",
        lambda **_: pd.DataFrame([_review_row(0)]),
    )
    monkeypatch.setattr(review.RawMolCache, "get", lambda *_: Chem.Mol(molecule))
    monkeypatch.setattr(review, "_ordered_candidates", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(review, "_render_review_png", lambda *args: args[-1])
    monkeypatch.setattr("builtins.input", lambda _prompt: "k")

    review.review_site_resolution(
        raw_dir=str(tmp_path),
        overrides_path=str(overrides_path),
        queue_csv=str(tmp_path / "queue.csv"),
        image_dir=str(tmp_path / "images"),
        open_images=False,
    )

    saved = pd.read_csv(overrides_path)
    assert saved["record_index"].astype(int).tolist() == [1]
    assert saved.iloc[0]["override_group_label"] == "keep_one"


def test_export_only_honours_record_cap_and_bare_queue_path(tmp_path, monkeypatch):
    queue = pd.DataFrame([_review_row(0), _review_row(1)])
    monkeypatch.setattr(
        review, "build_site_resolution_review_queue", lambda **_: queue
    )
    monkeypatch.chdir(tmp_path)

    _, queue_path, count = review.review_site_resolution(
        overrides_path="overrides.csv",
        queue_csv="queue.csv",
        max_records=1,
        export_only=True,
        open_images=False,
    )

    assert queue_path == "queue.csv"
    assert count == 1
    assert pd.read_csv("queue.csv")["record_index"].astype(int).tolist() == [0]


def test_fixed_sdf_writer_updates_only_selected_record_and_preserves_properties(
    tmp_path,
):
    source = _write_example_sdf(tmp_path / "source.sdf")
    output = tmp_path / "fixed.sdf"

    count = interactive_fixer._write_fixed_sdf(
        str(source), str(output), {1: "CC[O-]"}
    )
    records = read_sdf(str(output))

    assert count == 2
    assert records[0].original_smiles == "[NH3+]CC(=O)[O-]"
    assert records[0].raw_properties["custom_property"] == "preserve me"
    assert records[1].original_smiles == "CC[O-]"
    assert records[1].raw_properties["SMILES_original"] == "CCO"
    assert records[1].raw_properties["custom_property"] == "second record"


def test_manual_fixer_keep_command_logs_the_displayed_record(monkeypatch):
    record = _record("CN", index=23)
    groups = identify_groups(record.mol)
    assessment = ProtonationAssessment(
        record_index=23,
        original_smiles="CN",
        canonical_smiles="CN",
        pka_value=9.6,
        verdict=Verdict.AMBIGUOUS,
        explanation="test review",
    )
    for name in (
        "_print_record_header",
        "_print_ionizable_groups",
        "_show_atom_table",
        "_show_visualization",
    ):
        monkeypatch.setattr(interactive_fixer, name, lambda *args: None)
    commands = iter(["not a command", "k"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(commands))

    entry = interactive_fixer.ask_user(record, groups, assessment)

    assert entry.record_index == 23
    assert entry.name == "record 23"
    assert entry.action == "kept"
    assert entry.user_command == "k"
    assert entry.fixed_smiles == Chem.MolToSmiles(record.mol)


@pytest.mark.parametrize(
    ("command", "action", "atom_index"),
    [
        ("", "invalid", None),
        ("s", "skip", None),
        ("q", "quit", None),
        ("p not-an-index", "invalid", None),
        ("d not-an-index", "invalid", None),
        ("unknown", "invalid", None),
    ],
)
def test_manual_edit_command_parser_has_stable_control_actions(
    command, action, atom_index
):
    molecule = Chem.MolFromSmiles("CN")
    assert molecule is not None
    _, parsed_index, parsed_action = interactive_fixer._apply_user_edit(
        molecule, command
    )
    assert parsed_action == action
    assert parsed_index == atom_index


def test_manual_fixer_eof_is_a_non_destructive_skip(monkeypatch):
    record = _record("CN", index=12)
    groups = identify_groups(record.mol)
    assessment = ProtonationAssessment(
        record_index=12,
        original_smiles="CN",
        canonical_smiles="CN",
        pka_value=9.6,
        verdict=Verdict.AMBIGUOUS,
    )
    for name in (
        "_print_record_header",
        "_print_ionizable_groups",
        "_show_atom_table",
        "_show_visualization",
    ):
        monkeypatch.setattr(interactive_fixer, name, lambda *args: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError))

    entry = interactive_fixer.ask_user(record, groups, assessment)

    assert entry.record_index == 12
    assert entry.action == "skip"
    assert entry.user_command == "EOF"
    assert entry.fixed_smiles == record.canonical_smiles


def test_auto_only_fixer_covers_correct_fixed_no_site_and_skipped_paths(
    tmp_path, monkeypatch
):
    records = [_record("CN", index=index) for index in range(4)]
    assessments = {
        0: ProtonationAssessment(0, "CN", "CN", 9.6, Verdict.CORRECT),
        1: ProtonationAssessment(
            1,
            "CN",
            "CN",
            9.6,
            Verdict.NEEDS_PROTONATION,
            corrected_smiles="C[NH3+]",
        ),
        2: ProtonationAssessment(2, "CN", "CN", 9.6, Verdict.NO_IONIZABLE),
        3: ProtonationAssessment(3, "CN", "CN", 9.6, Verdict.AMBIGUOUS),
    }
    captured = {}
    monkeypatch.setattr(interactive_fixer, "read_sdf", lambda _path: records)
    monkeypatch.setattr(interactive_fixer, "identify_groups", lambda mol: object())
    monkeypatch.setattr(
        interactive_fixer,
        "evaluate_record",
        lambda record, _groups, ph: assessments[record.record_index],
    )
    monkeypatch.setattr(interactive_fixer, "get_vis_dir", lambda: str(tmp_path))

    def capture_fixes(_source, _output, fixes):
        captured["fixes"] = dict(fixes)
        return 4

    monkeypatch.setattr(
        interactive_fixer,
        "_write_fixed_sdf",
        capture_fixes,
    )
    monkeypatch.setattr(
        interactive_fixer,
        "_write_log",
        lambda entries, _path: captured.setdefault("entries", list(entries)),
    )

    _, _, entries = interactive_fixer.fix_sdf(
        "synthetic.sdf",
        out_path=str(tmp_path / "fixed.sdf"),
        log_path=str(tmp_path / "log.csv"),
        auto_only=True,
    )

    assert [entry.action for entry in entries] == ["kept", "auto", "kept", "skip"]
    assert captured["fixes"] == {1: "C[NH3+]"}
    assert captured["entries"] == entries


def test_auto_only_fixer_writes_every_input_record_and_audit_log(tmp_path):
    source = _write_example_sdf(tmp_path / "source.sdf")
    output = tmp_path / "fixed.sdf"
    log_path = tmp_path / "fixes.csv"

    returned_output, returned_log, entries = interactive_fixer.fix_sdf(
        str(source),
        out_path=str(output),
        log_path=str(log_path),
        auto_only=True,
    )

    assert returned_output == str(output)
    assert returned_log == str(log_path)
    assert len(read_sdf(str(output))) == 2
    assert len(entries) == 2
    assert {entry.record_index for entry in entries} == {0, 1}
    assert log_path.is_file()
    logged = pd.read_csv(log_path)
    assert logged["record_index"].astype(int).tolist() == [0, 1]
