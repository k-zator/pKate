"""Provenance failures and CLI forwarding for the primary dataset builders."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import build_chembl_pka_truth_dataset as chembl
import build_molecule_microstate_network_dataset as network_builder
import build_release_microstate_dataset as release_builder
import functional_group_pka_analysis as group_analysis
import ml_dataset_builder
import training_data_resolver as resolver


def test_resolver_dependency_inventory_and_assignment_provenance(tmp_path: Path) -> None:
    override = tmp_path / "overrides.csv"
    override.write_text("key\n", encoding="utf-8")
    dependencies = resolver._dependency_paths(str(override))
    assert str(override) in dependencies
    assert any(path.endswith("functional_group_pka_analysis.py") for path in dependencies)

    missing = tmp_path / "missing.csv"
    empty = tmp_path / "empty.csv"
    empty.write_text("pka_source_method\n", encoding="utf-8")
    wrong_header = tmp_path / "wrong.csv"
    wrong_header.write_text("method\nexperimental\n", encoding="utf-8")
    mixed = tmp_path / "mixed.csv"
    mixed.write_text("pka_source_method\nexperimental\nmarvin\n", encoding="utf-8")
    safe = tmp_path / "safe.csv"
    safe.write_text("pka_source_method\n Experimental \n", encoding="utf-8")
    assert not resolver._assignments_are_experimental_only(str(missing))
    assert not resolver._assignments_are_experimental_only(str(empty))
    assert not resolver._assignments_are_experimental_only(str(wrong_header))
    assert not resolver._assignments_are_experimental_only(str(mixed))
    assert resolver._assignments_are_experimental_only(str(safe))

    no_parent = tmp_path / "plain.csv"
    resolver._ensure_parent_dir(str(no_parent))
    nested = tmp_path / "a" / "b" / "file.csv"
    resolver._ensure_parent_dir(str(nested))
    assert nested.parent.is_dir()


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"single_group_strict": [True]}), "pka_source_method is missing"),
        (
            pd.DataFrame({
                "pka_source_method": ["marvin"],
                "single_group_strict": [True],
                "final_group_summary_label": ["amine"],
                "distribution_ok": [True],
            }),
            "No experimental pKa",
        ),
    ],
)
def test_resolver_refuses_uncertifiable_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame, message: str
) -> None:
    monkeypatch.setattr(resolver, "_trusted_raw_paths", lambda _raw: ["raw.sdf"])
    monkeypatch.setattr(resolver, "_dependency_paths", lambda **_kwargs: [])
    monkeypatch.setattr(group_analysis, "build_functional_group_table", lambda **_kwargs: frame.copy())
    with pytest.raises(ValueError, match=message):
        resolver.ensure_curated_training_data(
            assignments_path=str(tmp_path / "out" / "assignments.csv"),
            summary_path=str(tmp_path / "out" / "summary.csv"),
            raw_dir=str(tmp_path / "raw"),
        )


def test_resolver_preserves_existing_training_label_during_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = pd.DataFrame({
        "pka_source_method": ["experimental"],
        "single_group_strict": [True],
        "final_group_summary_label": ["primary_amine"],
        "training_group_label": ["amine_custom"],
        "distribution_ok": [True],
        "pka_value": [9.5],
    })
    captured = {}
    monkeypatch.setattr(resolver, "_trusted_raw_paths", lambda _raw: ["raw.sdf"])
    monkeypatch.setattr(resolver, "_dependency_paths", lambda **_kwargs: [])
    monkeypatch.setattr(group_analysis, "build_functional_group_table", lambda **_kwargs: source.copy())
    monkeypatch.setattr(
        group_analysis,
        "build_summary_table",
        lambda frame: captured.setdefault("frame", frame.copy()) or pd.DataFrame(),
    )
    # Avoid DataFrame truth-value evaluation in the capture lambda.
    def summary(frame):
        captured["frame"] = frame.copy()
        return pd.DataFrame({"final_group_label": ["amine_custom"], "count": [1]})
    monkeypatch.setattr(group_analysis, "build_summary_table", summary)
    assignments = tmp_path / "out" / "assignments.csv"
    summary_path = tmp_path / "out" / "summary.csv"
    resolver.ensure_curated_training_data(
        assignments_path=str(assignments),
        summary_path=str(summary_path),
        raw_dir=str(tmp_path),
    )
    assert captured["frame"].iloc[0]["final_group_label"] == "amine_custom"


def test_release_builder_main_forwards_review_and_discovery_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    received = {}
    frames = (
        pd.DataFrame({"x": [1, 2]}),
        pd.DataFrame({"x": [1]}),
        pd.DataFrame({"x": [1]}),
        pd.DataFrame({"x": [1, 2, 3]}),
    )
    monkeypatch.setattr(release_builder, "build_dataset", lambda **kwargs: received.update(kwargs) or frames)
    monkeypatch.setattr(sys, "argv", [
        "build_release_microstate_dataset.py",
        "--raw-dir", "raw",
        "--assignments", "assignments.csv",
        "--override-parts", "a.csv", "b.csv",
        "--output", str(tmp_path / "release.csv"),
        "--quarantine", str(tmp_path / "quarantine.csv"),
        "--clean-overrides", str(tmp_path / "clean.csv"),
        "--override-quarantine", str(tmp_path / "override-q.csv"),
        "--ph", "6.9",
        "--overlap-threshold", "0.6",
        "--max-tautomers", "7",
        "--measurement-exclusions", "excluded.csv",
        "--outlier-review-decisions", "outliers.csv",
        "--structure-review-decisions", "structures.csv",
        "--allow-unreviewed-detector-expansion",
    ])
    release_builder.main()
    assert received["allow_unreviewed_detector_expansion"] is True
    assert received["override_paths"] == ["a.csv", "b.csv"]
    assert received["ph"] == pytest.approx(6.9)
    output = capsys.readouterr().out
    assert "candidate-discovery" in output and "dataset_rows=2" in output


def test_network_builder_main_forwards_stage1_and_enumeration_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    received = {}
    monkeypatch.setattr(
        network_builder,
        "build_molecule_dataset",
        lambda **kwargs: received.update(kwargs) or (pd.DataFrame({"x": [1]}), pd.DataFrame()),
    )
    monkeypatch.setattr(sys, "argv", [
        "build_molecule_microstate_network_dataset.py",
        "--transitions", "transitions.csv",
        "--stage1-model", "stage1.pkl",
        "--stage1-metrics", "metrics.json",
        "--output", str(tmp_path / "network.csv"),
        "--quarantine", str(tmp_path / "q.csv"),
        "--ph", "8.1",
        "--overlap-threshold", "0.7",
        "--max-protonation-states", "20",
        "--max-tautomers-per-state", "6",
        "--stage1-workers", "2",
        "--allow-stage1-detector-mismatch",
    ])
    network_builder.main()
    assert received["allow_stage1_detector_mismatch"] is True
    assert received["max_protonation_states"] == 20
    assert received["stage1_workers"] == 2
    assert "molecule_rows=1 quarantine_rows=0" in capsys.readouterr().out


def test_network_measurement_target_mapping_validates_mapped_input() -> None:
    source, indices = network_builder._measurement_target_indices(pd.Series({
        "input_atom_mapped_smiles": "[CH3:1][C:2](=[O:3])[OH:4]",
        "site_atom_maps_json": json.dumps([2, 3, 4]),
    }))
    assert source.GetNumAtoms() == 4
    assert indices == {1, 2, 3}
    with pytest.raises(ValueError, match="unreadable"):
        network_builder._measurement_target_indices(pd.Series({
            "input_atom_mapped_smiles": "bad",
            "site_atom_maps_json": "[]",
        }))


def test_ml_dataset_builder_main_writes_rows_and_rejects_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "candidates.csv"
    monkeypatch.setattr(
        ml_dataset_builder,
        "ensure_curated_training_data",
        lambda **_kwargs: ("assignments.csv", "summary.csv"),
    )
    frame = pd.DataFrame({
        "molecule_key": ["m1", "m1", "m2"],
        "candidate_label": ["a", "b", "c"],
    })
    monkeypatch.setattr(ml_dataset_builder, "build_candidate_dataset", lambda _config: frame.copy())
    monkeypatch.setattr(sys, "argv", [
        "ml_dataset_builder.py",
        "--assignments", "a.csv",
        "--summary", "s.csv",
        "--raw-dir", "raw",
        "--out", str(output),
        "--min-candidates", "2",
    ])
    ml_dataset_builder.main()
    assert len(pd.read_csv(output)) == 3
    assert "Rows: 3 | Molecules: 2" in capsys.readouterr().out
    monkeypatch.setattr(ml_dataset_builder, "build_candidate_dataset", lambda _config: pd.DataFrame())
    with pytest.raises(RuntimeError, match="No candidate rows"):
        ml_dataset_builder.main()


def test_chembl_main_orchestrates_metadata_cleaning_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "chembl"
    raw = output_dir / "raw.csv"
    clean = output_dir / "clean.csv"
    summary = output_dir / "summary.csv"
    checkpoint = output_dir / "checkpoint.json"
    activities = [{
        "activity_id": 1,
        "assay_chembl_id": "A1",
        "target_chembl_id": "T1",
        "molecule_chembl_id": "M1",
    }]
    calls = []
    monkeypatch.setattr(chembl, "fetch_activity_rows", lambda **_kwargs: (activities, {"pages_fetched": 1}))
    monkeypatch.setattr(
        chembl,
        "fetch_entity_map",
        lambda endpoint, **_kwargs: calls.append(endpoint) or ({"A1": {"target_chembl_id": "T2"}} if endpoint == "assay" else {}),
    )
    monkeypatch.setattr(chembl, "fetch_status", lambda *_args: "CHEMBL_SYNTHETIC")
    monkeypatch.setattr(
        chembl,
        "build_raw_dataframe",
        lambda **_kwargs: pd.DataFrame({"activity_id": [1]}),
    )
    monkeypatch.setattr(
        chembl,
        "clean_observations",
        lambda _df: pd.DataFrame({"keep_observation": [True], "drop_reason": [""]}),
    )
    monkeypatch.setattr(
        chembl,
        "summarize_molecules",
        lambda _df: pd.DataFrame({"molecule_chembl_id": ["M1"]}),
    )
    monkeypatch.setattr(sys, "argv", [
        "build_chembl_pka_truth_dataset.py",
        "--out-raw", str(raw),
        "--out-clean", str(clean),
        "--out-summary", str(summary),
        "--checkpoint", str(checkpoint),
        "--limit", "10",
        "--max-pages", "2",
        "--max-records", "3",
        "--timeout", "1",
        "--max-retries", "0",
        "--sleep-seconds", "0",
        "--chunk-size", "2",
    ])
    chembl.main()
    assert calls == ["assay", "target", "molecule"]
    assert raw.exists() and clean.exists() and summary.exists()

    monkeypatch.setattr(chembl, "fetch_activity_rows", lambda **_kwargs: ([], {"pages_fetched": 0}))
    with pytest.raises(RuntimeError, match="No activities"):
        chembl.main()
