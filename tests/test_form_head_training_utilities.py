"""Small synthetic training tests for pair-form classifier heads."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import train_carboxyl_form_head as carboxyl
import train_pair_form_heads as pair_heads


def _carboxyl_rows(classes: list[str]) -> pd.DataFrame:
    labels = {
        "acid_form": "carboxylic_acid",
        "base_form": "carboxylate",
    }
    return pd.DataFrame([
        {
            "source_file": "synthetic.sdf",
            "record_index": index,
            "smiles": "CC(=O)O" if form == "acid_form" else "CC(=O)[O-]",
            "pka_value": 4.0 + 0.1 * index,
            "pka_type_canonical": "acidic",
            "final_group_family": "carboxyl",
            "pair_member_form": form,
            "group_mode": "pair_type",
            "resolved_groups": labels[form],
            "resolved_group_count": 1,
            "final_group_label": labels[form],
            "distribution_ok": True,
        }
        for index, form in enumerate(classes)
    ])


def _simple_features(frame: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
    acid = frame["candidate_label"].astype(str).str.contains("acid").astype(float)
    return pd.DataFrame({"acid_hint": acid, "row_signal": np.arange(len(frame), dtype=float)})


def test_carboxyl_row_preparation_validation_and_split(tmp_path: Path) -> None:
    path = tmp_path / "assignments.csv"
    source = _carboxyl_rows(["acid_form", "base_form", "acid_form"])
    source.to_csv(path, index=False)
    frame = carboxyl._prepare_training_rows(str(path))
    assert frame["target_is_acid_form"].tolist() == [1, 0, 1]
    assert frame["molecule_key"].nunique() == 3
    assert carboxyl._molecule_key(source.iloc[0]).startswith("synthetic.sdf::0::")
    split = carboxyl._split_by_molecule(frame, 0.34, 7)
    assert len(split["test"]) >= 1
    assert set(split["train"]).isdisjoint(set(split["test"]))

    source.drop(columns="resolved_groups").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required"):
        carboxyl._prepare_training_rows(str(path))


def test_carboxyl_main_trains_serializes_and_handles_single_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assignments = tmp_path / "assignments.csv"
    out_dir = tmp_path / "model"
    classes = ["acid_form", "base_form"] * 4
    _carboxyl_rows(classes).to_csv(assignments, index=False)
    monkeypatch.setattr(
        carboxyl,
        "ensure_curated_training_data",
        lambda **_kwargs: (str(assignments), "unused"),
    )
    monkeypatch.setattr(carboxyl, "build_carboxyl_form_feature_frame", _simple_features)
    monkeypatch.setattr(
        carboxyl,
        "_split_by_molecule",
        lambda *_args, **_kwargs: {"train": np.array([0, 1, 2, 3]), "test": np.array([4, 5, 6, 7])},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_carboxyl_form_head.py", "--out-dir", str(out_dir), "--max-rows", "8"],
    )
    carboxyl.main()
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["rows"] == 8 and metrics["test_rows"] == 4
    with (out_dir / "carboxyl_form_head.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    assert bundle["feature_columns"] == ["acid_hint", "row_signal"]

    single = tmp_path / "single.csv"
    _carboxyl_rows(["acid_form", "acid_form"]).to_csv(single, index=False)
    monkeypatch.setattr(carboxyl, "ensure_curated_training_data", lambda **_kwargs: (str(single), "unused"))
    carboxyl.main()
    assert json.loads((out_dir / "metrics.json").read_text())["status"] == "skipped_single_class"


def _candidate_rows() -> pd.DataFrame:
    labels = ["acid", "base"] * 4 + ["rare"]
    return pd.DataFrame([
        {
            "molecule_key": f"m{index}",
            "source_file": "source.sdf",
            "record_index": index,
            "smiles": "CCN",
            "pka_value": 7.0 + index / 10,
            "pka_type_canonical": "basic",
            "candidate_label": label,
            "candidate_instance_id": f"{label}#0",
            "candidate_own_form": "acid_form" if label == "acid" else "base_form",
            "pair_member_form": "base_form",
            "group_mode": "pair_type",
            "resolved_groups": label,
            "resolved_group_count": 1,
            "distribution_ok": True,
        }
        for index, label in enumerate(labels)
    ])


def test_pair_head_prepare_threshold_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _candidate_rows()
    monkeypatch.setattr(pair_heads, "build_candidate_dataset", lambda _config: source.copy())
    prepared = pair_heads._prepare_rows("assignments.csv", "summary.csv", max_rows=4)
    assert len(prepared) == 4
    assert "candidate_family" in prepared
    train, test = pair_heads._split_by_molecule(prepared, 0.25, 3)
    assert len(test) == 1 and len(train) == 3
    threshold = pair_heads._pick_best_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    )
    assert 0.2 < threshold <= 0.8

    monkeypatch.setattr(
        pair_heads,
        "build_candidate_dataset",
        lambda _config: source.drop(columns="resolved_groups"),
    )
    with pytest.raises(ValueError, match="Missing required"):
        pair_heads._prepare_rows("a.csv", "s.csv")
    monkeypatch.setattr(pair_heads, "build_candidate_dataset", lambda _config: pd.DataFrame())
    assert pair_heads._prepare_rows("a.csv", "s.csv").empty


def test_pair_head_main_trains_and_records_skipped_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _candidate_rows()
    out_dir = tmp_path / "heads"
    monkeypatch.setattr(pair_heads, "ensure_curated_training_data", lambda **_kwargs: ("a.csv", "s.csv"))
    monkeypatch.setattr(pair_heads, "_prepare_rows", lambda *_args, **_kwargs: base.copy())
    monkeypatch.setattr(pair_heads, "build_carboxyl_form_feature_frame", _simple_features)
    monkeypatch.setattr(
        pair_heads,
        "HEAD_SPECS",
        {
            "synthetic": {"acid_labels": ["acid"], "base_labels": ["base"]},
            "sparse": {"acid_labels": ["rare"], "base_labels": ["missing"]},
        },
    )
    monkeypatch.setattr(
        pair_heads,
        "_split_by_molecule",
        lambda *_args, **_kwargs: (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_pair_form_heads.py", "--out-dir", str(out_dir), "--min-rows", "2"],
    )
    pair_heads.main()
    metrics = pd.read_csv(out_dir / "metrics.csv")
    assert set(metrics["status"]) == {"trained", "skipped_low_rows"}
    with (out_dir / "pair_form_heads.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    assert set(bundle["heads"]) == {"synthetic"}


def test_pair_head_main_rejects_empty_training_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pair_heads, "ensure_curated_training_data", lambda **_kwargs: ("a", "s"))
    monkeypatch.setattr(pair_heads, "_prepare_rows", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(sys, "argv", ["train_pair_form_heads.py", "--out-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match="No pair-type"):
        pair_heads.main()
