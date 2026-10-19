"""Regression tests for retained legacy diagnostic scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import toy_protonation_assignment_check as toy
import visualize_stage3_failures as legacy_viz


def _candidate_frame() -> pd.DataFrame:
    common = {
        "pred_effective_pka": 7.2,
        "site_prob": 0.7,
        "member_presence_prob": 0.8,
        "selection_confidence": 0.75,
        "pka_eff_confidence": 0.65,
        "pred_effective_pka_ci_low": 6.5,
        "pred_effective_pka_ci_high": 7.9,
        "resolved_group_count": 2,
    }
    return pd.DataFrame([
        {
            **common,
            "molecule_key": "glycine",
            "smiles": "NCC(=O)O",
            "candidate_label": "primary_amine",
            "candidate_instance_id": "primary_amine#0",
            "true_label": "carboxylic_acid",
            "pred_member_form": "base_form",
            "true_pair_member_form": "acid_form",
            "combined_score": 0.9,
            "is_true_site": 0,
        },
        {
            **common,
            "molecule_key": "glycine",
            "smiles": "NCC(=O)O",
            "candidate_label": "carboxylic_acid",
            "candidate_instance_id": np.nan,
            "true_label": "carboxylic_acid",
            "pred_member_form": "acid_form",
            "true_pair_member_form": "acid_form",
            "combined_score": 0.7,
            "pred_effective_pka": np.nan,
            "is_true_site": 1,
        },
        {
            **common,
            "molecule_key": "amine",
            "smiles": "CCN",
            "candidate_label": "primary_amine",
            "candidate_instance_id": "primary_amine#0",
            "true_label": "primary_amine",
            "pred_member_form": "base_form",
            "true_pair_member_form": "base_form",
            "combined_score": 0.95,
            "resolved_group_count": 1,
            "is_true_site": 1,
        },
        {
            **common,
            "molecule_key": "phenol",
            "smiles": "Oc1ccccc1",
            "candidate_label": "phenol",
            "candidate_instance_id": "phenol#0",
            "true_label": "phenol",
            "pred_member_form": "base_form",
            "true_pair_member_form": "acid_form",
            "combined_score": 0.8,
            "resolved_group_count": 1,
            "is_true_site": 1,
        },
    ])


def test_legacy_failure_table_prediction_details_and_legends() -> None:
    frame = _candidate_frame()
    top = legacy_viz._top_prediction_per_molecule(frame)
    outcome = legacy_viz._build_failure_table(top)
    assert set(outcome["failure_type"]) == {"ok", "pair_form", "site+pair_form"}
    predictions = legacy_viz._all_group_predictions(frame)
    assert [row["rank"] for row in predictions["glycine"]] == [1, 2]
    assert predictions["glycine"][1]["group_id"] == "carboxylic_acid"
    assert legacy_viz._format_group_legend([], 3, False) == "NA"
    simple = legacy_viz._format_group_legend(predictions["glycine"], 1, False)
    detailed = legacy_viz._format_group_legend(predictions["glycine"], 0, True)
    assert "primary_amine" in simple
    assert "site=" in detailed and "pKa=NA" in detailed
    report = legacy_viz._format_example_report(outcome, predictions)
    assert "Example 1" in report and "groups:" in report


def test_legacy_site_collection_selection_and_sample_modes() -> None:
    mol, sites = legacy_viz._collect_sites("NCC(=O)O")
    assert mol is not None and sites
    assert legacy_viz._collect_sites("bad") == (None, [])
    assert legacy_viz._best_site_for_label(sites, None) is None
    assert legacy_viz._best_site_for_label([], "primary_amine") is None
    assert legacy_viz._best_site_for_label(sites, "primary_amine") is not None
    assert legacy_viz._best_site_for_label(sites, "missing") is None
    assert "#" in legacy_viz._resolved_site_summary(sites)

    outcome = legacy_viz._build_failure_table(
        legacy_viz._top_prediction_per_molecule(_candidate_frame())
    )
    failures = legacy_viz._select_example_rows(outcome, "failures", "all", 10, 0)
    pair = legacy_viz._select_example_rows(outcome, "failures", "pair_form", 10, 0)
    correct = legacy_viz._select_example_rows(outcome, "correct", "all", 10, 9)
    headline = legacy_viz._select_example_rows(outcome, "headline", "all", 10, 1)
    assert set(failures["failure_type"]) == {"pair_form", "site+pair_form"}
    assert set(pair["failure_type"]) == {"pair_form"}
    assert (correct["failure_type"] == "ok").all()
    assert len(headline) == 1


def test_legacy_grid_and_cli_write_all_diagnostic_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _candidate_frame()
    top = legacy_viz._top_prediction_per_molecule(frame)
    outcome = legacy_viz._build_failure_table(top)
    predictions = legacy_viz._all_group_predictions(frame)
    png = tmp_path / "grid.png"
    table = tmp_path / "grid.csv"
    groups = tmp_path / "groups.csv"
    legacy_viz._render_grid(
        outcome,
        predictions,
        png,
        table,
        groups,
        max_examples=10,
        show_all_groups=True,
        legend_full_details=True,
        max_legend_groups=0,
    )
    assert png.exists() and table.exists() and groups.exists()
    assert not pd.read_csv(groups).empty

    input_csv = tmp_path / "scores.csv"
    frame.to_csv(input_csv, index=False)
    out_dir = tmp_path / "cli"
    monkeypatch.setattr(sys, "argv", [
        "visualize_stage3_failures.py",
        "--in-csv", str(input_csv),
        "--out-dir", str(out_dir),
        "--example-set", "failures",
        "--failure-type", "site+pair_form",
        "--hide-all-groups",
    ])
    legacy_viz.main()
    assert (out_dir / "stage3_failures_site_plus_pair_form.png").exists()
    assert (out_dir / "stage3_failures_site_plus_pair_form.txt").exists()
    assert (out_dir / "stage3_failure_summary.csv").exists()

    missing = frame.drop(columns="site_prob")
    missing.to_csv(input_csv, index=False)
    with pytest.raises(ValueError, match="Missing required"):
        legacy_viz.main()


def test_toy_classifier_determinism_invalid_input_and_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acid = toy.classify_smiles("CC(=O)O")
    amine = toy.classify_smiles("CN")
    invalid = toy.classify_smiles("bad")
    assert acid["final_group_label"] == "carboxylic_acid"
    assert amine["group_mode"] == "pair_type"
    assert amine["neutral_input_risk"] is True
    assert invalid["resolved_group_count"] == 0
    assert toy.run_determinism_passes("CN", 4)

    calls = 0

    def alternating(_smiles):
        nonlocal calls
        calls += 1
        return {
            "final_group_label": str(calls % 2),
            "resolved_groups": "x",
            "resolved_group_count": 1,
        }

    monkeypatch.setattr(toy, "classify_smiles", alternating)
    assert not toy.run_determinism_passes("CN", 3)
    monkeypatch.undo()
    monkeypatch.chdir(tmp_path)
    toy.main()
    output = tmp_path / "data" / "processed" / "toy_protonation_assignment_check.csv"
    result = pd.read_csv(output)
    assert len(result) == len(toy.TOY_CASES)
    assert result["deterministic_over_3_runs"].all()
