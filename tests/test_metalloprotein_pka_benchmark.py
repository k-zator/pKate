"""Contracts for the locked 2026 metalloprotein pKa stress test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_metalloprotein_pka as benchmark


def test_repository_metalloprotein_panel_is_locked_and_evaluation_only():
    panel = benchmark.read_panel()
    assert len(panel) == 12
    assert int(panel["qualifier"].eq("exact").sum()) == 8
    assert int(panel["qualifier"].eq("approximate").sum()) == 3
    assert int(panel["qualifier"].eq("upper_bound").sum()) == 1
    assert panel["label_origin"].eq("experimental").all()
    assert panel["computational_teacher"].eq("").all()
    assert panel["source_doi"].eq("10.1021/acs.jctc.6c00884").all()


def test_metalloprotein_scorer_keeps_approximate_and_bound_labels_separate():
    panel = benchmark.read_panel().iloc[[0, 6, 8]].copy()
    predictions = pd.DataFrame([
        {"case_id": "adh_tfe", "predicted_pka": 4.8,
         "prediction_semantics": benchmark.PREDICTION_SEMANTICS},
        {"case_id": "ca_wt", "predicted_pka": 6.8,
         "prediction_semantics": benchmark.PREDICTION_SEMANTICS},
        {"case_id": "mb_compound_ii", "predicted_pka": 3.0,
         "prediction_semantics": benchmark.PREDICTION_SEMANTICS},
    ])
    scored, metrics = benchmark.score_predictions(panel, predictions)
    assert metrics["exact_mae"] == pytest.approx(0.5)
    assert metrics["approximate_mae"] == pytest.approx(0.2)
    assert metrics["upper_bound_satisfaction"] == 0.0
    assert scored.loc[
        scored["case_id"].eq("mb_compound_ii"), "bound_violation"
    ].iloc[0] == pytest.approx(0.3)


def test_metalloprotein_panel_rejects_training_eligibility(tmp_path):
    panel = benchmark.read_panel()
    panel.loc[0, "eligible_for_training"] = True
    path = tmp_path / "bad.csv"
    panel.to_csv(path, index=False)
    with pytest.raises(ValueError, match="evaluation-only"):
        benchmark.read_panel(path)


def test_repository_published_qm_predictions_are_external_and_score_reproducibly():
    panel = benchmark.read_panel()
    predictions = pd.read_csv(benchmark.DEFAULT_PUBLISHED_PREDICTIONS)
    assert len(predictions) == 12
    assert predictions["prediction_origin"].eq(
        "published_external_computation"
    ).all()
    assert predictions["eligible_for_training"].astype(str).str.lower().eq(
        "false"
    ).all()
    assert np.allclose(
        predictions["raw_calculated_pka"] - predictions["systematic_bias_removed"],
        predictions["predicted_pka"],
    )
    _scored, metrics = benchmark.score_predictions(panel, predictions)
    assert metrics["exact_mae"] == pytest.approx(1.525)
    assert metrics["approximate_mae"] == pytest.approx(4.2)
    assert metrics["upper_bound_satisfaction"] == 0.0


def test_metalloprotein_predictions_cannot_be_relabelled_as_training_data():
    panel = benchmark.read_panel()
    predictions = pd.read_csv(benchmark.DEFAULT_PUBLISHED_PREDICTIONS).iloc[:1].copy()
    predictions["eligible_for_training"] = True
    with pytest.raises(ValueError, match="must not be training labels"):
        benchmark.score_predictions(panel, predictions)
