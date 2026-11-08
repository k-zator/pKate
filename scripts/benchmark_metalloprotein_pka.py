#!/usr/bin/env python3
"""Validate or score the locked 2026 metalloprotein pKa benchmark.

The twelve Table 1 labels exercise metal-bound water, alcohol, serine and
hydroxide chemistry outside the current canonical residue fit.  They are
therefore a specialist, evaluation-only panel: this script never trains or
updates a model.  Exact and approximate point labels are reported separately;
the one censored value is evaluated only as an upper bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PROJECT_ROOT / "data/benchmarks/metalloprotein_pka_2026.csv"
DEFAULT_PUBLISHED_PREDICTIONS = (
    PROJECT_ROOT
    / "data/benchmarks/metalloprotein_pka_2026_published_qm_predictions.csv"
)
ALLOWED_QUALIFIERS = frozenset({"exact", "approximate", "upper_bound"})
PREDICTION_SEMANTICS = "individual_metal_bound_site_pka"


def read_panel(path: str | Path = DEFAULT_PANEL) -> pd.DataFrame:
    """Read the panel and fail closed on provenance or training leakage."""
    panel = pd.read_csv(path, dtype={"pdb": str}, keep_default_na=False)
    required = {
        "case_id", "pdb", "experimental_pka", "qualifier",
        "prediction_semantics", "label_origin", "computational_teacher",
        "evaluation_only", "eligible_for_training", "source_doi",
    }
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Metalloprotein benchmark missing columns: {missing}")
    if panel["case_id"].duplicated().any():
        raise ValueError("Metalloprotein benchmark case IDs must be unique")
    if not set(panel["qualifier"]).issubset(ALLOWED_QUALIFIERS):
        raise ValueError("Metalloprotein benchmark has an unknown qualifier")
    if not panel["prediction_semantics"].eq(PREDICTION_SEMANTICS).all():
        raise ValueError("Metalloprotein benchmark prediction semantics changed")
    if not panel["label_origin"].eq("experimental").all():
        raise ValueError("Metalloprotein benchmark contains nonexperimental labels")
    if panel["computational_teacher"].str.strip().ne("").any():
        raise ValueError("Metalloprotein benchmark contains teacher-derived labels")
    evaluation = panel["evaluation_only"].astype(str).str.lower()
    eligible = panel["eligible_for_training"].astype(str).str.lower()
    if evaluation.ne("true").any() or eligible.ne("false").any():
        raise ValueError("Metalloprotein benchmark must remain evaluation-only")
    panel["experimental_pka"] = pd.to_numeric(
        panel["experimental_pka"], errors="raise"
    )
    if not np.isfinite(panel["experimental_pka"]).all():
        raise ValueError("Metalloprotein benchmark contains nonfinite pKa labels")
    return panel


def score_predictions(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Score submitted pKa values without treating bounds as point targets."""
    required = {"case_id", "predicted_pka", "prediction_semantics"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table missing columns: {missing}")
    if predictions["case_id"].duplicated().any():
        raise ValueError("Prediction case IDs must be unique")
    unknown = sorted(set(predictions["case_id"]) - set(panel["case_id"]))
    if unknown:
        raise ValueError(f"Unknown metalloprotein case IDs: {unknown}")
    submitted = predictions.copy()
    submitted["predicted_pka"] = pd.to_numeric(
        submitted["predicted_pka"], errors="raise"
    )
    if not submitted["prediction_semantics"].eq(PREDICTION_SEMANTICS).all():
        raise ValueError("Prediction semantics do not match the benchmark")
    if "eligible_for_training" in submitted:
        eligible = submitted["eligible_for_training"].astype(str).str.lower()
        if eligible.ne("false").any():
            raise ValueError("Metalloprotein predictions must not be training labels")
    if "evaluation_only" in submitted:
        evaluation = submitted["evaluation_only"].astype(str).str.lower()
        if evaluation.ne("true").any():
            raise ValueError("Metalloprotein predictions must remain evaluation-only")
    scored = panel.merge(submitted, on=["case_id", "prediction_semantics"])
    if scored.empty:
        raise ValueError("No benchmark predictions were submitted")
    scored["absolute_error"] = np.where(
        scored["qualifier"].isin(["exact", "approximate"]),
        np.abs(scored["predicted_pka"] - scored["experimental_pka"]),
        np.nan,
    )
    scored["bound_satisfied"] = np.where(
        scored["qualifier"].eq("upper_bound"),
        scored["predicted_pka"] <= scored["experimental_pka"],
        np.nan,
    )
    scored["bound_violation"] = np.where(
        scored["qualifier"].eq("upper_bound"),
        np.maximum(scored["predicted_pka"] - scored["experimental_pka"], 0.0),
        np.nan,
    )

    def mae(qualifier: str) -> float | None:
        values = scored.loc[
            scored["qualifier"].eq(qualifier), "absolute_error"
        ].dropna()
        return float(values.mean()) if len(values) else None

    bounds = scored.loc[scored["qualifier"].eq("upper_bound")]
    metrics: Dict[str, object] = {
        "panel_rows": int(len(panel)),
        "submitted_rows": int(len(scored)),
        "exact_rows": int(scored["qualifier"].eq("exact").sum()),
        "exact_mae": mae("exact"),
        "approximate_rows": int(scored["qualifier"].eq("approximate").sum()),
        "approximate_mae": mae("approximate"),
        "upper_bound_rows": int(len(bounds)),
        "upper_bound_satisfaction": (
            float(bounds["bound_satisfied"].astype(bool).mean())
            if len(bounds) else None
        ),
        "mean_upper_bound_violation": (
            float(bounds["bound_violation"].mean()) if len(bounds) else None
        ),
        "evaluation_only": True,
    }
    return scored, metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    parser.add_argument("--predictions")
    parser.add_argument(
        "--published-qm",
        action="store_true",
        help=(
            "Score the committed external QM-cluster predictions transcribed "
            "from the paper's CC BY-NC Supporting Information."
        ),
    )
    parser.add_argument("--scored-output")
    parser.add_argument("--metrics-output")
    args = parser.parse_args(argv)
    panel = read_panel(args.panel)
    if args.predictions and args.published_qm:
        parser.error("use either --predictions or --published-qm, not both")
    prediction_path = (
        str(DEFAULT_PUBLISHED_PREDICTIONS)
        if args.published_qm else args.predictions
    )
    if prediction_path:
        submitted = pd.read_csv(prediction_path)
        scored, report = score_predictions(panel, submitted)
        if "prediction_origin" in submitted:
            report["prediction_origin"] = sorted(
                submitted["prediction_origin"].astype(str).unique().tolist()
            )
        if "method" in submitted:
            report["methods"] = sorted(
                submitted["method"].astype(str).unique().tolist()
            )
        report["pKate_model_prediction"] = False
        report["computational_teacher_used_for_training"] = False
        if args.scored_output:
            Path(args.scored_output).parent.mkdir(parents=True, exist_ok=True)
            scored.to_csv(args.scored_output, index=False)
    else:
        report = {
            "panel_rows": int(len(panel)),
            "exact_rows": int(panel["qualifier"].eq("exact").sum()),
            "approximate_rows": int(panel["qualifier"].eq("approximate").sum()),
            "upper_bound_rows": int(panel["qualifier"].eq("upper_bound").sum()),
            "evaluation_only": True,
            "status": "validated_without_predictions",
        }
    if args.metrics_output:
        destination = Path(args.metrics_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
