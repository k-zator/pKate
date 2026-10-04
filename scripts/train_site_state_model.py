"""LEGACY candidate-ranking model retained for reproducibility only.

Canonical Stage 3 obtains site marginals from the Stage 2 pairwise free-energy
model and does not use this classifier.
"""

import argparse
import json
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier # type: ignore
from sklearn.metrics import average_precision_score, roc_auc_score # type: ignore

from ml_dataset_builder import DatasetBuildConfig, build_candidate_dataset
from ml_features import build_feature_matrix
from ml_splits import random_molecule_split, scaffold_molecule_split, split_manifest
from stage3_protonation_state_inference import (
    _build_stage1_like_features,
    _build_stage2_features,
    _load_bundle,
)
from training_data_resolver import (
    DEFAULT_CURATED_ASSIGNMENTS_PATH,
    DEFAULT_CURATED_SUMMARY_PATH,
    DEFAULT_RAW_DIR,
    ensure_curated_training_data,
)


DEFAULT_STAGE1_MODEL_PATH = "data/processed/ml_models/stage1_intrinsic/stage1_intrinsic_model.pkl"
DEFAULT_STAGE2_MODEL_PATH = "data/processed/ml_models/stage2_delta/stage2_delta_model.pkl"


def _topk_site_accuracy(
    frame: pd.DataFrame,
    proba_col: str = "pred_proba",
    k: int = 1,
) -> float:
    if frame.empty:
        return 0.0
    grouped = frame.groupby("molecule_key", sort=False)
    correct = 0
    total = 0
    for _, g in grouped:
        ranked = g.sort_values(proba_col, ascending=False).head(k)
        hit = int((ranked["is_true_site"] == 1).any())
        correct += hit
        total += 1
    return float(correct / max(1, total))


def _fit_and_eval(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[HistGradientBoostingClassifier, Dict[str, float], pd.DataFrame]:
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        random_state=17,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    test_proba = model.predict_proba(X.iloc[test_idx])[:, 1]
    eval_df = df.iloc[test_idx].copy()
    eval_df["pred_proba"] = test_proba

    metrics = {
        "auroc": float(roc_auc_score(y.iloc[test_idx], test_proba)) if y.iloc[test_idx].nunique() > 1 else 0.0,
        "auprc": float(average_precision_score(y.iloc[test_idx], test_proba)),
        "top1_site_acc": _topk_site_accuracy(eval_df, k=1),
        "top3_site_acc": _topk_site_accuracy(eval_df, k=3),
    }
    return model, metrics, eval_df


def _add_predicted_pka_features(
    candidate_df: pd.DataFrame,
    stage1_model_path: str,
    stage2_model_path: str,
) -> pd.DataFrame:
    stage1_bundle = _load_bundle(stage1_model_path)
    stage2_bundle = _load_bundle(stage2_model_path)

    stage1_model = stage1_bundle.get("model")
    stage1_columns = stage1_bundle.get("feature_columns")
    if stage1_model is None or stage1_columns is None:
        raise ValueError(f"Stage 1 bundle missing required keys: {stage1_model_path}")

    stage2_model = stage2_bundle.get("model")
    stage2_columns = stage2_bundle.get("feature_columns")
    if stage2_model is None or stage2_columns is None:
        raise ValueError(f"Stage 2 bundle missing required keys: {stage2_model_path}")

    work = candidate_df.copy()
    X_stage1 = _build_stage1_like_features(
        work,
        feature_columns=stage1_columns,
        fp_bits=int(stage1_bundle.get("fp_bits", 512)),
        fp_radius=int(stage1_bundle.get("fp_radius", 2)),
    )
    work["intrinsic_pred_pka"] = stage1_model.predict(X_stage1)

    is_multi = pd.to_numeric(work["resolved_group_count"], errors="coerce").fillna(1) > 1
    work["pred_delta_pka"] = 0.0
    if is_multi.any():
        X_stage2 = _build_stage2_features(
            work.loc[is_multi],
            feature_columns=stage2_columns,
            fp_bits=int(stage2_bundle.get("fp_bits", 512)),
            fp_radius=int(stage2_bundle.get("fp_radius", 2)),
        )
        work.loc[is_multi, "pred_delta_pka"] = stage2_model.predict(X_stage2)

    work["pred_effective_pka"] = work["intrinsic_pred_pka"] + work["pred_delta_pka"]
    return work


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train candidate-ranking site/state model.")
    parser.add_argument("--assignments", default=DEFAULT_CURATED_ASSIGNMENTS_PATH)
    parser.add_argument("--summary", default=DEFAULT_CURATED_SUMMARY_PATH)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--stage1-model", default=DEFAULT_STAGE1_MODEL_PATH)
    parser.add_argument("--stage2-model", default=DEFAULT_STAGE2_MODEL_PATH)
    parser.add_argument("--out-dir", default="data/processed/ml_models/site_state_baseline")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--fp-bits", type=int, default=512)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument(
        "--include-observed-pka",
        action="store_true",
        help="Include observed pKa as an input feature (analysis only; not for dull-SMILES deployment).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    raw_dir = getattr(args, "raw_dir", DEFAULT_RAW_DIR)
    stage1_model_path = getattr(args, "stage1_model", DEFAULT_STAGE1_MODEL_PATH)
    stage2_model_path = getattr(args, "stage2_model", DEFAULT_STAGE2_MODEL_PATH)

    assignments_path, summary_path = ensure_curated_training_data(
        assignments_path=args.assignments,
        summary_path=args.summary,
        raw_dir=raw_dir,
    )
    config = DatasetBuildConfig(assignments_path=assignments_path, summary_path=summary_path)
    candidate_df = build_candidate_dataset(config)
    if candidate_df.empty:
        raise RuntimeError("Candidate dataset is empty.")
    candidate_df = _add_predicted_pka_features(
        candidate_df,
        stage1_model_path=stage1_model_path,
        stage2_model_path=stage2_model_path,
    )

    X, y = build_feature_matrix(
        candidate_df,
        nbits=args.fp_bits,
        radius=args.fp_radius,
        include_observed_pka=args.include_observed_pka,
    )

    r_train, r_test = random_molecule_split(candidate_df, test_frac=args.test_frac)
    s_train, s_test = scaffold_molecule_split(candidate_df, test_frac=args.test_frac)

    random_model, random_metrics, random_eval = _fit_and_eval(candidate_df, X, y, r_train, r_test)
    scaffold_model, scaffold_metrics, scaffold_eval = _fit_and_eval(candidate_df, X, y, s_train, s_test)

    random_bundle = {
        "model": random_model,
        "feature_columns": list(X.columns),
        "fp_bits": int(args.fp_bits),
        "fp_radius": int(args.fp_radius),
        "include_observed_pka": bool(args.include_observed_pka),
    }
    scaffold_bundle = {
        "model": scaffold_model,
        "feature_columns": list(X.columns),
        "fp_bits": int(args.fp_bits),
        "fp_radius": int(args.fp_radius),
        "include_observed_pka": bool(args.include_observed_pka),
    }

    with open(os.path.join(args.out_dir, "site_state_random_model.pkl"), "wb") as handle:
        pickle.dump(random_bundle, handle)
    with open(os.path.join(args.out_dir, "site_state_scaffold_model.pkl"), "wb") as handle:
        pickle.dump(scaffold_bundle, handle)

    random_eval.to_csv(os.path.join(args.out_dir, "random_eval_candidates.csv"), index=False)
    scaffold_eval.to_csv(os.path.join(args.out_dir, "scaffold_eval_candidates.csv"), index=False)

    random_manifest = split_manifest(candidate_df, r_train, r_test)
    scaffold_manifest = split_manifest(candidate_df, s_train, s_test)
    random_manifest.to_csv(os.path.join(args.out_dir, "random_split_manifest.csv"), index=False)
    scaffold_manifest.to_csv(os.path.join(args.out_dir, "scaffold_split_manifest.csv"), index=False)

    report = {
        "rows": int(len(candidate_df)),
        "molecules": int(candidate_df["molecule_key"].nunique()),
        "positive_rate": float(y.mean()),
        "include_observed_pka": bool(args.include_observed_pka),
        "uses_predicted_pka_features": True,
        "random": random_metrics,
        "scaffold": scaffold_metrics,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Saved training artifacts to", args.out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
