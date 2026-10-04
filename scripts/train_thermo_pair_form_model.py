"""LEGACY candidate pair-form model retained for reproducibility only.

Canonical Stage 3 predicts complete molecular microstates from the Stage 2
pairwise free-energy model and does not use this classifier.
"""

import argparse
import json
import os
import pickle
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from ml_splits import random_molecule_split, scaffold_molecule_split, split_manifest
from stage3_protonation_state_inference import (
    DEFAULT_STAGE3_OUTPUT_DIR,
    _apply_uncertainty_ph_fallback,
    _label_form,
    _molecule_level_summary,
)
from thermo_pair_form_model import apply_thermo_pair_form_model, build_thermo_pair_form_feature_frame


DEFAULT_STAGE3_CSV = os.path.join(DEFAULT_STAGE3_OUTPUT_DIR, "stage3_candidate_scores.csv")


def _reconstruct_pre_head_state(df: pd.DataFrame, pH: float, uncertain_threshold: float) -> pd.DataFrame:
    work = df.copy()
    work["pred_member_form"] = work["pred_member_form_baseline"]
    work["pred_member_form_rule_source"] = "baseline_henderson"
    drop_cols = [
        col
        for col in work.columns
        if col == "carboxyl_form_prob_acid" or col.startswith("pair_head_prob_acid__") or col == "thermo_pair_form_prob_acid"
    ]
    if drop_cols:
        work = work.drop(columns=drop_cols)
    return _apply_uncertainty_ph_fallback(work, pH=pH, uncertain_threshold=uncertain_threshold)


def _pair_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (pd.to_numeric(df.get("is_true_site", 0), errors="coerce").fillna(0) == 1)
        & df["true_pair_member_form"].isin(["acid_form", "base_form"])
        & df["candidate_member_form"].isin(["acid_form", "base_form"])
    )
    out = df.loc[mask].copy().reset_index(drop=True)
    if out.empty:
        raise RuntimeError("No true-site pair-form rows available for thermodynamic model training.")
    out["target_is_acid_form"] = (out["true_pair_member_form"] == "acid_form").astype(int)
    return out


def _top1_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("combined_score", ascending=False).groupby("molecule_key", as_index=False).first()


def _carboxyl_top1_pair_acc(df: pd.DataFrame) -> float:
    top1 = _top1_rows(df).copy()
    top1["true_family"] = top1["true_label"].map(lambda value: _label_form(str(value))[0] if pd.notna(value) else None)
    pair_df = top1[
        top1["true_pair_member_form"].isin(["acid_form", "base_form"])
        & (top1["true_family"] == "carboxyl")
    ].copy()
    if pair_df.empty:
        return float("nan")
    return float((pair_df["pred_member_form"] == pair_df["true_pair_member_form"]).mean())


def _score_variant(df: pd.DataFrame) -> Dict[str, float]:
    summary = _molecule_level_summary(df)
    summary["carboxyl_top1_pair_form_acc"] = _carboxyl_top1_pair_acc(df)
    return summary


def _fit_model(train_rows: pd.DataFrame, feature_mode: str) -> Tuple[Pipeline, pd.DataFrame]:
    X = build_thermo_pair_form_feature_frame(train_rows, feature_mode=feature_mode)
    y = train_rows["target_is_acid_form"].astype(int)
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, class_weight="balanced")),
        ]
    )
    model.fit(X, y)
    return model, X


def _row_level_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _tune_threshold_and_margin(
    seed_eval_df: pd.DataFrame,
    acid_prob: pd.Series,
) -> Tuple[float, float, Dict[str, float]]:
    best_score = None
    best_threshold = 0.5
    best_margin = 0.0
    best_metrics: Dict[str, float] = {}

    for threshold in np.linspace(0.15, 0.85, 29):
        for margin in np.linspace(0.0, 0.20, 9):
            work = seed_eval_df.copy()
            work["thermo_pair_form_prob_acid"] = acid_prob
            work["pred_member_form_rule_source"] = work["pred_member_form_rule_source"].fillna("baseline_henderson")
            mask = (
                work["candidate_member_form"].isin(["acid_form", "base_form"])
                & work["thermo_pair_form_prob_acid"].notna()
                & ((work["thermo_pair_form_prob_acid"] - threshold).abs() >= margin)
            )
            work.loc[mask, "pred_member_form"] = np.where(
                work.loc[mask, "thermo_pair_form_prob_acid"] >= threshold,
                "acid_form",
                "base_form",
            )
            work.loc[mask, "pred_member_form_rule_source"] = "thermo_pair_form_model"
            metrics = _score_variant(work)
            score_tuple = (
                metrics.get("pair_form_acc", float("nan")),
                metrics.get("carboxyl_top1_pair_form_acc", float("nan")),
                -float(mask.sum()),
            )
            if best_score is None or score_tuple > best_score:
                best_score = score_tuple
                best_threshold = float(threshold)
                best_margin = float(margin)
                best_metrics = metrics

    return best_threshold, best_margin, best_metrics


def _evaluate_split(
    name: str,
    current_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    true_site_df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_mode: str,
) -> Tuple[Dict, pd.DataFrame, Dict]:
    train_rows = true_site_df.iloc[train_idx].copy()
    test_rows = true_site_df.iloc[test_idx].copy()
    train_keys = set(train_rows["molecule_key"])
    test_keys = set(test_rows["molecule_key"])

    model, X_train = _fit_model(train_rows, feature_mode=feature_mode)

    train_seed = seed_df[seed_df["molecule_key"].isin(train_keys)].copy()
    test_seed = seed_df[seed_df["molecule_key"].isin(test_keys)].copy()
    current_train = current_df[current_df["molecule_key"].isin(train_keys)].copy()
    current_test = current_df[current_df["molecule_key"].isin(test_keys)].copy()

    train_prob = pd.Series(
        model.predict_proba(
            build_thermo_pair_form_feature_frame(train_seed, feature_mode=feature_mode).reindex(
                columns=X_train.columns,
                fill_value=0.0,
            )
        )[:, 1],
        index=train_seed.index,
    )
    decision_threshold, decision_margin, tuned_train_metrics = _tune_threshold_and_margin(train_seed, train_prob)

    bundle = {
        "model": model,
        "feature_columns": list(X_train.columns),
        "feature_mode": feature_mode,
        "decision_threshold": float(decision_threshold),
        "decision_margin": float(decision_margin),
    }

    variant_test = apply_thermo_pair_form_model(test_seed, bundle)

    test_true_prob = model.predict_proba(
        build_thermo_pair_form_feature_frame(test_rows, feature_mode=feature_mode).reindex(
            columns=X_train.columns,
            fill_value=0.0,
        )
    )[:, 1]
    test_row_metrics = _row_level_metrics(
        test_rows["target_is_acid_form"].to_numpy(dtype=int),
        test_true_prob,
        threshold=decision_threshold,
    )

    current_metrics = _score_variant(current_test)
    seed_metrics = _score_variant(test_seed)
    variant_metrics = _score_variant(variant_test)

    current_top1 = _top1_rows(current_test)[["molecule_key", "pred_member_form"]].rename(columns={"pred_member_form": "pred_member_form_current"})
    seed_top1 = _top1_rows(test_seed)[["molecule_key", "pred_member_form"]].rename(columns={"pred_member_form": "pred_member_form_seed"})
    variant_top1 = _top1_rows(variant_test).copy()
    compare = variant_top1.merge(current_top1, on="molecule_key", how="left").merge(seed_top1, on="molecule_key", how="left")
    compare["changed_vs_current"] = compare["pred_member_form"] != compare["pred_member_form_current"]
    compare["changed_vs_seed"] = compare["pred_member_form"] != compare["pred_member_form_seed"]

    report = {
        "feature_mode": feature_mode,
        "feature_count": int(len(X_train.columns)),
        "rows": int(len(test_rows)),
        "molecules": int(test_rows["molecule_key"].nunique()),
        "decision_threshold": float(decision_threshold),
        "decision_margin": float(decision_margin),
        "train_tuned_metrics": tuned_train_metrics,
        "row_level_test_metrics": test_row_metrics,
        "current": current_metrics,
        "no_head": seed_metrics,
        "thermo_pair_model": variant_metrics,
        "top1_changed_vs_current": int(compare["changed_vs_current"].sum()),
        "top1_changed_vs_no_head": int(compare["changed_vs_seed"].sum()),
        "top1_rule_source_counts": variant_top1["pred_member_form_rule_source"].value_counts(dropna=False).to_dict(),
    }
    return report, variant_test, bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a unified thermodynamic pair-form model.")
    parser.add_argument("--stage3-csv", default=DEFAULT_STAGE3_CSV)
    parser.add_argument("--out-dir", default="data/processed/ml_models_pruned_eval_unpooled_families_predicted_pka/thermo_pair_form_model_weak_prior")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--feature-mode", choices=["full", "weak_prior"], default="weak_prior")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    current_df = pd.read_csv(args.stage3_csv, low_memory=False)
    if current_df.empty:
        raise RuntimeError("Stage 3 candidate CSV is empty.")

    ph = float(pd.to_numeric(current_df.get("ph_target", 7.4), errors="coerce").fillna(7.4).iloc[0])
    uncertain_threshold = 0.55
    if "carboxyl_adjusted_ph_threshold" in current_df.columns:
        # The saved Stage 3 CSV does not persist uncertain_threshold, so use the promoted default.
        uncertain_threshold = 0.55

    seed_df = _reconstruct_pre_head_state(current_df, pH=ph, uncertain_threshold=uncertain_threshold)
    true_site_df = _pair_target_rows(current_df)

    random_train, random_test = random_molecule_split(
        true_site_df,
        test_frac=float(args.test_frac),
        seed=int(args.seed),
    )
    scaffold_train, scaffold_test = scaffold_molecule_split(
        true_site_df,
        test_frac=float(args.test_frac),
    )

    random_report, random_eval, random_bundle = _evaluate_split(
        name="random",
        current_df=current_df,
        seed_df=seed_df,
        true_site_df=true_site_df,
        train_idx=random_train,
        test_idx=random_test,
        feature_mode=str(args.feature_mode),
    )
    scaffold_report, scaffold_eval, scaffold_bundle = _evaluate_split(
        name="scaffold",
        current_df=current_df,
        seed_df=seed_df,
        true_site_df=true_site_df,
        train_idx=scaffold_train,
        test_idx=scaffold_test,
        feature_mode=str(args.feature_mode),
    )

    with open(os.path.join(args.out_dir, "thermo_pair_form_model.pkl"), "wb") as handle:
        pickle.dump(scaffold_bundle, handle)

    random_eval.to_csv(os.path.join(args.out_dir, "random_eval_candidates.csv"), index=False)
    scaffold_eval.to_csv(os.path.join(args.out_dir, "scaffold_eval_candidates.csv"), index=False)
    split_manifest(true_site_df, random_train, random_test).to_csv(
        os.path.join(args.out_dir, "random_split_manifest.csv"),
        index=False,
    )
    split_manifest(true_site_df, scaffold_train, scaffold_test).to_csv(
        os.path.join(args.out_dir, "scaffold_split_manifest.csv"),
        index=False,
    )

    report = {
        "stage3_csv": args.stage3_csv,
        "feature_mode": str(args.feature_mode),
        "rows": int(len(current_df)),
        "true_site_pair_rows": int(len(true_site_df)),
        "molecules": int(true_site_df["molecule_key"].nunique()),
        "ph": ph,
        "random": random_report,
        "scaffold": scaffold_report,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Saved thermodynamic pair-form artifacts to {args.out_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
