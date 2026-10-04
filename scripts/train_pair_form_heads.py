import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score  # type: ignore

from functional_group_pka_analysis import CONJUGATE_FAMILY_MAP
from ionization_context_features import build_carboxyl_form_feature_frame
from ml_dataset_builder import DatasetBuildConfig, build_candidate_dataset
from training_data_resolver import (
    DEFAULT_CURATED_ASSIGNMENTS_PATH,
    DEFAULT_CURATED_SUMMARY_PATH,
    DEFAULT_RAW_DIR,
    ensure_curated_training_data,
)


HEAD_SPECS: Dict[str, Dict[str, List[str]]] = {
    "carboxyl": {
        "acid_labels": ["carboxylic_acid"],
        "base_labels": ["carboxylate"],
    },
    "primary_amine": {
        "acid_labels": ["primary_ammonium"],
        "base_labels": ["primary_amine"],
    },
    "secondary_amine": {
        "acid_labels": ["secondary_ammonium"],
        "base_labels": ["secondary_amine"],
    },
    "tertiary_amine": {
        "acid_labels": ["tertiary_ammonium"],
        "base_labels": ["tertiary_amine"],
    },
    "aryl_amine": {
        "acid_labels": ["aryl_ammonium"],
        "base_labels": ["aniline"],
    },
    "pyridine_like": {
        "acid_labels": ["pyridinium", "quinolinium", "isoquinolinium"],
        "base_labels": ["pyridine", "quinoline", "isoquinoline"],
    },
    "imidazole_like": {
        "acid_labels": ["imidazolium"],
        "base_labels": ["imidazole"],
    },
    "pyrimidine_like": {
        "acid_labels": ["pyrimidinium"],
        "base_labels": ["pyrimidine"],
    },
    "pyrazine_like": {
        "acid_labels": ["pyrazinium"],
        "base_labels": ["pyrazine"],
    },
    "pyrrole_like": {
        "acid_labels": ["pyrrolium"],
        "base_labels": ["pyrrole"],
    },
    "triazine_like": {
        "acid_labels": ["triazinium"],
        "base_labels": ["triazine"],
    },
    "imine_like": {
        "acid_labels": ["iminium"],
        "base_labels": ["imine"],
    },
}


def _molecule_key(row: pd.Series) -> str:
    return f"{row['source_file']}::{int(row['record_index'])}::{row['smiles']}"


def _split_by_molecule(df: pd.DataFrame, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mols = np.array(sorted(df["molecule_key"].unique()))
    rng.shuffle(mols)
    n_test = max(1, int(len(mols) * test_frac))
    test_mols = set(mols[:n_test])
    test_mask = df["molecule_key"].isin(test_mols).to_numpy()
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, test_idx


def _prepare_rows(
    assignments_path: str,
    summary_path: str = DEFAULT_CURATED_SUMMARY_PATH,
    max_rows: int = 0,
) -> pd.DataFrame:
    cand = build_candidate_dataset(
        DatasetBuildConfig(
            assignments_path=assignments_path,
            summary_path=summary_path,
            min_candidates=1,
        )
    )
    if cand.empty:
        return cand

    required = {
        "molecule_key",
        "source_file",
        "record_index",
        "smiles",
        "candidate_label",
        "candidate_own_form",
        "pair_member_form",
        "group_mode",
        "resolved_groups",
        "resolved_group_count",
        "distribution_ok",
    }
    missing = required - set(cand.columns)
    if missing:
        raise ValueError(f"Missing required candidate columns: {sorted(missing)}")

    frame = cand[
        (cand["group_mode"] == "pair_type")
        & (cand["candidate_own_form"].isin(["acid_form", "base_form"]))
        & (cand["distribution_ok"])
    ].copy()
    if frame.empty:
        return frame

    if "candidate_family" not in frame.columns:
        frame["candidate_family"] = frame["candidate_label"].map(lambda x: CONJUGATE_FAMILY_MAP.get(str(x), str(x)))
    frame = frame.drop_duplicates(subset=["molecule_key", "candidate_instance_id", "pka_value"]).copy()
    if max_rows > 0 and len(frame) > max_rows:
        frame = frame.head(max_rows).copy()
    return frame


def _pick_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    best_thr = 0.5
    best_score = -1.0
    for thr in np.linspace(0.10, 0.90, 33):
        pred = (proba >= float(thr)).astype(int)
        pos_acc = float((pred[y_true == 1] == 1).mean()) if np.any(y_true == 1) else 0.0
        neg_acc = float((pred[y_true == 0] == 0).mean()) if np.any(y_true == 0) else 0.0
        balanced = 0.5 * (pos_acc + neg_acc)
        if balanced > best_score:
            best_score = balanced
            best_thr = float(thr)
    return best_thr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train strict pair-form heads for carboxyl and amine subtypes.")
    parser.add_argument("--assignments", default=DEFAULT_CURATED_ASSIGNMENTS_PATH)
    parser.add_argument("--summary", default=DEFAULT_CURATED_SUMMARY_PATH)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", default="data/processed/ml_models/pair_form_heads")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--min-rows", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    assignments_path, summary_path = ensure_curated_training_data(
        assignments_path=args.assignments,
        summary_path=getattr(args, "summary", DEFAULT_CURATED_SUMMARY_PATH),
        raw_dir=getattr(args, "raw_dir", DEFAULT_RAW_DIR),
    )
    base = _prepare_rows(assignments_path, summary_path, max_rows=int(args.max_rows))
    if base.empty:
        raise RuntimeError("No pair-type rows available for head training.")

    bundle: Dict[str, Dict] = {"heads": {}}
    metrics_rows: List[Dict] = []

    for head_name, spec in HEAD_SPECS.items():
        acid_labels = set(spec["acid_labels"])
        base_labels = set(spec["base_labels"])
        labels = acid_labels | base_labels

        frame = base[base["candidate_label"].isin(labels)].copy()
        if len(frame) < int(args.min_rows):
            metrics_rows.append(
                {
                    "head": head_name,
                    "rows": int(len(frame)),
                    "molecules": int(frame["molecule_key"].nunique()) if not frame.empty else 0,
                    "status": "skipped_low_rows",
                }
            )
            continue

        # Use candidate_own_form (derived from the candidate's OWN label)
        # instead of pair_member_form (which is inherited from the
        # measurement's selected group and can be wrong for multi-group
        # molecules where the candidate is not the selected group).
        frame["target_is_acid_form"] = (frame["candidate_own_form"] == "acid_form").astype(int)
        y = frame["target_is_acid_form"].to_numpy()
        if len(np.unique(y)) < 2:
            metrics_rows.append(
                {
                    "head": head_name,
                    "rows": int(len(frame)),
                    "molecules": int(frame["molecule_key"].nunique()),
                    "status": "skipped_single_class",
                }
            )
            continue

        X = build_carboxyl_form_feature_frame(
            frame,
            candidate_col="candidate_label",
            resolved_col="resolved_groups",
            smiles_col="smiles",
            pka_type_col="pka_type_canonical",
            include_pka_type=False,
        )

        train_idx, test_idx = _split_by_molecule(frame, test_frac=float(args.test_frac), seed=int(args.seed))
        if len(train_idx) == 0 or len(test_idx) == 0:
            metrics_rows.append(
                {
                    "head": head_name,
                    "rows": int(len(frame)),
                    "molecules": int(frame["molecule_key"].nunique()),
                    "status": "skipped_bad_split",
                }
            )
            continue

        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2:
            metrics_rows.append(
                {
                    "head": head_name,
                    "rows": int(len(frame)),
                    "molecules": int(frame["molecule_key"].nunique()),
                    "status": "skipped_single_class_train",
                }
            )
            continue
        if len(np.unique(y_test)) < 2:
            metrics_rows.append(
                {
                    "head": head_name,
                    "rows": int(len(frame)),
                    "molecules": int(frame["molecule_key"].nunique()),
                    "status": "skipped_single_class_test",
                }
            )
            continue

        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(X.iloc[train_idx], y_train)

        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        opt_thr = _pick_best_threshold(y_test, proba)
        pred = (proba >= opt_thr).astype(int)

        head_metrics = {
            "head": head_name,
            "rows": int(len(frame)),
            "molecules": int(frame["molecule_key"].nunique()),
            "test_rows": int(len(test_idx)),
            "acid_fraction": float(np.mean(y)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
            "decision_threshold": float(opt_thr),
            "decision_margin": float(max(0.05, min(0.20, 0.5 * abs(opt_thr - 0.5)))),
            "status": "trained",
        }
        metrics_rows.append(head_metrics)

        bundle["heads"][head_name] = {
            "model": model,
            "feature_columns": list(X.columns),
            "acid_labels": sorted(acid_labels),
            "base_labels": sorted(base_labels),
            "candidate_col": "candidate_label",
            "resolved_col": "resolved_groups",
            "smiles_col": "smiles",
            "pka_type_col": "pka_type_canonical",
            "include_pka_type": False,
            "decision_threshold": float(opt_thr),
            "decision_margin": float(max(0.05, min(0.20, 0.5 * abs(opt_thr - 0.5)))),
        }

    with open(os.path.join(args.out_dir, "pair_form_heads.pkl"), "wb") as handle:
        pickle.dump(bundle, handle)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_rows, handle, indent=2)

    print(f"Saved pair-form heads to {args.out_dir}")
    print(metrics_df.sort_values(["status", "head"]).to_string(index=False))


if __name__ == "__main__":
    main()
