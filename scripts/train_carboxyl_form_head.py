import argparse
import json
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score  # type: ignore

from ionization_context_features import build_carboxyl_form_feature_frame
from training_data_resolver import (
    DEFAULT_CURATED_ASSIGNMENTS_PATH,
    DEFAULT_RAW_DIR,
    ensure_curated_training_data,
)


def _molecule_key(row: pd.Series) -> str:
    return f"{row['source_file']}::{int(row['record_index'])}::{row['smiles']}"


def _prepare_training_rows(assignments_path: str) -> pd.DataFrame:
    df = pd.read_csv(assignments_path)
    required = {
        "source_file",
        "record_index",
        "smiles",
        "final_group_family",
        "pair_member_form",
        "group_mode",
        "resolved_groups",
        "resolved_group_count",
        "final_group_label",
        "distribution_ok",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    pair = df[
        (df["group_mode"] == "pair_type")
        & (df["final_group_family"] == "carboxyl")
        & (df["pair_member_form"].isin(["acid_form", "base_form"]))
        & (df["distribution_ok"]) 
    ].copy()

    if pair.empty:
        return pair

    pair["molecule_key"] = pair.apply(_molecule_key, axis=1)
    pair = pair.drop_duplicates(subset=["molecule_key", "pka_value", "final_group_label"]).copy()
    pair["candidate_label"] = pair["final_group_label"]
    pair["target_is_acid_form"] = (pair["pair_member_form"] == "acid_form").astype(int)
    return pair


def _split_by_molecule(df: pd.DataFrame, test_frac: float, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mols = np.array(sorted(df["molecule_key"].unique()))
    rng.shuffle(mols)
    n_test = max(1, int(len(mols) * test_frac))
    test_mols = set(mols[:n_test])
    test_mask = df["molecule_key"].isin(test_mols).to_numpy()
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    return {"train": train_idx, "test": test_idx}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dedicated carboxyl acid/base form head.")
    parser.add_argument("--assignments", default=DEFAULT_CURATED_ASSIGNMENTS_PATH)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", default="data/processed/ml_models/carboxyl_form_head")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    assignments_path, _ = ensure_curated_training_data(
        assignments_path=args.assignments,
        raw_dir=getattr(args, "raw_dir", DEFAULT_RAW_DIR),
    )
    frame = _prepare_training_rows(assignments_path)
    if args.max_rows > 0 and len(frame) > args.max_rows:
        frame = frame.head(int(args.max_rows)).copy()
    if frame.empty:
        raise RuntimeError("No carboxyl pair-type rows available for training.")

    X = build_carboxyl_form_feature_frame(
        frame,
        candidate_col="candidate_label",
        resolved_col="resolved_groups",
        smiles_col="smiles",
        pka_type_col="pka_type_canonical",
    )
    y = frame["target_is_acid_form"].astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        metrics = {
            "rows": int(len(frame)),
            "molecules": int(frame["molecule_key"].nunique()),
            "acid_fraction": float(np.mean(y)) if len(y) else float("nan"),
            "status": "skipped_single_class",
        }
        with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Skipped carboxyl form head training (single class) in {args.out_dir}")
        print(json.dumps(metrics, indent=2))
        return

    split = _split_by_molecule(frame, test_frac=float(args.test_frac), seed=int(args.seed))
    train_idx, test_idx = split["train"], split["test"]

    model = LogisticRegression(max_iter=1500, class_weight="balanced")
    model.fit(X.iloc[train_idx], y[train_idx])

    proba = model.predict_proba(X.iloc[test_idx])[:, 1]
    pred = (proba >= 0.5).astype(int)
    y_test = y[test_idx]

    metrics = {
        "rows": int(len(frame)),
        "molecules": int(frame["molecule_key"].nunique()),
        "test_rows": int(len(test_idx)),
        "acid_fraction": float(np.mean(y)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
    }

    bundle = {
        "model": model,
        "feature_columns": list(X.columns),
        "candidate_col": "candidate_label",
        "resolved_col": "resolved_groups",
        "smiles_col": "smiles",
        "pka_type_col": "pka_type_canonical",
    }

    with open(os.path.join(args.out_dir, "carboxyl_form_head.pkl"), "wb") as handle:
        pickle.dump(bundle, handle)

    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved carboxyl form head to {args.out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
