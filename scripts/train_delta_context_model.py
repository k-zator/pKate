"""LEGACY candidate-table delta model.

Use stage2_train_network_context.py for the canonical molecule-network Stage 2.
"""

import argparse
import json
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore

from ionization_context_features import build_ionization_context_frame
from functional_group_pka_analysis import training_group_label
from ml_features import molecule_descriptors, morgan_bits
from ml_splits import random_molecule_split, scaffold_molecule_split, split_manifest
from train_intrinsic_single_group_model import build_intrinsic_features, _local_site_features
from training_data_resolver import (
    DEFAULT_CURATED_ASSIGNMENTS_PATH,
    DEFAULT_RAW_DIR,
    ensure_curated_training_data,
)


def _molecule_key(row: pd.Series) -> str:
    return f"{row['source_file']}::{int(row['record_index'])}::{row['smiles']}"


def _load_stage1_bundle(path: str) -> Dict:
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    required = {"model", "feature_columns", "fp_bits", "fp_radius"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(f"Invalid Stage 1 bundle; missing keys: {sorted(missing)}")
    return bundle


def _build_stage1_like_features(df: pd.DataFrame, feature_columns, fp_bits: int, fp_radius: int) -> pd.DataFrame:
    X = build_intrinsic_features(df, nbits=fp_bits, radius=fp_radius)
    X = X.reindex(columns=feature_columns, fill_value=0.0)
    return X


def load_multi_group_rows(assignments_path: str) -> pd.DataFrame:
    df = pd.read_csv(assignments_path)
    required = {
        "source_file",
        "record_index",
        "smiles",
        "pka_value",
        "final_group_label",
        "assignment_status",
        "resolved_groups",
        "resolved_group_count",
        "pka_type_canonical",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    multi = df[df["assignment_status"] == "multi_group"].copy()
    multi = multi.dropna(subset=["smiles", "pka_value", "final_group_label", "resolved_groups"])

    # Filter out outlier pKa values flagged by distribution sanity checks.
    if "distribution_ok" in multi.columns:
        n_before = len(multi)
        multi = multi[multi["distribution_ok"].astype(bool)].copy()
        n_removed = n_before - len(multi)
        if n_removed > 0:
            print(f"  distribution_ok filter removed {n_removed}/{n_before} outlier rows")

    multi["molecule_key"] = multi.apply(_molecule_key, axis=1)
    multi = multi.drop_duplicates(subset=["molecule_key", "pka_value", "final_group_label"])
    multi.reset_index(drop=True, inplace=True)
    return multi


def _resolved_group_multihot(series: pd.Series) -> pd.DataFrame:
    labels = sorted({token for value in series for token in str(value).split("|") if token})
    if not labels:
        return pd.DataFrame(index=series.index)
    data = []
    for value in series:
        tokens = set(token for token in str(value).split("|") if token)
        data.append({f"ctx_has_{label}": float(label in tokens) for label in labels})
    return pd.DataFrame(data)


def build_delta_features(df: pd.DataFrame, nbits: int, radius: int) -> pd.DataFrame:
    smiles_unique = df["smiles"].dropna().unique()
    smiles_to_desc = {smi: molecule_descriptors(smi) for smi in smiles_unique}
    smiles_to_fp = {smi: morgan_bits(smi, nbits=nbits, radius=radius) for smi in smiles_unique}

    desc_df = pd.DataFrame(df["smiles"].map(smiles_to_desc).tolist())
    fp_matrix = np.vstack([smiles_to_fp.get(smi, np.zeros(nbits, dtype=np.float32)) for smi in df["smiles"]])
    fp_df = pd.DataFrame(fp_matrix, columns=[f"fp_{i}" for i in range(nbits)])

    label_col = "selected_group_label" if "selected_group_label" in df.columns else "final_group_label"
    site_feature_df, _ = _local_site_features(df, nbits=nbits, radius=radius, label_col=label_col)
    site_geom_cols = [
        col
        for col in site_feature_df.columns
        if col.startswith("mol_geom_") or col.startswith("site_geom_")
    ]
    site_geom_df = site_feature_df[site_geom_cols].copy()

    training_labels = df[label_col].astype(str).map(training_group_label)
    group_df = pd.get_dummies(training_labels, prefix="group")
    type_df = pd.get_dummies(df["pka_type_canonical"].fillna("unknown"), prefix="pka_type")
    mode_df = pd.get_dummies(df.get("group_mode", "unknown").fillna("unknown"), prefix="mode")
    form_df = pd.get_dummies(df.get("candidate_own_form", df.get("pair_member_form", "unknown")).fillna("unknown"), prefix="pair_form")
    resolved_df = _resolved_group_multihot(df["resolved_groups"]) 
    ion_ctx_df = build_ionization_context_frame(
        df,
        resolved_col="resolved_groups",
        candidate_col=label_col,
    )

    numeric_cols = ["resolved_group_count", "intrinsic_pred_pka"]
    if "molecule_formal_charge" in df.columns:
        numeric_cols.append("molecule_formal_charge")
    if "neutral_input_risk" in df.columns:
        numeric_cols.append("neutral_input_risk")

    numeric_df = df[numeric_cols].copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce").fillna(0.0)

    X = pd.concat(
        [numeric_df, ion_ctx_df, site_geom_df, desc_df, group_df, type_df, mode_df, form_df, resolved_df, fp_df],
        axis=1,
    )
    return X


def _eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _fit_and_eval(
    frame: pd.DataFrame,
    X: pd.DataFrame,
    y_delta: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[HistGradientBoostingRegressor, Dict[str, float], pd.DataFrame]:
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        random_state=17,
    )
    model.fit(X.iloc[train_idx], y_delta.iloc[train_idx])

    pred_delta = model.predict(X.iloc[test_idx])
    eval_df = frame.iloc[test_idx].copy()
    eval_df["pred_delta_pka"] = pred_delta
    eval_df["pred_effective_pka"] = eval_df["intrinsic_pred_pka"] + eval_df["pred_delta_pka"]
    eval_df["pred_intrinsic_only_pka"] = eval_df["intrinsic_pred_pka"]

    effective_metrics = _eval_regression(eval_df["pka_value"].to_numpy(), eval_df["pred_effective_pka"].to_numpy())
    intrinsic_metrics = _eval_regression(eval_df["pka_value"].to_numpy(), eval_df["pred_intrinsic_only_pka"].to_numpy())

    metrics = {
        "effective_mae": effective_metrics["mae"],
        "effective_rmse": effective_metrics["rmse"],
        "effective_r2": effective_metrics["r2"],
        "intrinsic_only_mae": intrinsic_metrics["mae"],
        "intrinsic_only_rmse": intrinsic_metrics["rmse"],
        "intrinsic_only_r2": intrinsic_metrics["r2"],
        "delta_target_std": float(np.std(y_delta.iloc[test_idx].to_numpy())),
    }
    return model, metrics, eval_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: train delta-pKa context model for multi-group molecules.")
    parser.add_argument("--assignments", default=DEFAULT_CURATED_ASSIGNMENTS_PATH)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--stage1-model", default="data/processed/ml_models/stage1_intrinsic/stage1_intrinsic_model.pkl")
    parser.add_argument("--out-dir", default="data/processed/ml_models/stage2_delta")
    parser.add_argument("--fp-bits", type=int, default=512)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    bundle = _load_stage1_bundle(args.stage1_model)
    stage1_model = bundle["model"]
    stage1_cols = bundle["feature_columns"]
    stage1_fp_bits = int(bundle["fp_bits"])
    stage1_fp_radius = int(bundle["fp_radius"])

    assignments_path, _ = ensure_curated_training_data(
        assignments_path=args.assignments,
        raw_dir=getattr(args, "raw_dir", DEFAULT_RAW_DIR),
    )
    multi_df = load_multi_group_rows(assignments_path)
    if multi_df.empty:
        raise RuntimeError("No multi-group rows available for Stage 2 training.")
    if args.max_rows > 0:
        multi_df = multi_df.head(int(args.max_rows)).copy()

    X_stage1 = _build_stage1_like_features(
        multi_df,
        feature_columns=stage1_cols,
        fp_bits=stage1_fp_bits,
        fp_radius=stage1_fp_radius,
    )
    multi_df["intrinsic_pred_pka"] = stage1_model.predict(X_stage1)
    multi_df["delta_pka"] = multi_df["pka_value"].astype(float) - multi_df["intrinsic_pred_pka"].astype(float)

    X_delta = build_delta_features(multi_df, nbits=args.fp_bits, radius=args.fp_radius)
    y_delta = multi_df["delta_pka"].astype(float)

    r_train, r_test = random_molecule_split(multi_df, test_frac=args.test_frac)
    s_train, s_test = scaffold_molecule_split(multi_df, test_frac=args.test_frac)

    random_model, random_metrics, random_eval = _fit_and_eval(multi_df, X_delta, y_delta, r_train, r_test)
    scaffold_model, scaffold_metrics, scaffold_eval = _fit_and_eval(multi_df, X_delta, y_delta, s_train, s_test)

    bundle_out = {
        "model": scaffold_model,
        "feature_columns": list(X_delta.columns),
        "fp_bits": int(args.fp_bits),
        "fp_radius": int(args.fp_radius),
    }
    with open(os.path.join(args.out_dir, "stage2_delta_model.pkl"), "wb") as handle:
        pickle.dump(bundle_out, handle)

    random_eval.to_csv(os.path.join(args.out_dir, "random_eval_delta.csv"), index=False)
    scaffold_eval.to_csv(os.path.join(args.out_dir, "scaffold_eval_delta.csv"), index=False)

    split_manifest(multi_df, r_train, r_test).to_csv(
        os.path.join(args.out_dir, "random_split_manifest.csv"),
        index=False,
    )
    split_manifest(multi_df, s_train, s_test).to_csv(
        os.path.join(args.out_dir, "scaffold_split_manifest.csv"),
        index=False,
    )

    report = {
        "rows": int(len(multi_df)),
        "molecules": int(multi_df["molecule_key"].nunique()),
        "delta_mean": float(multi_df["delta_pka"].mean()),
        "delta_std": float(multi_df["delta_pka"].std()),
        "random": random_metrics,
        "scaffold": scaffold_metrics,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Saved Stage 2 artifacts to", args.out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
