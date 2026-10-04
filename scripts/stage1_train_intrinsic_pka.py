#!/usr/bin/env python3
"""Train canonical Stage 1 from clean, replicate-aware single-site networks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd  # type: ignore
from sklearn.base import clone  # type: ignore
from sklearn.ensemble import (  # type: ignore
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore
from sklearn.model_selection import GroupKFold, GroupShuffleSplit  # type: ignore

from stage1_canonical_data import (
    STAGE1_TRAINING_SCHEMA_VERSION,
    TARGET_DEFINITION,
    load_stage1_training_sites,
)
from functional_group_pka_analysis import ACIDIC_FAMILIES
from reference_residual_model import ReferenceResidualRegressor
from train_intrinsic_single_group_model import build_intrinsic_features


DEFAULT_NETWORK_DATASET = "data/processed/pka_molecule_microstate_network_dataset.csv"
DEFAULT_OUT_DIR = "data/processed/ml_models_experimental_only/stage1_intrinsic"
DEFAULT_REFERENCE_PANEL = "data/curation/stage1_reference_anchor_panel.csv"


def _sha256(path: str) -> str:
    """Hash an artifact so a fitted model can be tied to its exact inputs."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _detector_versions(training: pd.DataFrame) -> List[str]:
    """Collect the functional-group detector schemas represented in training rows."""
    versions = set()
    for raw in training.get(
        "functional_group_detector_schema_versions_json",
        pd.Series(dtype=str),
    ):
        try:
            parsed = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = [str(raw)]
        versions.update(str(value) for value in parsed if str(value))
    return sorted(versions)


def _model_candidates() -> Dict[str, object]:
    """Construct the fixed, reproducible set of Stage 1 regression candidates."""
    return {
        "reference_hist_absolute_leaf15": ReferenceResidualRegressor(
            HistGradientBoostingRegressor(
                loss="absolute_error", learning_rate=0.05, max_iter=450,
                max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                random_state=17,
            )
        ),
        "reference_hist_absolute_leaf31": ReferenceResidualRegressor(
            HistGradientBoostingRegressor(
                loss="absolute_error", learning_rate=0.04, max_iter=550,
                max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
                random_state=17,
            )
        ),
        "reference_extra_trees_leaf2": ReferenceResidualRegressor(
            ExtraTreesRegressor(
                n_estimators=350, min_samples_leaf=2, max_features=0.7,
                n_jobs=-1, random_state=17,
            )
        ),
        "reference_extra_trees_leaf5": ReferenceResidualRegressor(
            ExtraTreesRegressor(
                n_estimators=350, min_samples_leaf=5, max_features=0.7,
                n_jobs=-1, random_state=17,
            )
        ),
        "reference_random_forest_leaf2": ReferenceResidualRegressor(
            RandomForestRegressor(
                n_estimators=400, min_samples_leaf=2, max_features=0.7,
                n_jobs=-1, random_state=17,
            )
        ),
    }


def _metrics(y_true: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    """Summarize central and tail pKa errors, both weighted and unweighted."""
    errors = np.abs(y_true - predicted)
    return {
        "mae": float(mean_absolute_error(y_true, predicted)),
        "weighted_mae": float(mean_absolute_error(y_true, predicted, sample_weight=weights)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predicted))),
        "weighted_rmse": float(np.sqrt(mean_squared_error(y_true, predicted, sample_weight=weights))),
        "r2": float(r2_score(y_true, predicted)),
        "weighted_r2": float(r2_score(y_true, predicted, sample_weight=weights)),
        "median_abs_error": float(np.median(errors)),
        "p90_abs_error": float(np.quantile(errors, 0.90)),
        "p95_abs_error": float(np.quantile(errors, 0.95)),
        "max_abs_error": float(np.max(errors)),
        "count_abs_error_gt_2": int(np.sum(errors > 2.0)),
        "count_abs_error_gt_3": int(np.sum(errors > 3.0)),
        "count_abs_error_gt_5": int(np.sum(errors > 5.0)),
    }


def _fit(model: object, X: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> object:
    """Clone and fit an estimator without mutating its reusable template."""
    fitted = clone(model)
    fitted.fit(X, y, sample_weight=weights)
    return fitted


def select_model(
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    selection_splits: int,
    test_fraction: float,
) -> Tuple[str, object, pd.DataFrame]:
    """Select a model by scaffold-grouped validation and the one-SE rule.

    Among candidates statistically tied on mean MAE, the ordering favors the
    more regularized model before considering tail and weighted errors.
    """
    splitter = GroupShuffleSplit(
        n_splits=selection_splits,
        test_size=test_fraction,
        random_state=17,
    )
    splits = list(splitter.split(X, y, groups))
    records: List[Dict] = []
    candidates = _model_candidates()
    for name, template in candidates.items():
        for fold, (train_idx, test_idx) in enumerate(splits):
            model = _fit(template, X.iloc[train_idx], y[train_idx], weights[train_idx])
            predicted = model.predict(X.iloc[test_idx])
            fold_metrics = _metrics(y[test_idx], predicted, weights[test_idx])
            records.append({
                "candidate": name,
                "selection_fold": int(fold),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_scaffold_groups": int(len(set(groups[train_idx]))),
                "test_scaffold_groups": int(len(set(groups[test_idx]))),
                **fold_metrics,
            })
    detail = pd.DataFrame(records)
    summary = detail.groupby("candidate").agg(
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_weighted_mae=("weighted_mae", "mean"),
        mean_p95_abs_error=("p95_abs_error", "mean"),
    ).sort_values(["mean_mae", "mean_weighted_mae"])
    best_name = str(summary.index[0])
    best = summary.loc[best_name]
    standard_error = float(best["std_mae"]) / max(1.0, math.sqrt(float(selection_splits)))
    eligible = summary[summary["mean_mae"] <= float(best["mean_mae"]) + standard_error].copy()
    # One-standard-error selection: among statistically indistinguishable
    # candidates, prefer the more regularized model and then its tail error.
    simplicity = {
        "reference_hist_absolute_leaf15": 0,
        "reference_extra_trees_leaf5": 1,
        "reference_hist_absolute_leaf31": 2,
        "reference_random_forest_leaf2": 3,
        "reference_extra_trees_leaf2": 4,
    }
    eligible["simplicity_rank"] = [simplicity.get(str(name), 99) for name in eligible.index]
    selected_name = str(
        eligible.sort_values(
            ["simplicity_rank", "mean_p95_abs_error", "mean_weighted_mae"]
        ).index[0]
    )
    return selected_name, candidates[selected_name], detail


def scaffold_oof_predictions(
    template: object,
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    groups: np.ndarray,
    folds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate out-of-fold pKa predictions for held-out Murcko scaffolds."""
    unique_groups = len(set(groups))
    n_splits = min(int(folds), unique_groups)
    if n_splits < 2:
        raise ValueError("At least two scaffold groups are required for validation")
    predicted = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        model = _fit(template, X.iloc[train_idx], y[train_idx], weights[train_idx])
        predicted[test_idx] = model.predict(X.iloc[test_idx])
        fold_ids[test_idx] = fold
    if not np.isfinite(predicted).all() or np.any(fold_ids < 0):
        raise AssertionError("Scaffold OOF prediction coverage is incomplete")
    return predicted, fold_ids


def label_interpolation_oof_predictions(
    template: object,
    X: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    folds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """OOF predictions that retain same-label analogues whenever possible."""
    n_splits = max(2, min(int(folds), len(y)))
    fold_ids = np.full(len(y), -1, dtype=int)
    offsets: Dict[str, int] = {}
    for label in sorted(set(str(value) for value in labels)):
        indices = np.flatnonzero(labels.astype(str) == label)
        seed = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)
        offset = seed % n_splits
        offsets[label] = offset
        for position, index in enumerate(indices):
            fold_ids[index] = (offset + position) % n_splits
    predicted = np.full(len(y), np.nan, dtype=float)
    for fold in range(n_splits):
        test_idx = np.flatnonzero(fold_ids == fold)
        train_idx = np.flatnonzero(fold_ids != fold)
        if not len(test_idx) or not len(train_idx):
            continue
        model = _fit(template, X.iloc[train_idx], y[train_idx], weights[train_idx])
        predicted[test_idx] = model.predict(X.iloc[test_idx])
    if not np.isfinite(predicted).all() or np.any(fold_ids < 0):
        raise AssertionError("Label-interpolation OOF prediction coverage is incomplete")
    return predicted, fold_ids


def _add_fold_support(frame: pd.DataFrame, fold_column: str) -> pd.DataFrame:
    """Annotate each validation row with same-label support available in its fold."""
    result = frame.copy()
    counts = []
    scaffold_counts = []
    for row in result.itertuples(index=False):
        train = result[result[fold_column] != getattr(row, fold_column)]
        same_label = train[train["candidate_label"].astype(str) == str(row.candidate_label)]
        counts.append(int(len(same_label)))
        scaffold_counts.append(int(same_label["scaffold_group"].nunique()))
    result["exact_label_training_rows_in_fold"] = counts
    result["exact_label_training_scaffolds_in_fold"] = scaffold_counts
    result["validation_regime"] = np.where(
        result["exact_label_training_rows_in_fold"] > 0,
        "label_supported",
        "zero_shot_exact_label_absent",
    )
    return result


def _error_calibration(frame: pd.DataFrame) -> Tuple[Dict, Dict[str, Dict], Dict[str, Dict]]:
    """Aggregate absolute-error calibration globally and by family and site label."""
    def summarize(part: pd.DataFrame) -> Dict:
        """Reduce one calibration stratum to coverage and error quantiles."""
        error = part["abs_error"].to_numpy(dtype=float)
        return {
            "rows": int(len(part)),
            "mae": float(np.mean(error)),
            "median_abs_error": float(np.median(error)),
            "p90_abs_error": float(np.quantile(error, 0.90)),
            "p95_abs_error": float(np.quantile(error, 0.95)),
        }

    global_calibration = summarize(frame)
    by_family = {
        str(family): summarize(part)
        for family, part in frame.groupby("site_family")
    }
    by_label = {
        str(label): summarize(part)
        for label, part in frame.groupby("candidate_label")
    }
    return global_calibration, by_family, by_label


def train_stage1(
    network_dataset_path: str,
    out_dir: str,
    fp_bits: int = 512,
    fp_radius: int = 2,
    max_replicate_range: float = 2.0,
    min_supported_pka: float = -5.0,
    max_supported_pka: float = 20.0,
    selection_splits: int = 3,
    scaffold_folds: int = 5,
    test_fraction: float = 0.2,
) -> Tuple[Dict, pd.DataFrame]:
    """Build, validate, train, and persist the canonical intrinsic-site pKa model.

    Model selection and both OOF regimes are diagnostic.  The deployable model
    is refit on every eligible experimental Stage 1 row before serialization.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    training, quarantine, funnel = load_stage1_training_sites(
        network_dataset_path,
        max_replicate_range=max_replicate_range,
        min_supported_pka=min_supported_pka,
        max_supported_pka=max_supported_pka,
    )
    if training.empty:
        raise RuntimeError("No eligible canonical single-site Stage 1 rows")
    if training[["molecule_id", "site_id"]].duplicated().any():
        raise ValueError("Canonical Stage 1 data contains duplicate molecule/site rows")
    if set(training["measurement_method"]) != {"experimental"}:
        raise ValueError("Canonical Stage 1 data is not experimental-only")

    training_path = os.path.join(out_dir, "stage1_training_sites.csv")
    quarantine_path = os.path.join(out_dir, "stage1_training_site_quarantine.csv")
    training.to_csv(training_path, index=False)
    quarantine.to_csv(quarantine_path, index=False)

    features = build_intrinsic_features(
        training,
        nbits=fp_bits,
        radius=fp_radius,
        label_col="candidate_label",
    )
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite canonical Stage 1 features")
    y = training["pka_value"].to_numpy(dtype=float)
    weights = training["sample_weight"].to_numpy(dtype=float)
    groups = training["scaffold_group"].astype(str).to_numpy()

    selected_name, template, selection_detail = select_model(
        features, y, weights, groups,
        selection_splits=selection_splits,
        test_fraction=test_fraction,
    )
    oof_prediction, fold_ids = scaffold_oof_predictions(
        template, features, y, weights, groups, folds=scaffold_folds
    )
    oof = training.copy()
    oof["scaffold_fold"] = fold_ids
    oof["pred_intrinsic_pka"] = oof_prediction
    oof["abs_error"] = np.abs(oof["pka_value"] - oof["pred_intrinsic_pka"])
    oof = _add_fold_support(oof, "scaffold_fold")
    scaffold_metrics = _metrics(y, oof_prediction, weights)

    interpolation_prediction, interpolation_fold_ids = label_interpolation_oof_predictions(
        template,
        features,
        y,
        weights,
        training["candidate_label"].astype(str).to_numpy(),
        folds=scaffold_folds,
    )
    interpolation_oof = training.copy()
    interpolation_oof["interpolation_fold"] = interpolation_fold_ids
    interpolation_oof["pred_intrinsic_pka"] = interpolation_prediction
    interpolation_oof["abs_error"] = np.abs(
        interpolation_oof["pka_value"] - interpolation_oof["pred_intrinsic_pka"]
    )
    interpolation_oof = _add_fold_support(interpolation_oof, "interpolation_fold")
    interpolation_metrics = _metrics(y, interpolation_prediction, weights)
    global_calibration, family_calibration, _ = _error_calibration(oof)
    _, _, label_calibration = _error_calibration(interpolation_oof)

    deployment_model = _fit(template, features, y, weights)
    training_reconstruction = training.copy()
    training_reconstruction["pred_intrinsic_pka"] = deployment_model.predict(features)
    training_reconstruction["abs_error"] = np.abs(
        training_reconstruction["pka_value"]
        - training_reconstruction["pred_intrinsic_pka"]
    )
    reconstruction_metrics = _metrics(
        y,
        training_reconstruction["pred_intrinsic_pka"].to_numpy(dtype=float),
        weights,
    )
    reference_panel = pd.DataFrame()
    reference_panel_metrics: Dict[str, object] = {
        "status": "unavailable",
        "interpretation": "project calibration panel; not an independent external test",
    }
    if os.path.exists(DEFAULT_REFERENCE_PANEL):
        reference_panel = pd.read_csv(DEFAULT_REFERENCE_PANEL, low_memory=False)
        reference_panel["candidate_label"] = reference_panel["site_label"].astype(str)
        reference_panel["final_group_label"] = reference_panel["site_label"].astype(str)
        reference_panel["pka_type_canonical"] = np.where(
            reference_panel["site_family"].astype(str).isin(ACIDIC_FAMILIES),
            "acidic",
            "basic",
        )
        reference_panel["atom_index_raw"] = np.nan
        with np.errstate(all="ignore"):
            panel_features = build_intrinsic_features(
                reference_panel,
                nbits=fp_bits,
                radius=fp_radius,
                label_col="candidate_label",
            ).reindex(columns=features.columns, fill_value=0.0)
        reference_panel["pred_intrinsic_pka"] = deployment_model.predict(panel_features)
        reference_panel["abs_error"] = np.abs(
            reference_panel["reference_pka"] - reference_panel["pred_intrinsic_pka"]
        )
        reference_panel_metrics = {
            "status": "evaluated",
            "interpretation": "project calibration panel; not an independent external test",
            **_metrics(
                reference_panel["reference_pka"].to_numpy(dtype=float),
                reference_panel["pred_intrinsic_pka"].to_numpy(dtype=float),
                np.ones(len(reference_panel), dtype=float),
            ),
        }
    detector_versions = _detector_versions(training)
    bundle = {
        "model": deployment_model,
        "feature_columns": list(features.columns),
        "fp_bits": int(fp_bits),
        "fp_radius": int(fp_radius),
        "stage1_schema_version": STAGE1_TRAINING_SCHEMA_VERSION,
        "target_definition": TARGET_DEFINITION,
        "target_aggregation": "median",
        "training_measurement_method": "experimental",
        "training_rows": int(len(training)),
        "deployment_fit_rows": int(len(training)),
        "trained_on_all_eligible_rows": True,
        "training_measurement_count": int(training["experimental_measurement_count"].sum()),
        "training_molecules": int(training["molecule_id"].nunique()),
        "training_structures": int(training["structure_key"].nunique()),
        "training_scaffold_groups": int(training["scaffold_group"].nunique()),
        "training_site_family_counts": {
            str(key): int(value) for key, value in training["site_family"].value_counts().items()
        },
        "training_site_label_counts": {
            str(key): int(value)
            for key, value in training["candidate_label"].value_counts().items()
        },
        "training_smiles_by_label": {
            str(label): sorted(set(part["smiles"].astype(str)))
            for label, part in training.groupby("candidate_label")
        },
        "training_dataset_sha256": _sha256(training_path),
        "reference_anchor_panel_sha256": (
            _sha256(DEFAULT_REFERENCE_PANEL) if os.path.exists(DEFAULT_REFERENCE_PANEL) else None
        ),
        "source_network_dataset_sha256": _sha256(network_dataset_path),
        "functional_group_detector_schema_versions": detector_versions,
        "max_replicate_range": float(max_replicate_range),
        "supported_pka_range": [float(min_supported_pka), float(max_supported_pka)],
        "sample_weight_definition": "replicate_count_x_robust_dispersion_x_mapping_confidence; normalized_mean_1",
        "selected_model": selected_name,
        "selected_model_class": type(deployment_model).__name__,
        "selected_model_params": deployment_model.get_params(),
        "validation_method": f"{scaffold_folds}-fold_grouped_Murcko_scaffold_OOF; acyclic_by_canonical_structure",
        "scaffold_validation_metrics": scaffold_metrics,
        "label_interpolation_validation_metrics": interpolation_metrics,
        "training_reconstruction_metrics": reconstruction_metrics,
        "reference_anchor_panel_metrics": reference_panel_metrics,
        "error_calibration_global": global_calibration,
        "error_calibration_by_family": family_calibration,
        "error_calibration_by_label": label_calibration,
    }
    model_path = os.path.join(out_dir, "stage1_intrinsic_model.pkl")
    with open(model_path, "wb") as handle:
        pickle.dump(bundle, handle)

    selection_detail.to_csv(os.path.join(out_dir, "model_selection_fold_metrics.csv"), index=False)
    selection_summary = selection_detail.groupby("candidate").agg(
        folds=("selection_fold", "count"),
        mean_mae=("mae", "mean"),
        std_mae=("mae", "std"),
        mean_weighted_mae=("weighted_mae", "mean"),
        mean_rmse=("rmse", "mean"),
    ).sort_values("mean_mae").reset_index()
    selection_summary.to_csv(os.path.join(out_dir, "model_selection_summary.csv"), index=False)
    oof.sort_values("abs_error", ascending=False).to_csv(
        os.path.join(out_dir, "scaffold_oof_predictions.csv"), index=False
    )
    interpolation_oof.sort_values("abs_error", ascending=False).to_csv(
        os.path.join(out_dir, "label_interpolation_oof_predictions.csv"), index=False
    )
    training_reconstruction.sort_values("abs_error", ascending=False).to_csv(
        os.path.join(out_dir, "training_reconstruction_predictions.csv"), index=False
    )
    if not reference_panel.empty:
        reference_panel.sort_values("abs_error", ascending=False).to_csv(
            os.path.join(out_dir, "reference_anchor_panel_predictions.csv"), index=False
        )
    oof[["molecule_id", "site_id", "structure_key", "scaffold_group", "scaffold_fold"]].to_csv(
        os.path.join(out_dir, "scaffold_split_manifest.csv"), index=False
    )

    report = {
        "stage1_schema_version": STAGE1_TRAINING_SCHEMA_VERSION,
        "target_definition": TARGET_DEFINITION,
        "measurement_method": "experimental",
        "rows": int(len(training)),
        "molecules": int(training["molecule_id"].nunique()),
        "structures": int(training["structure_key"].nunique()),
        "experimental_measurements": int(training["experimental_measurement_count"].sum()),
        "features": int(features.shape[1]),
        "funnel": funnel,
        "max_replicate_range": float(max_replicate_range),
        "supported_pka_range": [float(min_supported_pka), float(max_supported_pka)],
        "selected_model": selected_name,
        "model_selection": selection_summary.to_dict(orient="records"),
        "scaffold": scaffold_metrics,
        "label_interpolation": interpolation_metrics,
        "training_reconstruction": reconstruction_metrics,
        "reference_anchor_panel": reference_panel_metrics,
        "model_selection_policy": (
            "one_standard_error_then_prefer_regularized_reference_residual_model"
        ),
        "validation_interpretation": {
            "training_reconstruction": "fit check only; not a generalization estimate",
            "label_interpolation": "held-out molecules with same-label examples retained whenever count permits",
            "scaffold": "unseen Murcko scaffold; rows may be zero-shot for an exact site label",
        },
        "error_calibration_global": global_calibration,
        "error_calibration_by_family": family_calibration,
        "deployment_fit_rows": int(len(training)),
        "trained_on_all_eligible_rows": True,
        "training_dataset_sha256": _sha256(training_path),
        "functional_group_detector_schema_versions": detector_versions,
        "model_sha256": _sha256(model_path),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report, oof


def parse_args() -> argparse.Namespace:
    """Parse command-line paths and reproducible Stage 1 training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-dataset", default=DEFAULT_NETWORK_DATASET)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--fp-bits", type=int, default=512)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--max-replicate-range", type=float, default=2.0)
    parser.add_argument("--min-supported-pka", type=float, default=-5.0)
    parser.add_argument("--max-supported-pka", type=float, default=20.0)
    parser.add_argument("--selection-splits", type=int, default=3)
    parser.add_argument("--scaffold-folds", type=int, default=5)
    parser.add_argument("--test-frac", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    """Run Stage 1 training from the CLI and print its machine-readable report."""
    args = parse_args()
    report, _ = train_stage1(
        network_dataset_path=args.network_dataset,
        out_dir=args.out_dir,
        fp_bits=args.fp_bits,
        fp_radius=args.fp_radius,
        max_replicate_range=args.max_replicate_range,
        min_supported_pka=args.min_supported_pka,
        max_supported_pka=args.max_supported_pka,
        selection_splits=args.selection_splits,
        scaffold_folds=args.scaffold_folds,
        test_fraction=args.test_frac,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
