#!/usr/bin/env python3
"""Train canonical Stage 2 on residuals from the blind Stage-1 network baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd  # type: ignore
from sklearn.base import clone  # type: ignore
from sklearn.ensemble import (  # type: ignore
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore
from sklearn.model_selection import GroupShuffleSplit  # type: ignore

from ml_splits import random_molecule_split, scaffold_molecule_split
from stage1_canonical_data import scaffold_group_key
from stage2_network_context import (
    STAGE2_TARGET_DEFINITION,
    STAGE2_TRAINING_SCHEMA_VERSION,
    build_stage2_feature_matrix,
    build_stage2_site_table,
    build_stage2_weak_training_sites,
    detector_schema_versions,
    prepare_stage2_training_sites,
    project_macro_pkas_nonincreasing_with_fixed,
    validate_stage2_site_table,
)
from stage2_free_energy_coupling import (
    STAGE2_FREE_ENERGY_METHOD,
    STAGE2_FREE_ENERGY_SCHEMA_VERSION,
    infer_regularized_pairwise_free_energy,
    macro_pka_values_from_free_energy,
    pair_coupling_map,
)


DEFAULT_NETWORK_DATASET = "data/processed/pka_molecule_microstate_network_dataset.csv"
DEFAULT_OUT_DIR = "data/processed/ml_models_experimental_only/stage2_network_context"
MIN_FAMILY_TRAINING_ROWS = 50
PRIORITY_SUPPORTED_FAMILIES = frozenset({"guanidine_like", "phenol_phenolate"})


def supported_families_for_counts(
    family_counts: Dict[str, int],
    minimum_family_training_rows: int = MIN_FAMILY_TRAINING_ROWS,
) -> set[str]:
    """Apply the general evidence threshold plus explicit chemistry priorities."""
    return {
        str(family)
        for family, count in family_counts.items()
        if int(count) > 0
        and (
            int(count) >= int(minimum_family_training_rows)
            or str(family) in PRIORITY_SUPPORTED_FAMILIES
        )
    }


def _sha256(path: str) -> str:
    """Hash an artifact so Stage 2 remains bound to its exact inputs."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_candidates() -> Dict[str, object]:
    """Construct the fixed, reproducible set of context-residual regressors."""
    return {
        "hist_squared_leaf15": HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.04, max_iter=450,
            max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
            random_state=17,
        ),
        "hist_absolute_leaf15": HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.04, max_iter=500,
            max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
            random_state=17,
        ),
        "hist_absolute_leaf31": HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.035, max_iter=550,
            max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
            random_state=17,
        ),
        "extra_trees_leaf2": ExtraTreesRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=0.7,
            n_jobs=-1, random_state=17,
        ),
        "extra_trees_leaf5": ExtraTreesRegressor(
            n_estimators=350, min_samples_leaf=5, max_features=0.7,
            n_jobs=-1, random_state=17,
        ),
        "random_forest_leaf2": RandomForestRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=0.7,
            n_jobs=-1, random_state=17,
        ),
    }


def _metrics(
    y_true: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> Dict[str, float]:
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


def _add_pairwise_free_energy_predictions(full_evaluation: pd.DataFrame) -> pd.DataFrame:
    """Apply the deployment free-energy closure inside a validation fold."""
    work = full_evaluation.copy()
    work["stage2_supported_pairwise_free_energy_macro_pka"] = np.nan
    work["stage2_pairwise_free_energy_reconstruction_abs_gap"] = np.nan
    for molecule_key, indices in work.groupby("molecule_key").groups.items():
        part = work.loc[list(indices)].copy()
        ordered = part.sort_values("stage1_macro_step")
        sites = json.loads(str(part.iloc[0]["network_sites_json"]))
        nodes = json.loads(str(part.iloc[0]["microstate_nodes_json"]))
        row_lookup = {str(row.site_id): row for row in part.itertuples(index=False)}
        site_ids = [str(site["site_id"]) for site in sites]
        if set(site_ids) != set(row_lookup):
            raise ValueError(
                f"Validation molecule {molecule_key} does not contain every network site"
            )
        target_macro = ordered[
            "stage2_supported_projected_macro_pka"
        ].to_numpy(dtype=float).tolist()
        one_body, pair_terms, _ = infer_regularized_pairwise_free_energy(
            stage1_local_pkas=[
                float(row_lookup[site_id].stage1_intrinsic_pka) for site_id in site_ids
            ],
            stage2_macro_pkas=target_macro,
            adjustable=[
                bool(row_lookup[site_id].stage2_family_supported) for site_id in site_ids
            ],
            proposed_deltas=[
                float(row_lookup[site_id].stage2_supported_delta) for site_id in site_ids
            ],
            nodes=nodes,
            sites=sites,
            observed_macro_anchor_count=int(part["experimental_anchor_pka"].notna().sum()),
        )
        reconstructed = macro_pka_values_from_free_energy(
            nodes, sites, one_body, pair_coupling_map(pair_terms)
        )
        if len(reconstructed) != len(ordered):
            raise ValueError(
                f"Validation molecule {molecule_key} free-energy ladder has wrong length"
            )
        for row_index, value, target in zip(ordered.index, reconstructed, target_macro):
            work.loc[row_index, "stage2_supported_pairwise_free_energy_macro_pka"] = value
            work.loc[
                row_index, "stage2_pairwise_free_energy_reconstruction_abs_gap"
            ] = abs(float(value) - float(target))
    return work


def _fit_evaluate(
    model_template: object,
    training_df: pd.DataFrame,
    full_site_df: pd.DataFrame,
    full_features: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    minimum_family_training_rows: int,
) -> Tuple[Dict, pd.DataFrame, object]:
    """Fit one fold and evaluate the complete supported free-energy pipeline.

    Predictions are made for every site in each held-out molecule, family
    support gates are learned from the training fold only, and the projected
    macro ladder is closed into the same pairwise model used at deployment.
    """
    model = clone(model_template)
    target = training_df["stage2_delta_target"].astype(float).to_numpy()
    weights = training_df["sample_weight"].astype(float).to_numpy()
    train_positions = training_df.iloc[train_idx]["full_site_position"].to_numpy(dtype=int)
    model.fit(
        full_features.iloc[train_positions],
        target[train_idx],
        sample_weight=weights[train_idx],
    )
    train_family_counts = training_df.iloc[train_idx]["site_family"].value_counts()
    supported_families = supported_families_for_counts(
        {str(family): int(count) for family, count in train_family_counts.items()},
        minimum_family_training_rows=minimum_family_training_rows,
    )
    test_molecules = set(training_df.iloc[test_idx]["molecule_key"].astype(str))
    full_evaluation = full_site_df[
        full_site_df["molecule_key"].astype(str).isin(test_molecules)
    ].copy()
    full_positions = full_evaluation["full_site_position"].to_numpy(dtype=int)
    delta = model.predict(full_features.iloc[full_positions])
    full_evaluation["stage2_predicted_delta"] = delta
    full_evaluation["stage2_predicted_macro_pka"] = (
        full_evaluation["stage1_network_macro_pka"] + delta
    )
    full_evaluation["stage2_family_supported"] = full_evaluation["site_family"].isin(
        supported_families
    )
    full_evaluation["stage2_supported_delta"] = np.where(
        full_evaluation["stage2_family_supported"], delta, 0.0
    )
    full_evaluation["stage2_supported_macro_pka"] = (
        full_evaluation["stage1_network_macro_pka"]
        + full_evaluation["stage2_supported_delta"]
    )
    full_evaluation["stage2_projected_macro_pka"] = full_evaluation["stage2_predicted_macro_pka"]
    full_evaluation["stage2_supported_projected_macro_pka"] = full_evaluation["stage2_supported_macro_pka"]
    for _, indices in full_evaluation.groupby("molecule_key").groups.items():
        ordered_indices = full_evaluation.loc[list(indices)].sort_values("stage1_macro_step").index
        raw_values = full_evaluation.loc[ordered_indices, "stage2_predicted_macro_pka"].tolist()
        full_evaluation.loc[ordered_indices, "stage2_projected_macro_pka"] = (
            project_macro_pkas_nonincreasing_with_fixed(
                raw_values,
                [True] * len(raw_values),
            )
        )
        supported_values = full_evaluation.loc[
            ordered_indices, "stage2_supported_macro_pka"
        ].tolist()
        adjustable = full_evaluation.loc[
            ordered_indices, "stage2_family_supported"
        ].astype(bool).tolist()
        full_evaluation.loc[ordered_indices, "stage2_supported_projected_macro_pka"] = (
            project_macro_pkas_nonincreasing_with_fixed(supported_values, adjustable)
        )

    full_evaluation = _add_pairwise_free_energy_predictions(full_evaluation)

    test_keys = set(
        zip(
            training_df.iloc[test_idx]["molecule_key"].astype(str),
            training_df.iloc[test_idx]["site_id"].astype(str),
        )
    )
    evaluation = full_evaluation[
        [
            (str(molecule), str(site)) in test_keys
            for molecule, site in zip(full_evaluation["molecule_key"], full_evaluation["site_id"])
        ]
    ].copy()
    weight_lookup = {
        (str(row.molecule_key), str(row.site_id)): float(row.sample_weight)
        for row in training_df.iloc[test_idx].itertuples(index=False)
    }
    evaluation["sample_weight"] = [
        weight_lookup[(str(molecule), str(site))]
        for molecule, site in zip(evaluation["molecule_key"], evaluation["site_id"])
    ]
    evaluation["stage1_abs_error"] = (
        evaluation["experimental_anchor_pka"] - evaluation["stage1_network_macro_pka"]
    ).abs()
    evaluation["stage2_abs_error"] = (
        evaluation["experimental_anchor_pka"] - evaluation["stage2_predicted_macro_pka"]
    ).abs()
    evaluation["stage2_projected_abs_error"] = (
        evaluation["experimental_anchor_pka"] - evaluation["stage2_projected_macro_pka"]
    ).abs()
    evaluation["stage2_supported_projected_abs_error"] = (
        evaluation["experimental_anchor_pka"] - evaluation["stage2_supported_projected_macro_pka"]
    ).abs()
    evaluation["stage2_supported_pairwise_free_energy_abs_error"] = (
        evaluation["experimental_anchor_pka"]
        - evaluation["stage2_supported_pairwise_free_energy_macro_pka"]
    ).abs()
    y_true = evaluation["experimental_anchor_pka"].to_numpy(dtype=float)
    baseline = evaluation["stage1_network_macro_pka"].to_numpy(dtype=float)
    corrected = evaluation["stage2_predicted_macro_pka"].to_numpy(dtype=float)
    projected = evaluation["stage2_projected_macro_pka"].to_numpy(dtype=float)
    supported_projected = evaluation["stage2_supported_projected_macro_pka"].to_numpy(dtype=float)
    supported_pairwise = evaluation[
        "stage2_supported_pairwise_free_energy_macro_pka"
    ].to_numpy(dtype=float)
    evaluation_weights = evaluation["sample_weight"].to_numpy(dtype=float)
    metrics = {
        "rows": int(len(evaluation)),
        "molecules": int(evaluation["molecule_key"].nunique()),
        "full_ladder_sites_predicted": int(len(full_evaluation)),
        "supported_families_from_training_partition": sorted(supported_families),
        "stage1_network_baseline": _metrics(y_true, baseline, evaluation_weights),
        "stage2_corrected_raw": _metrics(y_true, corrected, evaluation_weights),
        "stage2_corrected_monotonic": _metrics(y_true, projected, evaluation_weights),
        "stage2_supported_family_monotonic": _metrics(
            y_true, supported_projected, evaluation_weights
        ),
        "stage2_supported_pairwise_free_energy": _metrics(
            y_true, supported_pairwise, evaluation_weights
        ),
    }
    metrics["mae_improvement"] = (
        metrics["stage1_network_baseline"]["mae"]
        - metrics["stage2_supported_pairwise_free_energy"]["mae"]
    )
    return metrics, evaluation, model


def _select_model(
    training_df: pd.DataFrame,
    full_site_df: pd.DataFrame,
    full_features: pd.DataFrame,
    selection_splits: int,
    test_frac: float,
) -> Tuple[str, object, pd.DataFrame]:
    """Choose the residual regressor by scaffold-grouped held-out performance."""
    groups = training_df["smiles"].map(scaffold_group_key).to_numpy(dtype=str)
    splitter = GroupShuffleSplit(
        n_splits=int(selection_splits),
        test_size=float(test_frac),
        random_state=17,
    )
    splits = list(splitter.split(training_df, groups=groups))
    records = []
    candidates = _model_candidates()
    for name, template in candidates.items():
        for fold, (train_idx, test_idx) in enumerate(splits):
            metrics, _, _ = _fit_evaluate(
                template,
                training_df,
                full_site_df,
                full_features,
                train_idx,
                test_idx,
                MIN_FAMILY_TRAINING_ROWS,
            )
            records.append({
                "candidate": name,
                "selection_fold": int(fold),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "test_molecules": int(training_df.iloc[test_idx]["molecule_key"].nunique()),
                "stage1_baseline_mae": metrics["stage1_network_baseline"]["mae"],
                "stage2_supported_mae": metrics["stage2_supported_pairwise_free_energy"]["mae"],
                "stage2_supported_weighted_mae": metrics["stage2_supported_pairwise_free_energy"]["weighted_mae"],
                "mae_improvement": metrics["mae_improvement"],
            })
    detail = pd.DataFrame(records)
    summary = detail.groupby("candidate").agg(
        mean_supported_mae=("stage2_supported_mae", "mean"),
        std_supported_mae=("stage2_supported_mae", "std"),
        mean_supported_weighted_mae=("stage2_supported_weighted_mae", "mean"),
        mean_mae_improvement=("mae_improvement", "mean"),
    ).sort_values(["mean_supported_mae", "mean_supported_weighted_mae"])
    selected_name = str(summary.index[0])
    return selected_name, candidates[selected_name], detail


def _split_manifest(site_df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
    """Record molecule/site membership for a reproducible validation split."""
    manifest = site_df[["molecule_key", "site_id", "smiles"]].copy()
    manifest["split"] = "unused"
    manifest.loc[train_idx, "split"] = "train"
    manifest.loc[test_idx, "split"] = "test"
    return manifest


def train_stage2(
    network_dataset_path: str,
    out_dir: str,
    fp_bits: int = 512,
    fp_radius: int = 2,
    test_frac: float = 0.2,
    max_replicate_range: float = 2.0,
    min_supported_pka: float = -5.0,
    max_supported_pka: float = 20.0,
    selection_splits: int = 3,
) -> Tuple[Dict, pd.DataFrame]:
    """Train and persist Stage 2 residual correction with provenance-safe evaluation.

    Exact site anchors train and validate the model. Ambiguous molecule-level
    measurements may contribute only low-weight, marginalized deployment data;
    they never establish family support or enter held-out metrics.
    """
    network_df = pd.read_csv(network_dataset_path, low_memory=False)
    if set(network_df["stage1_training_measurement_method"].astype(str).str.lower()) != {"experimental"}:
        raise ValueError("Network dataset does not use an experimental-only Stage 1 model")
    if "stage1_trained_on_all_eligible_rows" in network_df.columns:
        if not network_df["stage1_trained_on_all_eligible_rows"].eq(True).all():
            raise ValueError("Network dataset Stage 1 was not fit on all eligible rows")
    if not network_df["marvin_values_used"].eq(False).all() or not network_df["epik_values_used"].eq(False).all():
        raise ValueError("Network dataset contains prohibited Marvin or Epik labels")

    full_site_df = build_stage2_site_table(network_df, complex_only=True, anchored_only=False)
    full_site_df = full_site_df.reset_index(drop=True)
    full_site_df["full_site_position"] = np.arange(len(full_site_df), dtype=int)
    site_df, quarantine, funnel = prepare_stage2_training_sites(
        full_site_df,
        max_replicate_range=max_replicate_range,
        min_supported_pka=min_supported_pka,
        max_supported_pka=max_supported_pka,
    )
    position_lookup = {
        (str(row.molecule_key), str(row.site_id)): int(row.full_site_position)
        for row in full_site_df.itertuples(index=False)
    }
    site_df["full_site_position"] = [
        position_lookup[(str(molecule), str(site))]
        for molecule, site in zip(site_df["molecule_key"], site_df["site_id"])
    ]
    validate_stage2_site_table(site_df)
    if site_df.empty:
        raise RuntimeError("No experimentally anchored multisite rows available for Stage 2")
    features = build_stage2_feature_matrix(full_site_df, fp_bits=fp_bits, fp_radius=fp_radius)
    weak_site_df, weak_blocked = build_stage2_weak_training_sites(
        network_df, full_site_df
    )
    deployment_training = pd.concat(
        [site_df, weak_site_df], ignore_index=True, sort=False
    )
    target = deployment_training["stage2_delta_target"].astype(float)
    family_counts = site_df["site_family"].value_counts().to_dict()
    supported_families = supported_families_for_counts(family_counts)
    training_positions = deployment_training["full_site_position"].to_numpy(dtype=int)
    if not np.isfinite(features.to_numpy(dtype=float)).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite Stage 2 features or targets")

    random_train, random_test = random_molecule_split(site_df, test_frac=test_frac)
    scaffold_train, scaffold_test = scaffold_molecule_split(site_df, test_frac=test_frac)
    selected_name, selected_template, selection_detail = _select_model(
        site_df,
        full_site_df,
        features,
        selection_splits=selection_splits,
        test_frac=test_frac,
    )
    random_metrics, random_eval, _ = _fit_evaluate(
        selected_template, site_df, full_site_df, features, random_train, random_test,
        MIN_FAMILY_TRAINING_ROWS,
    )
    scaffold_metrics, scaffold_eval, _ = _fit_evaluate(
        selected_template, site_df, full_site_df, features, scaffold_train, scaffold_test,
        MIN_FAMILY_TRAINING_ROWS,
    )

    deployment_model = clone(selected_template)
    deployment_model.fit(
        features.iloc[training_positions],
        target,
        sample_weight=deployment_training["sample_weight"].astype(float),
    )
    scaffold_errors = scaffold_eval[
        "stage2_supported_pairwise_free_energy_abs_error"
    ].to_numpy(dtype=float)
    stage1_hashes = sorted(set(network_df["stage1_model_sha256"].astype(str)))
    if len(stage1_hashes) != 1:
        raise ValueError("Network dataset contains multiple Stage 1 model hashes")
    bundle = {
        "model": deployment_model,
        "feature_columns": list(features.columns),
        "fp_bits": int(fp_bits),
        "fp_radius": int(fp_radius),
        "stage2_schema_version": STAGE2_TRAINING_SCHEMA_VERSION,
        "stage2_free_energy_schema_version": STAGE2_FREE_ENERGY_SCHEMA_VERSION,
        "stage2_free_energy_method": STAGE2_FREE_ENERGY_METHOD,
        "selected_model": selected_name,
        "target_definition": STAGE2_TARGET_DEFINITION,
        "training_measurement_method": "experimental",
        "training_rows": int(len(deployment_training)),
        "exact_site_training_rows": int(len(site_df)),
        "weak_marginalized_training_rows": int(len(weak_site_df)),
        "training_molecules": int(deployment_training["molecule_key"].nunique()),
        "deployment_fit_rows": int(len(deployment_training)),
        "trained_on_all_eligible_rows": True,
        "network_dataset_sha256": _sha256(network_dataset_path),
        "network_dataset_schema_versions": sorted(set(network_df["dataset_schema_version"].astype(str))),
        "functional_group_detector_schema_versions": detector_schema_versions(network_df),
        "stage1_model_sha256": stage1_hashes[0],
        "stage1_schema_versions": sorted(set(network_df.get("stage1_schema_version", pd.Series(dtype=str)).astype(str))),
        "stage1_target_definitions": sorted(set(network_df.get("stage1_target_definition", pd.Series(dtype=str)).astype(str))),
        "minimum_family_training_rows": MIN_FAMILY_TRAINING_ROWS,
        "priority_supported_families": sorted(PRIORITY_SUPPORTED_FAMILIES),
        "family_support_policy": (
            "count_at_least_minimum_or_explicit_priority_family_with_training_data"
        ),
        "family_training_counts": {str(key): int(value) for key, value in family_counts.items()},
        "family_support_counts_exclude_weak_labels": True,
        "supported_families": sorted(supported_families),
        "scaffold_validation_mae": float(
            scaffold_metrics["stage2_supported_pairwise_free_energy"]["mae"]
        ),
        "scaffold_abs_error_p90": float(np.quantile(scaffold_errors, 0.90)),
        "scaffold_abs_error_p95": float(np.quantile(scaffold_errors, 0.95)),
    }
    report = {
        "stage2_schema_version": STAGE2_TRAINING_SCHEMA_VERSION,
        "stage2_free_energy_schema_version": STAGE2_FREE_ENERGY_SCHEMA_VERSION,
        "stage2_free_energy_method": STAGE2_FREE_ENERGY_METHOD,
        "selected_model": selected_name,
        "target_definition": STAGE2_TARGET_DEFINITION,
        "training_measurement_method": "experimental",
        "rows": int(len(deployment_training)),
        "exact_site_rows": int(len(site_df)),
        "weak_marginalized_rows": int(len(weak_site_df)),
        "weak_blocked_incomplete_candidate_space": int(len(weak_blocked)),
        "molecules": int(site_df["molecule_key"].nunique()),
        "features": int(features.shape[1]),
        "full_complex_site_rows": int(len(full_site_df)),
        "funnel": funnel,
        "max_replicate_range": float(max_replicate_range),
        "supported_pka_range": [float(min_supported_pka), float(max_supported_pka)],
        "minimum_family_training_rows": MIN_FAMILY_TRAINING_ROWS,
        "priority_supported_families": sorted(PRIORITY_SUPPORTED_FAMILIES),
        "family_support_policy": (
            "count_at_least_minimum_or_explicit_priority_family_with_training_data"
        ),
        "family_training_counts": {str(key): int(value) for key, value in family_counts.items()},
        "supported_families": sorted(supported_families),
        "model_selection": (
            selection_detail.groupby("candidate").agg(
                folds=("selection_fold", "size"),
                mean_supported_mae=("stage2_supported_mae", "mean"),
                std_supported_mae=("stage2_supported_mae", "std"),
                mean_supported_weighted_mae=("stage2_supported_weighted_mae", "mean"),
                mean_mae_improvement=("mae_improvement", "mean"),
            ).sort_values(["mean_supported_mae", "mean_supported_weighted_mae"])
            .reset_index().to_dict(orient="records")
        ),
        "random_split": random_metrics,
        "scaffold_split": scaffold_metrics,
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    deployment_training.to_csv(os.path.join(out_dir, "training_site_table.csv"), index=False)
    quarantine.to_csv(os.path.join(out_dir, "training_site_quarantine.csv"), index=False)
    weak_site_df.to_csv(os.path.join(out_dir, "weak_molecule_training_rows.csv"), index=False)
    weak_blocked.to_csv(os.path.join(out_dir, "weak_molecule_blocked.csv"), index=False)
    selection_detail.to_csv(os.path.join(out_dir, "model_selection_folds.csv"), index=False)
    bundle["training_dataset_sha256"] = _sha256(
        os.path.join(out_dir, "training_site_table.csv")
    )
    with open(os.path.join(out_dir, "stage2_network_context_model.pkl"), "wb") as handle:
        pickle.dump(bundle, handle)
    report["model_sha256"] = _sha256(
        os.path.join(out_dir, "stage2_network_context_model.pkl")
    )
    report["training_dataset_sha256"] = bundle["training_dataset_sha256"]
    report["network_dataset_sha256"] = bundle["network_dataset_sha256"]
    report["functional_group_detector_schema_versions"] = bundle[
        "functional_group_detector_schema_versions"
    ]
    report["stage1_model_sha256"] = bundle["stage1_model_sha256"]
    report["stage1_schema_versions"] = bundle["stage1_schema_versions"]
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    random_eval.sort_values("stage2_abs_error", ascending=False).to_csv(
        os.path.join(out_dir, "random_eval_predictions.csv"), index=False
    )
    scaffold_eval.sort_values("stage2_abs_error", ascending=False).to_csv(
        os.path.join(out_dir, "scaffold_eval_predictions.csv"), index=False
    )
    _split_manifest(site_df, random_train, random_test).to_csv(
        os.path.join(out_dir, "random_split_manifest.csv"), index=False
    )
    _split_manifest(site_df, scaffold_train, scaffold_test).to_csv(
        os.path.join(out_dir, "scaffold_split_manifest.csv"), index=False
    )
    return report, scaffold_eval


def parse_args() -> argparse.Namespace:
    """Parse command-line paths and reproducible Stage 2 training settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-dataset", default=DEFAULT_NETWORK_DATASET)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--fp-bits", type=int, default=512)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--max-replicate-range", type=float, default=2.0)
    parser.add_argument("--min-supported-pka", type=float, default=-5.0)
    parser.add_argument("--max-supported-pka", type=float, default=20.0)
    parser.add_argument("--selection-splits", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Run Stage 2 training from the CLI and print its machine-readable report."""
    args = parse_args()
    report, _ = train_stage2(
        network_dataset_path=args.network_dataset,
        out_dir=args.out_dir,
        fp_bits=args.fp_bits,
        fp_radius=args.fp_radius,
        test_frac=args.test_frac,
        max_replicate_range=args.max_replicate_range,
        min_supported_pka=args.min_supported_pka,
        max_supported_pka=args.max_supported_pka,
        selection_splits=args.selection_splits,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
