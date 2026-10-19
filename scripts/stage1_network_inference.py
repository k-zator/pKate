"""Canonical Stage 1 inference helpers for atom-mapped network sites."""

from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd  # type: ignore
from rdkit import rdBase  # type: ignore

from train_intrinsic_single_group_model import (
    STAGE1_FEATURE_SCHEMA_VERSION,
    build_intrinsic_features,
)
from ml_features import morgan_bits


@lru_cache(maxsize=128)
def _training_fingerprint_matrix(
    training_smiles: Tuple[str, ...], nbits: int, radius: int
) -> np.ndarray:
    """Cache label-specific training fingerprints for repeated site queries."""
    if not training_smiles:
        return np.empty((0, int(nbits)), dtype=bool)
    return np.vstack([
        morgan_bits(value, nbits=int(nbits), radius=int(radius)) > 0
        for value in training_smiles
    ])


def load_stage1_bundle(path: str, require_experimental_only: bool = True) -> Dict:
    """Load a deployable Stage 1 model and validate its fit and data provenance."""
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    required = {"model", "feature_columns", "fp_bits", "fp_radius"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Invalid Stage 1 bundle; missing keys: {sorted(missing)}")
    if require_experimental_only and str(bundle.get("training_measurement_method", "")).lower() != "experimental":
        raise ValueError("Stage 1 bundle is not certified experimental-only")
    # Every canonical version must prove that the deployable estimator was
    # refit on all eligible rows. Restricting this check to schema 2.0.0
    # accidentally weakened validation as soon as the schema was bumped.
    if bundle.get("stage1_schema_version"):
        if not bundle.get("trained_on_all_eligible_rows", False):
            raise ValueError("Canonical Stage 1 deployment model was not fit on all eligible rows")
        if int(bundle.get("deployment_fit_rows", -1)) != int(bundle.get("training_rows", -2)):
            raise ValueError("Canonical Stage 1 deployment-fit provenance is inconsistent")

    feature_version = bundle.get("stage1_feature_schema_version")
    if (
        feature_version is not None
        and str(feature_version) != STAGE1_FEATURE_SCHEMA_VERSION
    ):
        raise ValueError(
            "Stage 1 feature schema version mismatch: "
            f"bundle={feature_version!r}, code={STAGE1_FEATURE_SCHEMA_VERSION!r}"
        )

    feature_columns = bundle["feature_columns"]
    if not isinstance(feature_columns, (list, tuple)) or not feature_columns:
        raise ValueError("Invalid Stage 1 bundle; feature_columns must be non-empty")
    feature_columns = [str(column) for column in feature_columns]
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Invalid Stage 1 bundle; feature_columns contains duplicates")

    model = bundle["model"]
    fitted_columns = getattr(model, "feature_names_in_", None)
    if fitted_columns is not None:
        fitted_columns = [str(column) for column in fitted_columns]
        if fitted_columns != feature_columns:
            raise ValueError(
                "Stage 1 bundle feature columns do not match the fitted model"
            )
    fitted_count = getattr(model, "n_features_in_", None)
    if fitted_count is not None and int(fitted_count) != len(feature_columns):
        raise ValueError(
            "Stage 1 bundle feature count does not match the fitted model: "
            f"bundle={len(feature_columns)}, model={int(fitted_count)}"
        )
    return bundle


def _predict_chunk(payload: Tuple[str, List[Dict]]) -> List[Tuple[str, float]]:
    """Featurize and predict one worker chunk, returning ``(task_id, pKa)`` pairs."""
    bundle_path, records = payload
    bundle = load_stage1_bundle(bundle_path, require_experimental_only=True)
    frame = pd.DataFrame(records)
    with rdBase.BlockLogs():
        features = build_intrinsic_features(
            frame,
            nbits=int(bundle.get("fp_bits", 512)),
            radius=int(bundle.get("fp_radius", 2)),
            label_col="candidate_label",
        )
    features = features.reindex(columns=bundle["feature_columns"], fill_value=0.0)
    predictions = np.asarray(bundle["model"].predict(features), dtype=float).reshape(-1)
    if len(predictions) != len(frame):
        raise RuntimeError(
            "Stage 1 model returned a different number of pKa predictions "
            f"({len(predictions)}) than submitted site tasks ({len(frame)})"
        )
    if not np.isfinite(predictions).all():
        bad_task_ids = frame.loc[~np.isfinite(predictions), "task_id"].astype(str)
        raise ValueError(
            "Stage 1 model returned a non-finite pKa for task(s): "
            + ", ".join(bad_task_ids)
        )
    return [(task_id, float(value)) for task_id, value in zip(frame["task_id"], predictions)]


def predict_stage1_tasks(tasks: List[Dict], bundle_path: str, workers: int = 1) -> Dict[str, float]:
    """Predict intrinsic pKa for site tasks without using experimental pKa."""
    if not tasks:
        return {}
    task_ids = [str(task.get("task_id", "")) for task in tasks]
    if any(not task_id for task_id in task_ids):
        raise ValueError("Every Stage 1 prediction task requires a non-empty task_id")
    if len(task_ids) != len(set(task_ids)):
        duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
        raise ValueError(f"Duplicate Stage 1 task_id values: {duplicates}")
    workers = max(1, int(workers))
    if workers == 1:
        return dict(_predict_chunk((bundle_path, tasks)))

    by_molecule: Dict[str, List[Dict]] = {}
    for task in tasks:
        by_molecule.setdefault(task["molecule_id"], []).append(task)
    chunks: List[List[Dict]] = [[] for _ in range(workers)]
    chunk_sizes = [0] * workers
    for _, records in sorted(by_molecule.items(), key=lambda item: (-len(item[1]), item[0])):
        target = min(range(workers), key=lambda idx: chunk_sizes[idx])
        chunks[target].extend(records)
        chunk_sizes[target] += len(records)
    payloads = [(bundle_path, chunk) for chunk in chunks if chunk]

    predictions: Dict[str, float] = {}
    try:
        with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
            for result in executor.map(_predict_chunk, payloads):
                predictions.update(result)
    except PermissionError:
        predictions = dict(_predict_chunk((bundle_path, tasks)))
    return predictions


def stage1_applicability(smiles: str, label: str, bundle: Dict) -> Dict[str, object]:
    """Return exact-label support and nearest-neighbour applicability."""
    training_smiles = bundle.get("training_smiles_by_label", {}).get(str(label), [])
    count = int(bundle.get("training_site_label_counts", {}).get(str(label), len(training_smiles)))
    if not training_smiles:
        return {
            "stage1_exact_label_training_rows": count,
            "stage1_nearest_same_label_tanimoto": None,
            "stage1_applicability_domain": "zero_shot_exact_label_absent",
        }
    nbits = int(bundle.get("fp_bits", 512))
    radius = int(bundle.get("fp_radius", 2))
    query = morgan_bits(str(smiles), nbits=nbits, radius=radius) > 0
    matrix = _training_fingerprint_matrix(tuple(map(str, training_smiles)), nbits, radius)
    intersections = np.logical_and(matrix, query).sum(axis=1)
    unions = np.logical_or(matrix, query).sum(axis=1)
    similarities = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=float),
        where=unions > 0,
    )
    best = float(np.max(similarities)) if len(similarities) else 0.0
    domain = (
        "interpolation_high_similarity"
        if best >= 0.70 and count >= 5
        else "interpolation_limited_support"
        if best >= 0.40 and count >= 2
        else "extrapolation_low_similarity_or_sparse_label"
    )
    return {
        "stage1_exact_label_training_rows": count,
        "stage1_nearest_same_label_tanimoto": best,
        "stage1_applicability_domain": domain,
    }
