"""Empirical pKa-side calibration for canonical Stage 3 site calls.

The target is whether an observed scalar pKa is above or below the requested
pH. It is a scaffold-held-out pKa proxy, not direct microstate-state evidence.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd  # type: ignore


EMPIRICAL_CALIBRATION_SCHEMA_VERSION = "1.0.0"
EMPIRICAL_CALIBRATION_METHOD = "hierarchical_scaffold_holdout_signed_residual_ecdf"
LABEL_SHRINKAGE_ROWS = 20.0
FAMILY_SHRINKAGE_ROWS = 40.0


def _sha256(path: str) -> str:
    """Hash each held-out prediction artifact used to build calibration."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_values(values: Iterable[float]) -> list[float]:
    """Convert an iterable to finite floating-point values only."""
    return [float(value) for value in values if math.isfinite(float(value))]


def _build_source(
    frame: pd.DataFrame,
    prediction_column: str,
    label_column: str,
    family_column: str,
) -> Dict:
    """Build global, family, and exact-label residual pools from held-out rows."""
    clean = frame.copy()
    clean["_observed"] = pd.to_numeric(clean["experimental_anchor_pka"], errors="coerce")
    clean["_predicted"] = pd.to_numeric(clean[prediction_column], errors="coerce")
    clean = clean[np.isfinite(clean["_observed"]) & np.isfinite(clean["_predicted"])].copy()
    clean["_residual"] = clean["_observed"] - clean["_predicted"]
    clean["_label"] = clean.get(label_column, pd.Series("unknown", index=clean.index)).fillna(
        "unknown"
    ).astype(str)
    clean["_family"] = clean.get(family_column, pd.Series("unknown", index=clean.index)).fillna(
        "unknown"
    ).astype(str)
    return {
        "prediction_column": prediction_column,
        "rows": int(len(clean)),
        "global_residuals": _clean_values(clean["_residual"]),
        "family_residuals": {
            str(key): _clean_values(group["_residual"])
            for key, group in clean.groupby("_family", sort=True)
        },
        "label_residuals": {
            str(key): _clean_values(group["_residual"])
            for key, group in clean.groupby("_label", sort=True)
        },
    }


def build_empirical_calibration(stage1_oof_path: str, stage2_eval_path: str) -> Dict:
    """Create the hierarchical Stage 3 calibration artifact from held-out errors.

    Stage 1 supplies single-site scaffold OOF residuals and Stage 2 supplies
    multisite scaffold-holdout residuals. These calibrate scalar pKa-side calls,
    not tautomer probabilities or directly observed microstate populations.
    """
    stage1 = pd.read_csv(stage1_oof_path, low_memory=False)
    stage2 = pd.read_csv(stage2_eval_path, low_memory=False)
    stage2_prediction_column = (
        "stage2_supported_pairwise_free_energy_macro_pka"
        if "stage2_supported_pairwise_free_energy_macro_pka" in stage2.columns
        else "stage2_supported_projected_macro_pka"
    )
    return {
        "schema_version": EMPIRICAL_CALIBRATION_SCHEMA_VERSION,
        "method": EMPIRICAL_CALIBRATION_METHOD,
        "target": "observed_scalar_pka_is_on_predicted_site_form_side_of_requested_ph",
        "semantics": (
            "scaffold-held-out pKa residual calibration proxy; not direct experimental "
            "microstate-state or tautomer-population calibration"
        ),
        "label_shrinkage_rows": LABEL_SHRINKAGE_ROWS,
        "family_shrinkage_rows": FAMILY_SHRINKAGE_ROWS,
        "stage1_oof_sha256": _sha256(stage1_oof_path),
        "stage2_eval_sha256": _sha256(stage2_eval_path),
        "sources": {
            "stage1_single_site": _build_source(
                stage1, "pred_intrinsic_pka", "final_group_label", "site_family"
            ),
            "stage2_multisite": _build_source(
                stage2, stage2_prediction_column, "site_label", "site_family"
            ),
        },
    }


def _smoothed_probability_at_or_above(values: Sequence[float], threshold: float) -> float:
    """Estimate an empirical upper-tail probability with half-count smoothing."""
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return 0.5
    successes = float(np.sum(array >= float(threshold)))
    return float((successes + 0.5) / (len(array) + 1.0))


def _hierarchy_components(source: Mapping, site_label: str, site_family: str) -> list[tuple]:
    """Blend sparse label/family residual pools toward the global distribution."""
    global_values = list(source.get("global_residuals", []))
    family_values = list(source.get("family_residuals", {}).get(str(site_family), []))
    label_values = list(source.get("label_residuals", {}).get(str(site_label), []))
    family_weight = len(family_values) / (len(family_values) + FAMILY_SHRINKAGE_ROWS)
    label_weight = len(label_values) / (len(label_values) + LABEL_SHRINKAGE_ROWS)
    components = []
    if label_values:
        components.append((label_values, float(label_weight), "label"))
    remaining = 1.0 - label_weight
    if family_values:
        components.append((family_values, float(remaining * family_weight), "family"))
    components.append((global_values, float(remaining * (1.0 - family_weight)), "global"))
    total = sum(component[1] for component in components)
    if total <= 0.0:
        return [(global_values, 1.0, "global")]
    return [(values, weight / total, name) for values, weight, name in components]


def _weighted_quantile(components: Sequence[tuple], quantile: float) -> float:
    """Evaluate a quantile of the weighted hierarchical residual mixture."""
    samples = []
    weights = []
    for values, component_weight, _ in components:
        if not values or component_weight <= 0.0:
            continue
        samples.extend(float(value) for value in values)
        weights.extend([float(component_weight) / len(values)] * len(values))
    if not samples:
        return float("nan")
    order = np.argsort(np.asarray(samples, dtype=float))
    sorted_samples = np.asarray(samples, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights) / float(np.sum(sorted_weights))
    index = int(np.searchsorted(cumulative, float(quantile), side="left"))
    return float(sorted_samples[min(index, len(sorted_samples) - 1)])


def calibrate_site_call(
    calibration: Mapping,
    source_name: str,
    site_label: str,
    site_family: str,
    predicted_pka: float,
    ph: float,
    predicted_site_form: str,
) -> Dict:
    """Calibrate whether a scalar observed pKa supports the predicted side at pH.

    The returned confidence and interval use held-out residual distributions with
    exact-label/family/global shrinkage. They are empirical proxies, not direct
    experimental validation of the selected complete microstate.
    """
    source = calibration["sources"][str(source_name)]
    components = _hierarchy_components(source, str(site_label), str(site_family))
    threshold = float(ph) - float(predicted_pka)
    acid_probability = float(sum(
        float(weight) * _smoothed_probability_at_or_above(values, threshold)
        for values, weight, _ in components
    ))
    if str(predicted_site_form) == "acid_form":
        selected_confidence = acid_probability
    elif str(predicted_site_form) == "base_form":
        selected_confidence = 1.0 - acid_probability
    else:
        selected_confidence = float("nan")
    active_names = [name for _, weight, name in components if weight > 1e-12]
    label_rows = len(source.get("label_residuals", {}).get(str(site_label), []))
    family_rows = len(source.get("family_residuals", {}).get(str(site_family), []))
    return {
        "source": str(source_name),
        "scope": "_".join(active_names) + "_hierarchical_shrinkage",
        "source_rows": int(source.get("rows", 0)),
        "label_rows": int(label_rows),
        "family_rows": int(family_rows),
        "probability_observed_pka_at_or_above_ph": acid_probability,
        "predicted_site_form_pka_side_confidence": float(selected_confidence),
        "empirical_macro_pka_interval_90_low": float(
            float(predicted_pka) + _weighted_quantile(components, 0.05)
        ),
        "empirical_macro_pka_interval_90_high": float(
            float(predicted_pka) + _weighted_quantile(components, 0.95)
        ),
        "method": EMPIRICAL_CALIBRATION_METHOD,
        "semantics": (
            "probability that scalar observed pKa supports the Stage 3 acid/base-side "
            "call at this pH; proxy only for coupled microstate correctness"
        ),
    }


def calibration_diagnostics(calibration: Mapping, ph: float) -> Dict:
    """Summarize calibration coverage and global held-out residual magnitudes."""
    result = {}
    for source_name, source in calibration["sources"].items():
        residuals = np.asarray(source.get("global_residuals", []), dtype=float)
        if len(residuals):
            absolute = np.abs(residuals)
            result[source_name] = {
                "rows": int(len(residuals)),
                "median_abs_residual": float(np.median(absolute)),
                "p90_abs_residual": float(np.quantile(absolute, 0.90)),
                "p95_abs_residual": float(np.quantile(absolute, 0.95)),
                "calibration_ph": float(ph),
                "status": "residual_reference_summary_not_independent_state_validation",
            }
        else:
            result[source_name] = {"rows": 0, "status": "unavailable"}
    return result
