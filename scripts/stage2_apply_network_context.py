#!/usr/bin/env python3
"""Apply canonical Stage 2 corrections to molecule microstate networks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd  # type: ignore

from stage2_network_context import (
    STAGE2_TARGET_DEFINITION,
    STAGE2_TRAINING_SCHEMA_VERSION,
    _network_is_limited,
    build_stage2_feature_matrix,
    build_stage2_site_table,
    detector_schema_versions,
    project_macro_pkas_nonincreasing_with_fixed,
)
from stage2_free_energy_coupling import (
    STAGE2_FREE_ENERGY_METHOD,
    STAGE2_FREE_ENERGY_SCHEMA_VERSION,
    contextual_edge_pkas,
    infer_regularized_pairwise_free_energy,
    pair_coupling_map,
)


DEFAULT_INPUT = "data/processed/pka_molecule_microstate_network_dataset.csv"
DEFAULT_MODEL = "data/processed/ml_models_experimental_only/stage2_network_context/stage2_network_context_model.pkl"
DEFAULT_OUTPUT = "data/processed/pka_molecule_microstate_network_stage2_predictions.csv"


def _json(value: object) -> str:
    """Serialize nested network fields deterministically for stable artifacts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: str) -> str:
    """Hash an artifact for model, input, and output provenance checks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: list[float]) -> Dict[str, float | int | None]:
    """Summarize a diagnostic distribution while handling an empty sample."""
    if not values:
        return {"count": 0, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def apply_stage2(
    network_dataset_path: str,
    model_path: str,
    output_path: str,
    allow_compatible_network: bool = False,
    report_path: str | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Apply Stage 2 and emit a cycle-consistent pairwise free-energy model.

    Supported families receive learned residual corrections; other sites retain
    their Stage 1 baseline. The corrected macro ladder is monotonically projected
    before regularized one-body and site-pair terms are inferred for Stage 3.
    """
    network_df = pd.read_csv(network_dataset_path, low_memory=False)
    with open(model_path, "rb") as handle:
        bundle = pickle.load(handle)
    required = {
        "model",
        "feature_columns",
        "target_definition",
        "training_measurement_method",
        "stage1_model_sha256",
        "stage2_schema_version",
        "network_dataset_sha256",
        "network_dataset_schema_versions",
        "functional_group_detector_schema_versions",
        "stage1_schema_versions",
        "trained_on_all_eligible_rows",
        "deployment_fit_rows",
        "training_rows",
        "stage2_free_energy_schema_version",
        "stage2_free_energy_method",
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Stage 2 bundle missing provenance: {sorted(missing)}")
    if str(bundle["training_measurement_method"]).lower() != "experimental":
        raise ValueError("Stage 2 model is not experimental-only")
    if str(bundle["stage2_schema_version"]) != STAGE2_TRAINING_SCHEMA_VERSION:
        raise ValueError(
            "Stage 2 bundle schema is stale: "
            f"{bundle['stage2_schema_version']} != {STAGE2_TRAINING_SCHEMA_VERSION}"
        )
    if str(bundle["target_definition"]) != STAGE2_TARGET_DEFINITION:
        raise ValueError("Stage 2 bundle target definition is incompatible")
    if str(bundle["stage2_free_energy_schema_version"]) != STAGE2_FREE_ENERGY_SCHEMA_VERSION:
        raise ValueError("Stage 2 bundle free-energy schema is incompatible")
    if str(bundle["stage2_free_energy_method"]) != STAGE2_FREE_ENERGY_METHOD:
        raise ValueError("Stage 2 bundle free-energy method is incompatible")
    if not bool(bundle["trained_on_all_eligible_rows"]):
        raise ValueError("Stage 2 deployment model was not fit on all eligible rows")
    if int(bundle["deployment_fit_rows"]) != int(bundle["training_rows"]):
        raise ValueError("Stage 2 deployment-fit row count does not match training rows")
    if not network_df["marvin_values_used"].eq(False).all() or not network_df["epik_values_used"].eq(False).all():
        raise ValueError("Network dataset contains prohibited Marvin or Epik labels")
    if not network_df["stage1_trained_on_all_eligible_rows"].eq(True).all():
        raise ValueError("Network dataset Stage 1 was not fit on all eligible rows")
    stage1_hashes = set(network_df["stage1_model_sha256"].astype(str))
    if stage1_hashes != {str(bundle["stage1_model_sha256"])}:
        raise ValueError("Stage 2 model was trained against a different Stage 1 model")
    network_schema_versions = sorted(set(network_df["dataset_schema_version"].astype(str)))
    if network_schema_versions != list(bundle["network_dataset_schema_versions"]):
        raise ValueError("Stage 2 model and network dataset schema versions differ")
    detector_versions = detector_schema_versions(network_df)
    if detector_versions != list(bundle["functional_group_detector_schema_versions"]):
        raise ValueError("Stage 2 model and functional-group detector versions differ")
    stage1_schema_versions = sorted(set(network_df["stage1_schema_version"].astype(str)))
    if stage1_schema_versions != list(bundle["stage1_schema_versions"]):
        raise ValueError("Stage 2 model and Stage 1 schema versions differ")
    if not allow_compatible_network and _sha256(network_dataset_path) != str(bundle["network_dataset_sha256"]):
        raise ValueError(
            "Stage 2 model was trained on a different network snapshot; retrain Stage 2 "
            "or pass --allow-compatible-network for deliberate inference on a compatible dataset"
        )

    site_df = build_stage2_site_table(network_df, complex_only=True, anchored_only=False)
    if site_df.empty:
        # A single submitted molecule may have only one ionization coordinate.
        # It needs no Stage 2 context correction, but should still pass through
        # the canonical free-energy and Stage 3 machinery.
        site_df = pd.DataFrame(
            columns=[
                "molecule_id",
                "site_id",
                "stage1_network_macro_pka",
                "stage2_predicted_delta",
                "stage2_predicted_macro_pka",
            ]
        )
    else:
        features = build_stage2_feature_matrix(
            site_df,
            fp_bits=int(bundle.get("fp_bits", 512)),
            fp_radius=int(bundle.get("fp_radius", 2)),
        ).reindex(columns=bundle["feature_columns"], fill_value=0.0)
        # Parallel tree reduction can differ in the last few floating-point bits.
        # Stabilize the downstream nonlinear closure and artifact hashes.
        site_df["stage2_predicted_delta"] = np.round(
            bundle["model"].predict(features), 10
        )
        site_df["stage2_predicted_macro_pka"] = (
            site_df["stage1_network_macro_pka"] + site_df["stage2_predicted_delta"]
        )
    supported_families = set(str(value) for value in bundle.get("supported_families", []))
    priority_families = set(
        str(value) for value in bundle.get("priority_supported_families", [])
    )
    minimum_family_rows = int(bundle.get("minimum_family_training_rows", 50))
    prediction_lookup = {
        (row.molecule_id, row.site_id): row
        for row in site_df.itertuples(index=False)
    }

    rows = []
    model_sha256 = _sha256(model_path)
    order_violation_count = 0
    projected_molecule_count = 0
    projected_site_count = 0
    free_energy_optimizer_failure_count = 0
    pair_parameter_molecule_count = 0
    active_pair_parameter_count = 0
    prior_dependent_pair_molecule_count = 0
    free_energy_reconstruction_errors = []
    absolute_pair_couplings = []
    contextual_edge_pka_spreads = []
    maximum_cycle_closure_error = 0.0
    for _, row in network_df.iterrows():
        sites = json.loads(str(row["sites_json"]))
        nodes = json.loads(str(row["microstate_nodes_json"]))
        edges = json.loads(str(row["microstate_edges_json"]))
        network_limited = _network_is_limited(row)
        baseline_steps = json.loads(str(row.get(
            "baseline_macro_pka_steps_json",
            row["independent_site_macro_pka_steps_json"],
        )))
        corrected_steps = []
        applied = 0
        for site in sites:
            prediction = prediction_lookup.get((str(row["molecule_id"]), str(site["site_id"])))
            if prediction is None:
                baseline_raw = site.get("rank_associated_macro_pka")
                baseline = float(
                    site.get("stage1_intrinsic_pka")
                    if baseline_raw is None else baseline_raw
                )
                site.update({
                    "stage2_applied": False,
                    "stage2_predicted_delta": 0.0,
                    "stage2_predicted_macro_pka": (
                        None if network_limited else baseline
                    ),
                    "stage2_prediction_confidence": (
                        "unavailable_incomplete_or_truncated_network"
                        if network_limited else "not_applicable_single_site"
                    ),
                })
                stage1_pka = float(site.get("stage1_intrinsic_pka", baseline))
                low = float(site.get("stage1_predicted_pka_ci_low", stage1_pka - 3.0))
                high = float(site.get("stage1_predicted_pka_ci_high", stage1_pka + 3.0))
                half_width = max(stage1_pka - low, high - stage1_pka)
                site["stage2_predicted_macro_pka_ci_low"] = baseline - half_width
                site["stage2_predicted_macro_pka_ci_high"] = baseline + half_width
            else:
                family_supported = str(site["family"]) in supported_families
                family_training_count = int(
                    bundle.get("family_training_counts", {}).get(str(site["family"]), 0)
                )
                priority_limited_support = (
                    str(site["family"]) in priority_families
                    and family_training_count < minimum_family_rows
                )
                applied += int(family_supported)
                raw_delta = float(prediction.stage2_predicted_delta)
                baseline = float(prediction.stage1_network_macro_pka)
                site.update({
                    "stage2_applied": family_supported,
                    "stage2_raw_model_delta": raw_delta,
                    "stage2_predicted_delta": raw_delta if family_supported else 0.0,
                    "stage2_raw_predicted_macro_pka": baseline + (raw_delta if family_supported else 0.0),
                    "stage2_prediction_confidence": (
                        "model_estimate_priority_family_limited_support"
                        if priority_limited_support
                        else "model_estimate_supported_family"
                        if family_supported
                        else "fallback_stage1_insufficient_family_training_support"
                    ),
                    "stage2_family_training_count": family_training_count,
                    "stage2_priority_family": str(site["family"]) in priority_families,
                    "stage2_scaffold_validation_mae": float(bundle["scaffold_validation_mae"]),
                    "stage2_abs_error_p90": float(bundle["scaffold_abs_error_p90"]),
                    "stage2_abs_error_p95": float(bundle["scaffold_abs_error_p95"]),
                })
                stage1_pka = float(site.get("stage1_intrinsic_pka", baseline))
                low = float(site.get("stage1_predicted_pka_ci_low", stage1_pka - 3.0))
                high = float(site.get("stage1_predicted_pka_ci_high", stage1_pka + 3.0))
                stage1_half_width = max(stage1_pka - low, high - stage1_pka)
                stage2_half_width = (
                    math.sqrt(
                        stage1_half_width * stage1_half_width
                        + float(bundle["scaffold_abs_error_p95"]) ** 2
                    )
                    if family_supported
                    else stage1_half_width
                )
                raw_center = baseline + (raw_delta if family_supported else 0.0)
                site["stage2_predicted_macro_pka_ci_low"] = raw_center - stage2_half_width
                site["stage2_predicted_macro_pka_ci_high"] = raw_center + stage2_half_width

        site_lookup = {site["site_id"]: site for site in sites}
        for step in baseline_steps:
            site = site_lookup[step["rank_associated_site_id"]]
            corrected_steps.append({
                **step,
                "stage1_network_macro_pka": float(step["predicted_macro_pka"]),
                "stage2_predicted_delta": float(site["stage2_predicted_delta"]),
                "stage2_raw_predicted_macro_pka": float(
                    site.get("stage2_raw_predicted_macro_pka", site.get("stage2_predicted_macro_pka"))
                ),
                "stage2_applied": bool(site["stage2_applied"]),
            })
        corrected_values = [step["stage2_raw_predicted_macro_pka"] for step in corrected_steps]
        order_violation = any(
            corrected_values[idx] < corrected_values[idx + 1]
            for idx in range(len(corrected_values) - 1)
        )
        order_violation_count += int(order_violation)
        adjustable = [bool(step["stage2_applied"]) for step in corrected_steps]
        projected_values = project_macro_pkas_nonincreasing_with_fixed(
            corrected_values,
            adjustable,
        )
        adjusted_here = 0
        for step, projected in zip(corrected_steps, projected_values):
            step["stage2_predicted_macro_pka"] = projected
            step["stage2_monotonic_projection_applied"] = not abs(
                projected - step["stage2_raw_predicted_macro_pka"]
            ) < 1e-12
            adjusted_here += int(step["stage2_monotonic_projection_applied"])
            site = site_lookup[step["rank_associated_site_id"]]
            site["stage2_predicted_macro_pka"] = projected
            old_center = float(site.get("stage2_raw_predicted_macro_pka", projected))
            low = float(site.get("stage2_predicted_macro_pka_ci_low", projected))
            high = float(site.get("stage2_predicted_macro_pka_ci_high", projected))
            site["stage2_predicted_macro_pka_ci_low"] = projected + (low - old_center)
            site["stage2_predicted_macro_pka_ci_high"] = projected + (high - old_center)
            site["stage2_monotonic_projection_applied"] = step["stage2_monotonic_projection_applied"]
        projected_molecule_count += int(adjusted_here > 0)
        projected_site_count += adjusted_here

        if network_limited:
            one_body_map = {
                str(site["site_id"]): float(site["stage1_intrinsic_pka"])
                for site in sites
            }
            pair_terms = []
            free_energy_diagnostics = {
                "status": "unavailable_incomplete_or_truncated_network",
                "method": STAGE2_FREE_ENERGY_METHOD,
                "optimizer_success": True,
                "pair_parameter_count": 0,
                "active_pair_parameter_count": 0,
                "pair_coupling_identifiability": "unavailable_incomplete_network",
                "pair_couplings_experimentally_identified": False,
                "macro_reconstruction_mae": None,
                "macro_reconstruction_max_abs_error": None,
            }
            cycle_error = 0.0
            for edge in edges:
                edge["stage2_contextual_edge_pka"] = None
                edge["stage2_pair_interaction_shift"] = None
                edge["stage2_edge_energy_model"] = (
                    "unavailable_incomplete_or_truncated_network"
                )
        else:
            stage1_local = [float(site["stage1_intrinsic_pka"]) for site in sites]
            site_adjustable = [bool(site.get("stage2_applied", False)) for site in sites]
            proposed_deltas = [
                float(site.get("stage2_predicted_delta", 0.0)) for site in sites
            ]
            observed_anchor_count = sum(
                site.get("experimental_anchor_pka") is not None for site in sites
            )
            one_body_map, pair_terms, free_energy_diagnostics = (
                infer_regularized_pairwise_free_energy(
                    stage1_local_pkas=stage1_local,
                    stage2_macro_pkas=projected_values,
                    adjustable=site_adjustable,
                    proposed_deltas=proposed_deltas,
                    nodes=nodes,
                    sites=sites,
                    observed_macro_anchor_count=observed_anchor_count,
                )
            )
            pair_map = pair_coupling_map(pair_terms)
            edge_values, cycle_error = contextual_edge_pkas(
                nodes, edges, sites, one_body_map, pair_map
            )
            for edge in edges:
                edge_value = edge_values[str(edge["edge_id"])]
                edge.update({
                    "stage2_contextual_edge_pka": edge_value["contextual_edge_pka"],
                    "stage2_one_body_pka": edge_value["one_body_pka"],
                    "stage2_pair_interaction_shift": edge_value["pair_interaction_shift"],
                    "stage2_cycle_closure_error": edge_value["cycle_closure_error"],
                    "stage2_edge_energy_model": (
                        "cycle_consistent_symmetric_pairwise_free_energy"
                    ),
                })
            free_energy_optimizer_failure_count += int(
                not bool(free_energy_diagnostics["optimizer_success"])
            )
            pair_parameter_molecule_count += int(
                int(free_energy_diagnostics["pair_parameter_count"]) > 0
            )
            active_pair_parameter_count += int(
                free_energy_diagnostics["active_pair_parameter_count"]
            )
            prior_dependent_pair_molecule_count += int(
                str(free_energy_diagnostics["pair_coupling_identifiability"])
                .startswith("regularized_prior_dependent")
            )
            free_energy_reconstruction_errors.extend(
                [float(free_energy_diagnostics["macro_reconstruction_mae"])]
            )
            maximum_cycle_closure_error = max(
                maximum_cycle_closure_error, float(cycle_error)
            )
            absolute_pair_couplings.extend(
                abs(float(term["coupling_log10_units"])) for term in pair_terms
            )
            contextual_by_site: Dict[str, list[float]] = {}
            for edge in edges:
                contextual = edge.get("stage2_contextual_edge_pka")
                if contextual is not None:
                    contextual_by_site.setdefault(str(edge["site_id"]), []).append(
                        float(contextual)
                    )
            contextual_edge_pka_spreads.extend(
                max(values) - min(values) for values in contextual_by_site.values()
            )

        partner_count = {str(site["site_id"]): 0 for site in sites}
        for term in pair_terms:
            partner_count[str(term["site_i"])] += 1
            partner_count[str(term["site_j"])] += 1
        for site in sites:
            site_id = str(site["site_id"])
            site["stage2_one_body_pka"] = float(one_body_map[site_id])
            site["stage2_one_body_shift_from_stage1"] = float(
                one_body_map[site_id] - float(site["stage1_intrinsic_pka"])
            )
            site["stage2_pair_coupling_partner_count"] = int(partner_count[site_id])
            site["stage2_free_energy_identifiability"] = free_energy_diagnostics[
                "pair_coupling_identifiability"
            ]

        free_energy_model = {
            "schema_version": STAGE2_FREE_ENERGY_SCHEMA_VERSION,
            "method": STAGE2_FREE_ENERGY_METHOD,
            "energy_convention": (
                "log10_weight=sum(one_body_pka*x)+sum(J*x_i*x_j)-nH*pH"
            ),
            "temperature_k_for_reported_delta_g": 298.15,
            "one_body_terms": [
                {
                    "site_id": str(site["site_id"]),
                    "one_body_pka": float(one_body_map[str(site["site_id"])]),
                    "stage1_intrinsic_pka": float(site["stage1_intrinsic_pka"]),
                    "stage2_supported": bool(site.get("stage2_applied", False)),
                }
                for site in sites
            ],
            "pair_couplings": pair_terms,
            "diagnostics": free_energy_diagnostics,
        }
        output = row.to_dict()
        output.update({
            "dataset_schema_version": str(row["dataset_schema_version"]),
            "sites_json": _json(sites),
            "microstate_edges_json": _json(edges),
            "stage2_macro_pka_steps_json": _json(corrected_steps),
            "stage2_free_energy_model_json": _json(free_energy_model),
            "stage2_free_energy_schema_version": STAGE2_FREE_ENERGY_SCHEMA_VERSION,
            "stage2_free_energy_method": STAGE2_FREE_ENERGY_METHOD,
            "stage2_pair_parameter_count": int(
                free_energy_diagnostics["pair_parameter_count"]
            ),
            "stage2_active_pair_parameter_count": int(
                free_energy_diagnostics["active_pair_parameter_count"]
            ),
            "stage2_pair_coupling_identifiability": free_energy_diagnostics[
                "pair_coupling_identifiability"
            ],
            "stage2_free_energy_macro_reconstruction_mae": (
                free_energy_diagnostics["macro_reconstruction_mae"]
            ),
            "stage2_max_cycle_closure_error": float(cycle_error),
            "stage2_applied_site_count": applied,
            "stage2_raw_macro_order_violation": order_violation,
            "stage2_monotonic_projection_applied": adjusted_here > 0,
            "stage2_model_sha256": model_sha256,
            "stage2_schema_version": bundle["stage2_schema_version"],
            "stage2_selected_model": bundle.get("selected_model"),
            "stage2_target_definition": bundle["target_definition"],
            "stage2_training_measurement_method": bundle["training_measurement_method"],
            "stage2_training_rows": bundle.get("training_rows"),
            "stage2_scaffold_validation_mae": bundle.get("scaffold_validation_mae"),
            "stage2_population_update_status": (
                "cycle_consistent_free_energy_model_emitted; populations_deferred_to_stage3"
            ),
        })
        rows.append(output)

    result = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    report = {
        "molecule_rows": int(len(result)),
        "complex_site_predictions": int(len(site_df)),
        "molecules_with_macro_order_violation": int(order_violation_count),
        "molecules_with_monotonic_projection": int(projected_molecule_count),
        "sites_adjusted_by_monotonic_projection": int(projected_site_count),
        "free_energy_optimizer_failures": int(free_energy_optimizer_failure_count),
        "molecules_with_pair_parameters": int(pair_parameter_molecule_count),
        "active_pair_parameters": int(active_pair_parameter_count),
        "prior_dependent_pair_molecules": int(prior_dependent_pair_molecule_count),
        "mean_free_energy_macro_reconstruction_mae": (
            float(sum(free_energy_reconstruction_errors) / len(free_energy_reconstruction_errors))
            if free_energy_reconstruction_errors else None
        ),
        "absolute_pair_coupling_log10_units": _distribution(absolute_pair_couplings),
        "contextual_edge_pka_spread": _distribution(contextual_edge_pka_spreads),
        "maximum_cycle_closure_error": float(maximum_cycle_closure_error),
        "model_sha256": model_sha256,
        "stage1_model_sha256": bundle["stage1_model_sha256"],
        "stage2_schema_version": bundle["stage2_schema_version"],
        "stage2_free_energy_schema_version": STAGE2_FREE_ENERGY_SCHEMA_VERSION,
        "stage2_free_energy_method": STAGE2_FREE_ENERGY_METHOD,
        "selected_model": bundle.get("selected_model"),
        "output_sha256": _sha256(output_path),
    }
    report_path = report_path or os.path.join(
        os.path.dirname(model_path), "stage2_free_energy_application_metrics.json"
    )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return result, report


def parse_args() -> argparse.Namespace:
    """Parse Stage 2 application paths and compatibility policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-dataset", default=DEFAULT_INPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-compatible-network",
        action="store_true",
        help="Allow a different network file when schema, detector and Stage 1 provenance match.",
    )
    return parser.parse_args()


def main() -> None:
    """Apply Stage 2 from the CLI and print the free-energy diagnostics."""
    args = parse_args()
    _, report = apply_stage2(
        args.network_dataset,
        args.model,
        args.output,
        allow_compatible_network=args.allow_compatible_network,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
