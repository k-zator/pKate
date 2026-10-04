#!/usr/bin/env python3
"""Canonical Stage 3: infer internally consistent microstate populations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore

from stage3_empirical_calibration import (
    EMPIRICAL_CALIBRATION_METHOD,
    EMPIRICAL_CALIBRATION_SCHEMA_VERSION,
    build_empirical_calibration,
    calibrate_site_call,
    calibration_diagnostics,
)
from stage2_free_energy_coupling import (
    STAGE2_FREE_ENERGY_METHOD,
    STAGE2_FREE_ENERGY_SCHEMA_VERSION,
    annotate_pairwise_microstate_populations,
    macro_pka_values_from_free_energy,
    pair_coupling_map,
    site_marginals_from_pairwise_model,
)
from stage3_microstate_thermodynamics import (
    STAGE3_LOCAL_PKA_METHOD,
    STAGE3_SCHEMA_VERSION,
)
from stage3_tautomer_ranking import (
    DEFAULT_MAX_TAUTOMERS_PER_CONFIGURATION,
    DEFAULT_TAUTOMER_SCORE_TEMPERATURE,
    TAUTOMER_RANKING_METHOD,
    TAUTOMER_RANKING_SCHEMA_VERSION,
    rank_tautomers_within_configurations,
    select_top_tautomer_in_configuration,
)


DEFAULT_INPUT = "data/processed/pka_molecule_microstate_network_stage2_predictions.csv"
DEFAULT_STAGE2_MODEL = (
    "data/processed/ml_models_experimental_only/stage2_network_context/"
    "stage2_network_context_model.pkl"
)
DEFAULT_OUTPUT = "data/processed/pka_molecule_microstate_network_stage3_predictions.csv"
DEFAULT_OUT_DIR = "data/processed/ml_models_experimental_only/stage3_microstates"
DEFAULT_STAGE1_OOF = (
    "data/processed/ml_models_experimental_only/stage1_intrinsic/"
    "scaffold_oof_predictions.csv"
)
DEFAULT_STAGE2_EVAL = (
    "data/processed/ml_models_experimental_only/stage2_network_context/"
    "scaffold_eval_predictions.csv"
)
DEFAULT_STAGE2_QUARANTINE = (
    "data/processed/ml_models_experimental_only/stage2_network_context/"
    "training_site_quarantine.csv"
)

STAGE3_OUTPUT_CONTRACT = {
    "schema_version": STAGE3_SCHEMA_VERSION,
    "primary_outputs": {
        "molecule_table": "one row per molecule with the complete populated microstate network",
        "site_table": "one row per detected ionizable site with its coupled marginal and confidence",
        "tautomer_table": (
            "one row per enumerated tautomer with rank and normalized heuristic weight "
            "within its deduplicated protonation configuration"
        ),
    },
    "pka_fields": {
        "stage2_predicted_macro_pka": (
            "site-associated macroscopic pKa reported by Stage 2; this is the primary "
            "quantity to compare with a scalar experimental pKa"
        ),
        "stage3_reconstructed_macro_pka": (
            "macroscopic pKa implied by the actual Stage 3 pairwise energy model"
        ),
        "stage3_one_body_pka": (
            "one-body coefficient in the pairwise free-energy model; not generally an "
            "observable macroscopic pKa in a multisite molecule"
        ),
        "stage3_contextual_edge_pka_at_dominant_background": (
            "microscopic edge pKa for this site with all other coordinates fixed to "
            "the dominant Stage 3 configuration at the requested pH"
        ),
        "stage3_contextual_edge_pka_min/max": (
            "range of microscopic edge pKas over every enumerated background"
        ),
        "stage3_empirical_pka_side_call_confidence": (
            "hierarchically calibrated probability that the scalar observed pKa lies "
            "on the side of the requested pH supporting the coupled Stage 3 site call; "
            "not direct microstate-state accuracy"
        ),
        "stage3_empirical_macro_pka_interval_90_low/high": (
            "90% residual interval around the deployment scalar macro-pKa prediction, "
            "using exact-label/family/global scaffold-held-out residual shrinkage"
        ),
    },
    "compatibility_aliases": {
        "stage3_effective_local_pka": "stage3_one_body_pka",
        "stage3_effective_local_pka_ci_low": "stage3_one_body_pka_interval_low",
        "stage3_effective_local_pka_ci_high": "stage3_one_body_pka_interval_high",
    },
    "uncertainty_note": (
        "one-body and probability intervals remain sensitivity intervals. The separate "
        "empirical pKa-side confidence uses scaffold-held-out signed residuals and is a "
        "proxy for state-call reliability, not direct microstate-state validation."
    ),
    "tautomer_note": (
        "The physically modeled dominant protonation configuration is selected first, "
        "then that configuration is enumerated once from all equivalent reference-node "
        "seeds and RDKit rule scores rank the deduplicated candidates. Their normalized "
        "weights are not physical free energies or calibrated tautomer probabilities."
    ),
}


def _json(value: object) -> str:
    """Serialize nested Stage 3 fields deterministically for stable artifacts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _truthy(value: object, default: bool = False) -> bool:
    """Interpret common CSV Boolean spellings, using ``default`` for missing values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _sha256(path: str) -> str:
    """Hash an artifact for end-to-end Stage 1/2/3 provenance checks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protonated_probability(local_pka: float, ph: float) -> float:
    """Evaluate the isolated-site Henderson-Hasselbalch protonated fraction."""
    exponent = max(-300.0, min(300.0, float(ph) - float(local_pka)))
    return float(1.0 / (1.0 + 10.0 ** exponent))


def _uncertainty_aware_state_confidence(
    probability_low: float,
    probability_high: float,
) -> float:
    """Score how decisively an entire probability interval lies on one side of 0.5."""
    if probability_low >= 0.5:
        return float(2.0 * (probability_low - 0.5))
    if probability_high <= 0.5:
        return float(2.0 * (0.5 - probability_high))
    return 0.0


def _contextual_edge_pka_for_background(
    edges: List[Dict],
    site_id: str,
    group_levels: Dict[str, int] | None,
) -> float | None:
    """Return a site's edge pKa with every other coordinate held at a background.

    An edge's ``other_site_forms`` stores coupling-group levels rather than acid/base
    labels.  Tautomer-expanded edges can duplicate the same thermodynamic transition,
    so matching values are averaged after verifying that they are numerically equal.
    """
    if not group_levels:
        return None
    matches = []
    for edge in edges:
        if str(edge.get("site_id")) != str(site_id):
            continue
        value = edge.get("stage3_contextual_edge_pka")
        if value is None:
            continue
        other_levels = edge.get("other_site_forms") or {}
        if all(
            str(group_id) in group_levels
            and int(group_levels[str(group_id)]) == int(level)
            for group_id, level in other_levels.items()
        ):
            matches.append(float(value))
    if not matches:
        return None
    if max(matches) - min(matches) > 1e-8:
        raise ValueError(
            f"Thermodynamically duplicate edges disagree for {site_id} in background "
            f"{group_levels}: {matches}"
        )
    return float(sum(matches) / len(matches))


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float | int]:
    """Summarize central and tail pKa errors, including a defined empty result."""
    if len(observed) == 0:
        return {
            "rows": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "median_abs_error": float("nan"),
            "p90_abs_error": float("nan"),
            "p95_abs_error": float("nan"),
            "max_abs_error": float("nan"),
            "count_abs_error_gt_2": 0,
            "count_abs_error_gt_3": 0,
            "count_abs_error_gt_5": 0,
        }
    error = np.abs(observed - predicted)
    return {
        "rows": int(len(observed)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "median_abs_error": float(np.median(error)),
        "p90_abs_error": float(np.quantile(error, 0.90)),
        "p95_abs_error": float(np.quantile(error, 0.95)),
        "max_abs_error": float(np.max(error)),
        "count_abs_error_gt_2": int(np.sum(error > 2.0)),
        "count_abs_error_gt_3": int(np.sum(error > 3.0)),
        "count_abs_error_gt_5": int(np.sum(error > 5.0)),
    }


def _validate_input(network_df: pd.DataFrame, stage2_model_path: str) -> Dict:
    """Validate network completeness fields and exact Stage 1/2 model provenance."""
    required_columns = {
        "molecule_id",
        "representative_input_smiles",
        "representative_atom_mapped_smiles",
        "network_confidence",
        "sites_json",
        "microstate_nodes_json",
        "microstate_edges_json",
        "protonation_state_enumeration_truncated",
        "incomplete_site_ids_json",
        "stage2_macro_pka_steps_json",
        "stage1_model_sha256",
        "stage2_model_sha256",
        "stage2_schema_version",
        "stage2_free_energy_model_json",
        "stage2_free_energy_schema_version",
        "stage2_free_energy_method",
        "marvin_values_used",
        "epik_values_used",
    }
    missing = required_columns - set(network_df.columns)
    if missing:
        raise ValueError(f"Stage 3 input missing canonical columns: {sorted(missing)}")
    if network_df["molecule_id"].isna().any() or network_df["molecule_id"].astype(str).duplicated().any():
        raise ValueError("Stage 3 input requires one non-null row per unique molecule_id")
    if not network_df["marvin_values_used"].eq(False).all() or not network_df["epik_values_used"].eq(False).all():
        raise ValueError("Stage 3 input contains prohibited Marvin or Epik labels")
    with open(stage2_model_path, "rb") as handle:
        bundle = pickle.load(handle)
    model_hash = _sha256(stage2_model_path)
    if set(network_df["stage2_model_sha256"].astype(str)) != {model_hash}:
        raise ValueError("Stage 3 input was generated by a different Stage 2 model")
    if set(network_df["stage2_schema_version"].astype(str)) != {
        str(bundle["stage2_schema_version"])
    }:
        raise ValueError("Stage 3 input and Stage 2 bundle schemas differ")
    if set(network_df["stage1_model_sha256"].astype(str)) != {
        str(bundle["stage1_model_sha256"])
    }:
        raise ValueError("Stage 3 input and Stage 2 bundle reference different Stage 1 models")
    if set(network_df["stage2_free_energy_schema_version"].astype(str)) != {
        STAGE2_FREE_ENERGY_SCHEMA_VERSION
    }:
        raise ValueError("Stage 3 input uses an incompatible free-energy schema")
    if set(network_df["stage2_free_energy_method"].astype(str)) != {
        STAGE2_FREE_ENERGY_METHOD
    }:
        raise ValueError("Stage 3 input uses an incompatible free-energy method")
    if str(bundle.get("stage2_free_energy_schema_version")) != STAGE2_FREE_ENERGY_SCHEMA_VERSION:
        raise ValueError("Stage 2 bundle does not declare the required free-energy schema")
    return bundle


def _heldout_composite_metrics(stage1_oof_path: str, stage2_eval_path: str) -> Dict:
    """Report Stage 1 and Stage 2 held-out pKa accuracy without implying one test set."""
    stage1 = pd.read_csv(stage1_oof_path, low_memory=False)
    stage2 = pd.read_csv(stage2_eval_path, low_memory=False)
    stage1_observed = stage1["experimental_anchor_pka"].to_numpy(dtype=float)
    stage1_predicted = stage1["pred_intrinsic_pka"].to_numpy(dtype=float)
    stage2_observed = stage2["experimental_anchor_pka"].to_numpy(dtype=float)
    stage2_prediction_column = (
        "stage2_supported_pairwise_free_energy_macro_pka"
        if "stage2_supported_pairwise_free_energy_macro_pka" in stage2.columns
        else "stage2_supported_projected_macro_pka"
    )
    stage2_predicted = stage2[stage2_prediction_column].to_numpy(dtype=float)
    return {
        "interpretation": (
            "internal aggregate of Stage 1 five-fold scaffold OOF single-site predictions "
            "and Stage 2 scaffold-validation multisite predictions; protocols differ and "
            "this is not an external test set"
        ),
        "single_site_stage1_oof": _metrics(stage1_observed, stage1_predicted),
        "multisite_stage2_scaffold_holdout": _metrics(stage2_observed, stage2_predicted),
        "combined_anchor_rows": _metrics(
            np.concatenate([stage1_observed, stage2_observed]),
            np.concatenate([stage1_predicted, stage2_predicted]),
        ),
    }


def apply_stage3(
    input_path: str,
    stage2_model_path: str,
    output_path: str,
    out_dir: str,
    ph: float = 7.4,
    regularization: float = 1e-3,
    stage1_oof_path: str = DEFAULT_STAGE1_OOF,
    stage2_eval_path: str = DEFAULT_STAGE2_EVAL,
    stage2_quarantine_path: str = DEFAULT_STAGE2_QUARANTINE,
    tautomer_score_temperature: float = DEFAULT_TAUTOMER_SCORE_TEMPERATURE,
    max_tautomers_per_configuration: int = DEFAULT_MAX_TAUTOMERS_PER_CONFIGURATION,
    max_molecules: int = 0,
) -> Tuple[pd.DataFrame, Dict]:
    """Apply the canonical coupled-population and tautomer-resolution stage.

    Stage 2 supplies a cycle-consistent pairwise free-energy model.  This
    function uses it to populate protonation configurations at ``ph``, computes
    site marginals and uncertainty diagnostics, and only then resolves a
    complete displayed microstate by configuration-level tautomer enumeration.

    ``max_tautomers_per_configuration`` limits retained, deduplicated candidates
    for a whole protonation configuration.  It is intentionally independent of
    the older per-node source cap stored in the network CSV.  Tautomer rule
    scores never feed back into the pKa model or configuration populations.

    ``max_molecules`` is a deterministic leading-row subset intended for smoke
    tests and benchmarks.  Zero processes the complete input.
    """
    if not math.isfinite(float(ph)):
        raise ValueError("Stage 3 pH must be finite")
    if (
        not math.isfinite(float(tautomer_score_temperature))
        or float(tautomer_score_temperature) <= 0.0
    ):
        raise ValueError("Tautomer score temperature must be finite and positive")
    if int(max_tautomers_per_configuration) < 1:
        raise ValueError("max_tautomers_per_configuration must be at least one")
    network_df = pd.read_csv(input_path, low_memory=False)
    stage2_bundle = _validate_input(network_df, stage2_model_path)
    input_sha256 = _sha256(input_path)
    stage2_model_sha256 = _sha256(stage2_model_path)
    if int(max_molecules) > 0:
        network_df = network_df.head(int(max_molecules)).copy()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    empirical_calibration = build_empirical_calibration(
        stage1_oof_path, stage2_eval_path
    )
    empirical_calibration_path = os.path.join(
        out_dir, "stage3_empirical_call_calibration.json"
    )
    with open(empirical_calibration_path, "w", encoding="utf-8") as handle:
        json.dump(empirical_calibration, handle, indent=2)
    empirical_calibration_sha256 = _sha256(empirical_calibration_path)
    quarantined_keys = set()
    if os.path.exists(stage2_quarantine_path):
        quarantine = pd.read_csv(stage2_quarantine_path, low_memory=False)
        quarantined_keys = set(zip(
            quarantine.get("molecule_id", pd.Series(dtype=str)).astype(str),
            quarantine.get("site_id", pd.Series(dtype=str)).astype(str),
        ))

    output_rows: List[Dict] = []
    site_rows: List[Dict] = []
    tautomer_rows: List[Dict] = []
    reconstruction_errors = []
    apparent_observed = []
    apparent_predicted = []
    complete_population_molecules = 0
    unavailable_population_molecules = 0
    optimizer_failures = 0
    robust_site_state_calls = 0
    uncertainty_crossing_site_states = 0
    dominant_microstates_changed_by_pair_coupling = 0
    molecules_with_site_calls_changed_by_pair_coupling = 0
    site_calls_changed_by_pair_coupling = 0
    tautomer_ranked_molecules = 0
    tautomer_truncated_configurations = 0
    tautomer_tied_top_configurations = 0

    for _, row in network_df.iterrows():
        sites = json.loads(str(row["sites_json"]))
        nodes = json.loads(str(row["microstate_nodes_json"]))
        edges = json.loads(str(row["microstate_edges_json"]))
        stage2_steps = sorted(
            json.loads(str(row["stage2_macro_pka_steps_json"])),
            key=lambda step: int(step["protonation_step"]),
        )
        site_lookup = {str(site["site_id"]): site for site in sites}
        all_site_ids = [str(site["site_id"]) for site in sites]
        limited = _truthy(row["protonation_state_enumeration_truncated"]) or bool(
            json.loads(str(row["incomplete_site_ids_json"]))
        )
        free_energy_model = json.loads(str(row["stage2_free_energy_model_json"]))

        if limited:
            # A partial graph has no defensible whole-molecule macro ladder.
            # Retain every detected site with its Stage 1 local estimate for
            # auditing, but do not run a partial inverse or report populations.
            ordered_site_ids = all_site_ids
            stage1_local = [
                float(site_lookup[site_id]["stage1_intrinsic_pka"])
                for site_id in ordered_site_ids
            ]
            stage2_macro: List[float] = []
            one_body_map = dict(zip(ordered_site_ids, stage1_local))
            pair_map = {}
            reconstructed_macro: List[float] = []
            free_energy_diagnostics = {
                "status": "unavailable_incomplete_or_truncated_network",
                "optimizer_success": True,
                "pair_parameter_count": 0,
                "macro_reconstruction_mae": None,
                "macro_reconstruction_max_abs_error": None,
            }
        else:
            ordered_site_ids = [
                str(step["rank_associated_site_id"]) for step in stage2_steps
            ]
            if len(ordered_site_ids) != len(set(ordered_site_ids)):
                raise ValueError(
                    f"Complete network {row['molecule_id']} maps more than one macro step "
                    "to the same site"
                )
            if set(ordered_site_ids) != set(all_site_ids):
                raise ValueError(
                    f"Complete network {row['molecule_id']} does not map every site "
                    "to exactly one macro step"
                )
            stage1_local = [
                float(site_lookup[site_id]["stage1_intrinsic_pka"])
                for site_id in ordered_site_ids
            ]
            stage2_macro = [
                float(step["stage2_predicted_macro_pka"]) for step in stage2_steps
            ]
            if str(free_energy_model.get("schema_version")) != STAGE2_FREE_ENERGY_SCHEMA_VERSION:
                raise ValueError(
                    f"Molecule {row['molecule_id']} has an incompatible Stage 2 free-energy model"
                )
            one_body_map = {
                str(term["site_id"]): float(term["one_body_pka"])
                for term in free_energy_model.get("one_body_terms", [])
            }
            if set(one_body_map) != set(all_site_ids):
                raise ValueError(
                    f"Molecule {row['molecule_id']} free-energy model does not cover every site"
                )
            pair_map = pair_coupling_map(free_energy_model.get("pair_couplings", []))
            free_energy_diagnostics = dict(free_energy_model.get("diagnostics", {}))
            optimizer_failures += int(
                not bool(free_energy_diagnostics.get("optimizer_success", False))
            )
            reconstructed_macro = macro_pka_values_from_free_energy(
                nodes, sites, one_body_map, pair_map
            )
            reconstruction_errors.extend(
                abs(predicted - target)
                for predicted, target in zip(reconstructed_macro, stage2_macro)
            )
        site_step_index = {
            site_id: idx for idx, site_id in enumerate(ordered_site_ids)
        } if not limited else {}

        reconstructed_steps = []
        for step, reconstructed in zip(stage2_steps, reconstructed_macro):
            reconstructed_steps.append({
                **step,
                "stage3_reconstructed_macro_pka": float(reconstructed),
                "stage3_reconstruction_error": float(
                    reconstructed - float(step["stage2_predicted_macro_pka"])
                ),
            })
        for edge in edges:
            contextual_pka = edge.get("stage2_contextual_edge_pka")
            edge["stage3_contextual_edge_pka"] = (
                float(contextual_pka) if contextual_pka is not None else None
            )
            edge["stage3_one_body_pka"] = float(one_body_map[str(edge["site_id"])])
            edge["stage3_edge_energy_model"] = (
                "cycle_consistent_symmetric_pairwise_free_energy"
                if not limited else "unavailable_incomplete_or_truncated_network"
            )

        population_summary: Dict[str, float | int | str]
        marginals: Dict[str, float]
        uncoupled_marginals: Dict[str, float] = {}
        uncoupled_dominant_node_id = None
        if limited:
            population_summary = {
                "status": "unavailable_incomplete_or_truncated_network",
                "population_sum": 0.0,
                "dominant_population": 0.0,
                "configuration_entropy": float("nan"),
                "effective_configuration_count": float("nan"),
            }
            marginals = {}
            unavailable_population_molecules += 1
        else:
            population_summary = annotate_pairwise_microstate_populations(
                nodes, sites, one_body_map, pair_map, ph
            )
            marginals = site_marginals_from_pairwise_model(
                nodes, sites, one_body_map, pair_map, ph
            )
            if pair_map:
                uncoupled_nodes = [dict(node) for node in nodes]
                annotate_pairwise_microstate_populations(
                    uncoupled_nodes, sites, one_body_map, {}, ph
                )
                uncoupled_marginals = site_marginals_from_pairwise_model(
                    nodes, sites, one_body_map, {}, ph
                )
                uncoupled_dominant_node_id = str(max(
                    uncoupled_nodes,
                    key=lambda node: (
                        float(node.get("stage3_configuration_population_at_ph", 0.0)),
                        str(node.get("node_id", "")),
                    ),
                )["node_id"])
            complete_population_molecules += int(
                population_summary["status"] == "complete_pairwise_coupled_population"
            )

        # Protonation populations above are final.  The following block only
        # enumerates/ranks drawings within each populated configuration.
        tautomer_summary: Dict = {
            "status": "unavailable_incomplete_or_truncated_network",
            "ranking_method": TAUTOMER_RANKING_METHOD,
            "ranking_schema_version": TAUTOMER_RANKING_SCHEMA_VERSION,
            "ranked_tautomer_count": 0,
            "complete_microstate_heuristic_population_sum": 0.0,
        }
        dominant_tautomer = None
        molecule_tautomer_rows: List[Dict] = []
        if marginals and nodes:
            molecule_tautomer_rows, tautomer_summary, dominant_tautomer = (
                rank_tautomers_within_configurations(
                    nodes,
                    molecule_id=str(row["molecule_id"]),
                    score_temperature=float(tautomer_score_temperature),
                    max_tautomers_per_configuration=int(
                        max_tautomers_per_configuration
                    ),
                )
            )
            # Preserve the network builder's node-level cap as provenance.  It
            # is not the active Stage 3 configuration-level retention limit.
            source_tautomer_cap = row.get("tautomer_cap_per_protonation_state", 0)
            source_tautomer_cap = (
                0 if pd.isna(source_tautomer_cap) else int(source_tautomer_cap)
            )
            for tautomer_row in molecule_tautomer_rows:
                tautomer_row["population_ph"] = float(ph)
                tautomer_row[
                    "source_tautomer_cap_per_protonation_state"
                ] = source_tautomer_cap
                tautomer_row["stage3_max_tautomers_per_configuration"] = int(
                    max_tautomers_per_configuration
                )
            tautomer_rows.extend(molecule_tautomer_rows)
            tautomer_ranked_molecules += int(bool(molecule_tautomer_rows))
            tautomer_truncated_configurations += int(
                tautomer_summary.get("truncated_configuration_count", 0)
            )
            tautomer_tied_top_configurations += int(
                tautomer_summary.get("tied_top_configuration_count", 0)
            )

        dominant_node = None
        if marginals and nodes:
            dominant_node = max(
                nodes,
                key=lambda node: (
                    float(node.get("stage3_configuration_population_at_ph", 0.0)),
                    str(node.get("node_id", "")),
                ),
            )
            dominant_microstates_changed_by_pair_coupling += int(
                uncoupled_dominant_node_id is not None
                and str(dominant_node["node_id"]) != uncoupled_dominant_node_id
            )
        # Enforce the physical ordering: choose the dominant protonation
        # configuration first, then its rank-1 tautomer.  A rule score can
        # never promote a tautomer from a less populated configuration.
        if dominant_node and molecule_tautomer_rows:
            dominant_configuration_id = str(
                dominant_node.get("stage3_tautomer_configuration_id", "")
            )
            selected_tautomer = select_top_tautomer_in_configuration(
                molecule_tautomer_rows, dominant_configuration_id
            )
            if selected_tautomer is not None:
                dominant_tautomer = selected_tautomer
                tautomer_summary["selected_complete_microstate_policy"] = (
                    "dominant_protonation_configuration_then_top_rdkit_ranked_tautomer"
                )
                tautomer_summary[
                    "selected_complete_microstate_heuristic_population"
                ] = float(
                    dominant_tautomer[
                        "complete_microstate_heuristic_population_at_ph"
                    ]
                )
        node_lookup = {str(node.get("node_id", "")): node for node in nodes}
        dominant_complete_node = (
            node_lookup.get(str(dominant_tautomer["origin_node_id"]))
            if dominant_tautomer else dominant_node
        )
        molecule_site_call_changes = 0
        for site_id in ordered_site_ids:
            effective_pka = float(one_body_map[site_id])
            site = site_lookup[site_id]
            probability = marginals.get(site_id)
            uncoupled_probability = uncoupled_marginals.get(site_id)
            pair_changes_site_call = (
                probability is not None
                and uncoupled_probability is not None
                and (probability >= 0.5) != (uncoupled_probability >= 0.5)
            )
            molecule_site_call_changes += int(pair_changes_site_call)
            if limited:
                predicted_macro = None
                reconstructed_site_macro = None
                stage1_pka = float(site["stage1_intrinsic_pka"])
                macro_low = float(site.get(
                    "stage1_predicted_pka_ci_low", stage1_pka - 3.0
                ))
                macro_high = float(site.get(
                    "stage1_predicted_pka_ci_high", stage1_pka + 3.0
                ))
                macro_half_width = max(
                    stage1_pka - macro_low,
                    macro_high - stage1_pka,
                    0.5,
                )
                inverse_gap = 0.0
            else:
                step_idx = site_step_index[site_id]
                predicted_macro = float(stage2_macro[step_idx])
                reconstructed_site_macro = float(reconstructed_macro[step_idx])
                macro_low = float(site.get(
                    "stage2_predicted_macro_pka_ci_low", predicted_macro - 3.0
                ))
                macro_high = float(site.get(
                    "stage2_predicted_macro_pka_ci_high", predicted_macro + 3.0
                ))
                macro_half_width = max(
                    predicted_macro - macro_low,
                    macro_high - predicted_macro,
                    0.5,
                )
                inverse_gap = abs(reconstructed_site_macro - predicted_macro)
            effective_half_width = macro_half_width + inverse_gap
            effective_low = float(effective_pka - effective_half_width)
            effective_high = float(effective_pka + effective_half_width)
            if probability is None:
                probability_low = None
                probability_high = None
            else:
                low_model = dict(one_body_map)
                high_model = dict(one_body_map)
                low_model[site_id] = effective_low
                high_model[site_id] = effective_high
                probability_low = site_marginals_from_pairwise_model(
                    nodes, sites, low_model, pair_map, ph
                )[site_id]
                probability_high = site_marginals_from_pairwise_model(
                    nodes, sites, high_model, pair_map, ph
                )[site_id]
            uncertainty_confidence = _uncertainty_aware_state_confidence(
                probability_low, probability_high
            ) if probability is not None else 0.0
            uncertainty_crosses = (
                None
                if probability is None
                else probability_low <= 0.5 <= probability_high
            )
            robust_site_state_calls += int(
                probability is not None and not uncertainty_crosses
            )
            uncertainty_crossing_site_states += int(uncertainty_crosses is True)
            predicted_site_form = (
                "acid_form" if probability is not None and probability >= 0.5
                else "base_form" if probability is not None else "unavailable"
            )
            calibration_source = (
                "stage1_single_site" if len(all_site_ids) == 1 else "stage2_multisite"
            )
            calibration_point_pka = (
                float(predicted_macro)
                if predicted_macro is not None
                else float(site["stage1_intrinsic_pka"])
            )
            empirical_site_calibration = (
                calibrate_site_call(
                    empirical_calibration,
                    source_name=calibration_source,
                    site_label=str(site["label"]),
                    site_family=str(site["family"]),
                    predicted_pka=calibration_point_pka,
                    ph=float(ph),
                    predicted_site_form=predicted_site_form,
                )
                if probability is not None else None
            )
            empirical_call_confidence = (
                float(empirical_site_calibration[
                    "predicted_site_form_pka_side_confidence"
                ])
                if empirical_site_calibration is not None else 0.0
            )
            empirical_centered_evidence = max(
                0.0, min(1.0, (2.0 * empirical_call_confidence) - 1.0)
            )
            conditional_state_decisiveness = (
                float(abs((2.0 * probability) - 1.0))
                if probability is not None else 0.0
            )
            local_support = str(site.get("local_pka_confidence", "none"))
            applicability_factor = (
                1.0 if local_support.startswith("high")
                else 0.80 if local_support.startswith("medium")
                else 0.55 if local_support.startswith("low")
                else 0.30
            )
            reconstruction_factor = 0.0 if limited else math.exp(-inverse_gap)
            identifiability_status = str(free_energy_diagnostics.get(
                "pair_coupling_identifiability", "unknown"
            ))
            identifiability_factor = (
                0.65
                if identifiability_status.startswith("regularized_prior_dependent")
                else 1.0
            )
            overall_confidence = float(
                empirical_centered_evidence
                * conditional_state_decisiveness
                * applicability_factor
                * reconstruction_factor
                * identifiability_factor
            )
            overall_tier = (
                "high" if overall_confidence >= 0.70
                else "medium" if overall_confidence >= 0.40
                else "low" if overall_confidence >= 0.10
                else "very_low"
            )
            site_contextual_edge_pkas = [
                float(edge["stage3_contextual_edge_pka"])
                for edge in edges
                if str(edge["site_id"]) == site_id
                and edge.get("stage3_contextual_edge_pka") is not None
            ]
            dominant_contextual_edge_pka = _contextual_edge_pka_for_background(
                edges,
                site_id,
                (
                    dominant_complete_node.get("group_levels")
                    if dominant_complete_node else None
                ),
            )
            site.update({
                "stage3_one_body_pka": float(effective_pka),
                "stage3_one_body_pka_interval_low": effective_low,
                "stage3_one_body_pka_interval_high": effective_high,
                "stage3_effective_local_pka": float(effective_pka),
                "stage3_effective_local_pka_semantics": (
                    "deprecated_compatibility_alias_of_stage3_one_body_pka"
                ),
                "stage3_local_shift_from_stage1": float(
                    effective_pka - float(site["stage1_intrinsic_pka"])
                ),
                "stage3_local_pka_method": (
                    "stage1_fallback_incomplete_network_no_macro_inverse"
                    if limited else STAGE3_LOCAL_PKA_METHOD
                ),
                "stage3_protonated_probability_at_ph": (
                    float(probability) if probability is not None else None
                ),
                "stage3_predicted_site_form_at_ph": predicted_site_form,
                "stage3_site_state_decision_confidence": (
                    float(abs((2.0 * probability) - 1.0)) if probability is not None else None
                ),
                "stage3_effective_local_pka_ci_low": effective_low,
                "stage3_effective_local_pka_ci_high": effective_high,
                "stage3_contextual_edge_pka_min": (
                    min(site_contextual_edge_pkas) if site_contextual_edge_pkas else None
                ),
                "stage3_contextual_edge_pka_max": (
                    max(site_contextual_edge_pkas) if site_contextual_edge_pkas else None
                ),
                "stage3_contextual_edge_pka_at_dominant_background": (
                    dominant_contextual_edge_pka
                ),
                "stage3_protonated_probability_ci_low": probability_low,
                "stage3_protonated_probability_ci_high": probability_high,
                "stage3_uncertainty_crosses_state_boundary": uncertainty_crosses,
                "stage3_uncertainty_aware_site_state_confidence": uncertainty_confidence,
                "stage3_empirical_pka_side_call_confidence": (
                    empirical_call_confidence
                    if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_pka_side_centered_evidence": (
                    empirical_centered_evidence
                    if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_probability_observed_pka_at_or_above_ph": (
                    empirical_site_calibration[
                        "probability_observed_pka_at_or_above_ph"
                    ] if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_macro_pka_interval_90_low": (
                    empirical_site_calibration["empirical_macro_pka_interval_90_low"]
                    if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_macro_pka_interval_90_high": (
                    empirical_site_calibration["empirical_macro_pka_interval_90_high"]
                    if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_calibration_source": (
                    empirical_site_calibration["source"]
                    if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_calibration_scope": (
                    empirical_site_calibration["scope"]
                    if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_calibration_label_rows": (
                    empirical_site_calibration["label_rows"]
                    if empirical_site_calibration is not None else 0
                ),
                "stage3_empirical_calibration_family_rows": (
                    empirical_site_calibration["family_rows"]
                    if empirical_site_calibration is not None else 0
                ),
                "stage3_empirical_calibration_method": EMPIRICAL_CALIBRATION_METHOD,
                "stage3_overall_site_state_confidence": overall_confidence,
                "stage3_overall_site_state_confidence_tier": overall_tier,
                "stage3_confidence_components": {
                    "conditional_state_decisiveness": conditional_state_decisiveness,
                    "pka_interval_robustness": uncertainty_confidence,
                    "empirical_pka_side_call_confidence": empirical_call_confidence,
                    "empirical_pka_side_centered_evidence": empirical_centered_evidence,
                    "stage1_applicability_factor": applicability_factor,
                    "free_energy_reconstruction_factor": reconstruction_factor,
                    "pair_coupling_identifiability_factor": identifiability_factor,
                    "network_complete": not limited,
                },
                "stage3_pair_coupling_identifiability": identifiability_status,
                "stage3_uncoupled_protonated_probability_at_ph": uncoupled_probability,
                "stage3_site_call_changed_by_pair_coupling": pair_changes_site_call,
                "stage3_population_status": population_summary["status"],
            })
            anchor = site.get("experimental_anchor_pka")
            site["stage3_reconstructed_macro_pka"] = reconstructed_site_macro
            if (
                anchor is not None
                and reconstructed_site_macro is not None
                and (str(row["molecule_id"]), site_id) not in quarantined_keys
            ):
                apparent_observed.append(float(anchor))
                apparent_predicted.append(reconstructed_site_macro)
            site_rows.append({
                "molecule_id": str(row["molecule_id"]),
                "input_smiles": str(row["representative_input_smiles"]),
                "atom_mapped_smiles": str(row["representative_atom_mapped_smiles"]),
                "site_id": site_id,
                "site_label": str(site["label"]),
                "site_family": str(site["family"]),
                # This is the selected site's form in the representative source
                # drawing, not an experimental equilibrium-state observation.
                "input_member_form": site.get("input_member_form"),
                "site_atom_maps_json": _json(site.get("atom_maps", [])),
                "site_center_maps_json": _json(site.get("center_maps", [])),
                "network_confidence": str(row["network_confidence"]),
                "stage1_intrinsic_pka": float(site["stage1_intrinsic_pka"]),
                "stage2_predicted_macro_pka": predicted_macro,
                "stage3_reconstructed_macro_pka": reconstructed_site_macro,
                "stage3_macro_reconstruction_abs_error": (
                    float(abs(reconstructed_site_macro - predicted_macro))
                    if reconstructed_site_macro is not None and predicted_macro is not None
                    else None
                ),
                "stage2_applied": bool(site.get("stage2_applied", False)),
                "stage2_prediction_confidence": site.get("stage2_prediction_confidence"),
                "stage2_pair_coupling_identifiability": identifiability_status,
                "stage3_one_body_pka": float(effective_pka),
                "stage3_one_body_pka_interval_low": effective_low,
                "stage3_one_body_pka_interval_high": effective_high,
                "stage3_effective_local_pka": float(effective_pka),
                "stage3_effective_local_pka_semantics": (
                    "deprecated_compatibility_alias_of_stage3_one_body_pka"
                ),
                "stage3_contextual_edge_pka_min": (
                    min(site_contextual_edge_pkas) if site_contextual_edge_pkas else None
                ),
                "stage3_contextual_edge_pka_max": (
                    max(site_contextual_edge_pkas) if site_contextual_edge_pkas else None
                ),
                "stage3_contextual_edge_pka_at_dominant_background": (
                    dominant_contextual_edge_pka
                ),
                "stage3_scalar_experimental_comparison_pka": predicted_macro,
                "stage3_scalar_experimental_comparison_pka_semantics": (
                    "site_associated_stage2_macroscopic_pka"
                    if predicted_macro is not None else "unavailable_incomplete_network"
                ),
                "stage3_protonated_probability_at_ph": probability,
                "stage3_uncoupled_protonated_probability_at_ph": uncoupled_probability,
                "stage3_site_call_changed_by_pair_coupling": pair_changes_site_call,
                "stage3_predicted_site_form_at_ph": site["stage3_predicted_site_form_at_ph"],
                "stage3_site_state_decision_confidence": site["stage3_site_state_decision_confidence"],
                "stage3_effective_local_pka_ci_low": effective_low,
                "stage3_effective_local_pka_ci_high": effective_high,
                "stage3_protonated_probability_ci_low": probability_low,
                "stage3_protonated_probability_ci_high": probability_high,
                "stage3_uncertainty_crosses_state_boundary": uncertainty_crosses,
                "stage3_uncertainty_aware_site_state_confidence": uncertainty_confidence,
                "stage3_empirical_pka_side_call_confidence": (
                    empirical_call_confidence
                    if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_pka_side_centered_evidence": (
                    empirical_centered_evidence
                    if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_probability_observed_pka_at_or_above_ph": (
                    empirical_site_calibration[
                        "probability_observed_pka_at_or_above_ph"
                    ] if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_macro_pka_interval_90_low": (
                    empirical_site_calibration["empirical_macro_pka_interval_90_low"]
                    if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_macro_pka_interval_90_high": (
                    empirical_site_calibration["empirical_macro_pka_interval_90_high"]
                    if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_calibration_source": (
                    empirical_site_calibration["source"]
                    if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_calibration_scope": (
                    empirical_site_calibration["scope"]
                    if empirical_site_calibration is not None else None
                ),
                "stage3_empirical_calibration_label_rows": (
                    empirical_site_calibration["label_rows"]
                    if empirical_site_calibration is not None else 0
                ),
                "stage3_empirical_calibration_family_rows": (
                    empirical_site_calibration["family_rows"]
                    if empirical_site_calibration is not None else 0
                ),
                "stage3_empirical_calibration_method": EMPIRICAL_CALIBRATION_METHOD,
                "stage3_empirical_calibration_sha256": empirical_calibration_sha256,
                "stage3_overall_site_state_confidence": overall_confidence,
                "stage3_overall_site_state_confidence_tier": overall_tier,
                "stage1_local_pka_confidence": local_support,
                "stage1_applicability_domain": site.get("stage1_applicability_domain"),
                "stage1_exact_label_training_rows": site.get("stage1_exact_label_training_rows"),
                "stage1_nearest_same_label_tanimoto": site.get("stage1_nearest_same_label_tanimoto"),
                "experimental_anchor_evidence_tier": site.get("experimental_anchor_evidence_tier"),
                "experimental_anchor_pka": anchor,
                "population_ph": float(ph),
                "population_status": population_summary["status"],
                "population_model": population_summary.get("population_model"),
                "dominant_atom_mapped_smiles": (
                    dominant_tautomer.get("atom_mapped_smiles")
                    if dominant_tautomer else None
                ),
                "dominant_configuration_population": (
                    dominant_node.get("stage3_configuration_population_at_ph")
                    if dominant_node else None
                ),
                "dominant_complete_microstate_heuristic_population": (
                    dominant_tautomer.get(
                        "complete_microstate_heuristic_population_at_ph"
                    ) if dominant_tautomer else None
                ),
                "dominant_tautomer_ranking_semantics": (
                    dominant_tautomer.get("ranking_semantics")
                    if dominant_tautomer else None
                ),
            })

        site_calls_changed_by_pair_coupling += int(molecule_site_call_changes)
        molecules_with_site_calls_changed_by_pair_coupling += int(
            molecule_site_call_changes > 0
        )

        output = row.to_dict()
        output.update({
            "stage3_schema_version": STAGE3_SCHEMA_VERSION,
            "stage3_input_sha256": input_sha256,
            "stage3_stage2_model_sha256": stage2_model_sha256,
            "stage3_local_pka_method": STAGE3_LOCAL_PKA_METHOD,
            "stage3_regularization": None,
            "stage3_deprecated_regularization_argument": float(regularization),
            "stage3_population_ph": float(ph),
            "stage3_population_status": population_summary["status"],
            "stage3_complete_microstate_available": bool(dominant_node is not None),
            "stage3_population_model": (
                population_summary.get(
                    "population_model",
                    "unavailable_incomplete_or_truncated_network",
                )
            ),
            "stage3_population_caveat": (
                "Stage 2 pair couplings are a regularized closure of sparse macroscopic "
                "pKa evidence, not uniquely measured interaction energies. Tautomer "
                "weights use RDKit rule scores and are ranking aids, not learned or "
                "physical relative energies."
            ),
            "stage3_empirical_calibration_schema_version": (
                EMPIRICAL_CALIBRATION_SCHEMA_VERSION
            ),
            "stage3_empirical_calibration_method": EMPIRICAL_CALIBRATION_METHOD,
            "stage3_empirical_calibration_sha256": empirical_calibration_sha256,
            "stage3_tautomer_ranking_schema_version": TAUTOMER_RANKING_SCHEMA_VERSION,
            "stage3_tautomer_ranking_method": TAUTOMER_RANKING_METHOD,
            "stage3_tautomer_score_temperature_rule_units": float(
                tautomer_score_temperature
            ),
            "stage3_max_tautomers_per_configuration": int(
                max_tautomers_per_configuration
            ),
            "stage3_free_energy_diagnostics_json": _json(free_energy_diagnostics),
            "stage3_population_summary_json": _json(population_summary),
            "stage3_tautomer_ranking_summary_json": _json(tautomer_summary),
            "sites_json": _json(sites),
            "stage3_microstate_nodes_json": _json(nodes),
            "stage3_microstate_edges_json": _json(edges),
            "stage3_macro_pka_steps_json": _json(reconstructed_steps),
            "stage3_dominant_node_id": (
                dominant_complete_node.get("node_id")
                if dominant_complete_node else None
            ),
            "stage3_dominant_configuration_node_id": (
                dominant_node.get("node_id") if dominant_node else None
            ),
            "stage3_dominant_atom_mapped_smiles": (
                dominant_tautomer.get("atom_mapped_smiles")
                if dominant_tautomer else None
            ),
            "stage3_dominant_configuration_reference_atom_mapped_smiles": (
                dominant_node.get("reference_atom_mapped_smiles")
                if dominant_node else None
            ),
            "stage3_dominant_formal_charge": (
                dominant_complete_node.get("formal_charge")
                if dominant_complete_node else None
            ),
            "stage3_dominant_population": (
                dominant_node.get("stage3_configuration_population_at_ph") if dominant_node else None
            ),
            "stage3_dominant_node_population": (
                dominant_complete_node.get("stage3_node_population_at_ph")
                if dominant_complete_node else None
            ),
            "stage3_dominant_complete_microstate_heuristic_population": (
                dominant_tautomer.get(
                    "complete_microstate_heuristic_population_at_ph"
                ) if dominant_tautomer else None
            ),
            "stage3_dominant_tautomer_rank_within_configuration": (
                dominant_tautomer.get("tautomer_rank_within_configuration")
                if dominant_tautomer else None
            ),
            "stage3_dominant_rdkit_tautomer_score": (
                dominant_tautomer.get("rdkit_tautomer_score")
                if dominant_tautomer else None
            ),
            "stage3_dominant_tautomer_heuristic_conditional_weight": (
                dominant_tautomer.get("tautomer_heuristic_conditional_weight")
                if dominant_tautomer else None
            ),
            "stage3_dominant_tautomer_ranking_semantics": (
                dominant_tautomer.get("ranking_semantics")
                if dominant_tautomer else None
            ),
            "stage3_dominant_population_gap": population_summary.get(
                "dominant_population_gap"
            ),
            "stage3_configuration_entropy": population_summary.get(
                "configuration_entropy"
            ),
            "stage3_effective_configuration_count": population_summary.get(
                "effective_configuration_count"
            ),
        })
        output_rows.append(output)

    result = pd.DataFrame(output_rows)
    site_table = pd.DataFrame(site_rows)
    tautomer_table = pd.DataFrame(tautomer_rows)
    result.to_csv(output_path, index=False)
    site_output_path = os.path.join(out_dir, "stage3_site_predictions.csv")
    tautomer_output_path = os.path.join(out_dir, "stage3_ranked_tautomers.csv")
    site_table.to_csv(site_output_path, index=False)
    tautomer_table.to_csv(tautomer_output_path, index=False)
    with open(os.path.join(out_dir, "stage3_output_schema.json"), "w", encoding="utf-8") as handle:
        json.dump(STAGE3_OUTPUT_CONTRACT, handle, indent=2)
    coupling_changed_sites = site_table[
        site_table.get(
            "stage3_site_call_changed_by_pair_coupling",
            pd.Series(False, index=site_table.index),
        ).fillna(False).astype(bool)
    ].copy()
    coupling_changed_sites.to_csv(
        os.path.join(out_dir, "stage3_coupling_changed_site_calls.csv"), index=False
    )

    reconstruction_array = np.asarray(reconstruction_errors, dtype=float)
    reconstruction_summary = (
        {
            "steps": int(len(reconstruction_array)),
            "mae": float(np.mean(reconstruction_array)),
            "p95_abs_error": float(np.quantile(reconstruction_array, 0.95)),
            "max_abs_error": float(np.max(reconstruction_array)),
        }
        if len(reconstruction_array)
        else {
            "steps": 0,
            "mae": float("nan"),
            "p95_abs_error": float("nan"),
            "max_abs_error": float("nan"),
        }
    )
    empirical_confidence_array = pd.to_numeric(
        site_table.get(
            "stage3_empirical_pka_side_call_confidence",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    ).dropna().to_numpy(dtype=float)
    empirical_confidence_summary = {
        "site_calls": int(len(empirical_confidence_array)),
        "minimum": (
            float(np.min(empirical_confidence_array))
            if len(empirical_confidence_array) else None
        ),
        "median": (
            float(np.median(empirical_confidence_array))
            if len(empirical_confidence_array) else None
        ),
        "p10": (
            float(np.quantile(empirical_confidence_array, 0.10))
            if len(empirical_confidence_array) else None
        ),
        "p90": (
            float(np.quantile(empirical_confidence_array, 0.90))
            if len(empirical_confidence_array) else None
        ),
        "conflicts_below_half": int(np.sum(empirical_confidence_array < 0.5)),
    }
    report = {
        "stage3_schema_version": STAGE3_SCHEMA_VERSION,
        "local_pka_method": STAGE3_LOCAL_PKA_METHOD,
        "population_ph": float(ph),
        "regularization": None,
        "deprecated_regularization_argument": float(regularization),
        "molecule_rows": int(len(result)),
        "site_rows": int(len(site_table)),
        "ranked_tautomer_rows": int(len(tautomer_table)),
        "tautomer_ranked_molecules": int(tautomer_ranked_molecules),
        "tautomer_truncated_configurations": int(tautomer_truncated_configurations),
        "tautomer_tied_top_configurations": int(tautomer_tied_top_configurations),
        "tautomer_ranking_schema_version": TAUTOMER_RANKING_SCHEMA_VERSION,
        "tautomer_ranking_method": TAUTOMER_RANKING_METHOD,
        "tautomer_score_temperature_rule_units": float(tautomer_score_temperature),
        "max_tautomers_per_configuration": int(max_tautomers_per_configuration),
        "tautomer_enumeration_scope": "deduplicated_protonation_configuration",
        "tautomer_ranking_semantics": (
            "rank_only_not_physical_free_energy_or_calibrated_tautomer_probability"
        ),
        "complete_population_molecules": int(complete_population_molecules),
        "unavailable_incomplete_network_molecules": int(unavailable_population_molecules),
        "stage2_free_energy_optimizer_failures": int(optimizer_failures),
        "molecules_with_pair_parameters": int(
            pd.to_numeric(
                result.get("stage2_pair_parameter_count", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).gt(0).sum()
        ),
        "active_pair_parameters": int(
            pd.to_numeric(
                result.get("stage2_active_pair_parameter_count", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0).sum()
        ),
        "pair_coupling_identifiability": (
            "regularized_prior_dependent_not_directly_measured"
        ),
        "dominant_microstates_changed_by_pair_coupling_at_ph": int(
            dominant_microstates_changed_by_pair_coupling
        ),
        "molecules_with_site_calls_changed_by_pair_coupling_at_ph": int(
            molecules_with_site_calls_changed_by_pair_coupling
        ),
        "site_calls_changed_by_pair_coupling_at_ph": int(
            site_calls_changed_by_pair_coupling
        ),
        "robust_site_state_calls_pka_interval_does_not_cross_half": int(robust_site_state_calls),
        "site_state_intervals_crossing_half": int(uncertainty_crossing_site_states),
        "macro_reconstruction_vs_stage2": reconstruction_summary,
        "apparent_deployment_anchor_fit_not_holdout": _metrics(
            np.asarray(apparent_observed, dtype=float),
            np.asarray(apparent_predicted, dtype=float),
        ),
        "heldout_pka_performance": _heldout_composite_metrics(
            stage1_oof_path, stage2_eval_path
        ),
        "empirical_pka_side_calibration": {
            "schema_version": EMPIRICAL_CALIBRATION_SCHEMA_VERSION,
            "method": EMPIRICAL_CALIBRATION_METHOD,
            "artifact_sha256": empirical_calibration_sha256,
            "diagnostics": calibration_diagnostics(empirical_calibration, ph),
            "deployment_confidence_summary": empirical_confidence_summary,
            "status": (
                "calibrated_from_scaffold_heldout_pka_residuals_not_direct_"
                "experimental_microstate_states"
            ),
        },
        "microstate_state_performance": {
            "experimental_state_labels": 0,
            "status": "not_evaluable_from_scalar_pka_labels",
        },
        "stage1_model_sha256": str(stage2_bundle["stage1_model_sha256"]),
        "stage2_model_sha256": stage2_model_sha256,
        "input_sha256": input_sha256,
        "output_sha256": _sha256(output_path),
        "site_output_sha256": _sha256(site_output_path),
        "tautomer_output_sha256": _sha256(tautomer_output_path),
        "empirical_calibration_artifact_sha256": empirical_calibration_sha256,
        "output_contract": STAGE3_OUTPUT_CONTRACT,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return result, report


def parse_args() -> argparse.Namespace:
    """Parse Stage 3 population, calibration, and tautomer-enumeration settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--stage2-model", default=DEFAULT_STAGE2_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument(
        "--regularization", type=float, default=1e-3,
        help="Deprecated compatibility argument; Stage 2 now emits the free-energy model.",
    )
    parser.add_argument("--stage1-oof", default=DEFAULT_STAGE1_OOF)
    parser.add_argument("--stage2-eval", default=DEFAULT_STAGE2_EVAL)
    parser.add_argument("--stage2-quarantine", default=DEFAULT_STAGE2_QUARANTINE)
    parser.add_argument(
        "--tautomer-score-temperature",
        type=float,
        default=DEFAULT_TAUTOMER_SCORE_TEMPERATURE,
        help=(
            "Softmax scale in RDKit tautomer-score units; affects heuristic rank "
            "weights only and has no thermodynamic temperature meaning."
        ),
    )
    parser.add_argument(
        "--max-tautomers-per-configuration",
        type=int,
        default=DEFAULT_MAX_TAUTOMERS_PER_CONFIGURATION,
        help=(
            "Maximum deduplicated tautomer candidates retained per protonation "
            "configuration after enumerating all equivalent reference-node seeds."
        ),
    )
    parser.add_argument("--max-molecules", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run Stage 3 from the CLI and print its machine-readable report."""
    args = parse_args()
    _, report = apply_stage3(
        input_path=args.input,
        stage2_model_path=args.stage2_model,
        output_path=args.output,
        out_dir=args.out_dir,
        ph=args.ph,
        regularization=args.regularization,
        stage1_oof_path=args.stage1_oof,
        stage2_eval_path=args.stage2_eval,
        stage2_quarantine_path=args.stage2_quarantine,
        tautomer_score_temperature=args.tautomer_score_temperature,
        max_tautomers_per_configuration=args.max_tautomers_per_configuration,
        max_molecules=args.max_molecules,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
