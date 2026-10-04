#!/usr/bin/env python3
"""Public pKₐte API and CLI for predicting a previously unseen molecule.

Examples
--------
Python::

    from pkate import predict

    result = predict("NCCCC(=O)O", ph=7.4)
    print(result.dominant_microstate_smiles)
    print(result.site_predictions)

Command line::

    python -m pkate "NCCCC(=O)O" --ph 7.4

The function constructs a new atom-mapped protonation network, predicts every
detected transition with Stage 1, applies the trained Stage 2 context model,
and runs the canonical Stage 3 population and tautomer inference.  No
experimental pKₐ supplied with the query is required or consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import inchi


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pkate-matplotlib")
)

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    # The research modules predate the public API and are intentionally kept in
    # scripts/ for now.  Resolve them relative to this file, never the caller's
    # working directory, so renaming or moving the repository remains safe.
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_molecule_microstate_network_dataset import (  # noqa: E402
    MOLECULE_NETWORK_DATASET_SCHEMA_VERSION,
    coupled_network_thermodynamics,
    stage1_local_prediction_for_site,
)
from functional_group_pka_analysis import (  # noqa: E402
    ACIDIC_FAMILIES,
    PAIR_TYPE_FAMILY_FORMS,
    expand_pair_site_transitions,
    training_group_label,
)
from microstate_enumerator import (  # noqa: E402
    DEFAULT_MAX_PROTONATION_STATES,
    ensure_heavy_atom_maps,
    enumerate_joint_protonation_network,
    mapped_smiles,
)
from pka_evidence_policy import (  # noqa: E402
    unresolved_ionizable_contexts,
)
from stage1_network_inference import (  # noqa: E402
    load_stage1_bundle,
    predict_stage1_tasks,
    stage1_applicability,
)
from stage2_apply_network_context import apply_stage2  # noqa: E402
from stage3_apply_microstate_inference import (  # noqa: E402
    DEFAULT_MAX_TAUTOMERS_PER_CONFIGURATION,
    DEFAULT_TAUTOMER_SCORE_TEMPERATURE,
    apply_stage3,
)
from substructure_match import (  # noqa: E402
    FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION,
    find_sites_with_metadata,
    resolve_overlapping_sites,
)


@dataclass(frozen=True)
class ModelArtifacts:
    """Paths to the fitted model and held-out calibration artifacts."""

    stage1_model: Path
    stage1_metrics: Path
    stage1_oof: Path
    stage2_model: Path
    stage2_eval: Path
    stage2_quarantine: Path

    @classmethod
    def defaults(cls, project_root: Path = PROJECT_ROOT) -> "ModelArtifacts":
        """Return the canonical experimental-only artifact locations."""
        root = Path(project_root)
        model_root = root / "data" / "processed" / "ml_models_experimental_only"
        return cls(
            stage1_model=model_root / "stage1_intrinsic" / "stage1_intrinsic_model.pkl",
            stage1_metrics=model_root / "stage1_intrinsic" / "metrics.json",
            stage1_oof=model_root / "stage1_intrinsic" / "scaffold_oof_predictions.csv",
            stage2_model=(
                model_root
                / "stage2_network_context"
                / "stage2_network_context_model.pkl"
            ),
            stage2_eval=(
                model_root / "stage2_network_context" / "scaffold_eval_predictions.csv"
            ),
            stage2_quarantine=(
                model_root / "stage2_network_context" / "training_site_quarantine.csv"
            ),
        )

    def validate(self) -> None:
        """Fail with one actionable message when deployment artifacts are absent."""
        missing = [str(path) for path in asdict(self).values() if not Path(path).is_file()]
        if missing:
            joined = "\n  - ".join(missing)
            raise FileNotFoundError(
                "pKₐte requires the fitted Stage 1/2 and calibration artifacts. "
                f"Missing:\n  - {joined}\n"
                "Restore or train the canonical models, or pass ModelArtifacts "
                "with their locations."
            )


@dataclass(frozen=True)
class PredictionResult:
    """User-facing prediction for one input molecule at one requested pH."""

    molecule_id: str
    input_smiles: str
    canonical_input_smiles: str
    ph: float
    status: str
    dominant_microstate_smiles: Optional[str]
    dominant_atom_mapped_smiles: Optional[str]
    dominant_formal_charge: Optional[int]
    dominant_configuration_probability: Optional[float]
    dominant_complete_microstate_heuristic_weight: Optional[float]
    site_predictions: Tuple[Dict[str, Any], ...]
    ranked_tautomers: Tuple[Dict[str, Any], ...]
    warnings: Tuple[str, ...]
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the prediction."""
        return asdict(self)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: object) -> Optional[int]:
    parsed = _optional_float(value)
    return int(parsed) if parsed is not None else None


def _parse_json_list(value: object) -> list:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _unmapped_smiles(atom_mapped_smiles: object) -> Optional[str]:
    if atom_mapped_smiles is None or pd.isna(atom_mapped_smiles):
        return None
    molecule = Chem.MolFromSmiles(str(atom_mapped_smiles))
    if molecule is None:
        return None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _molecule_group_key(molecule: Chem.Mol, smiles: str) -> str:
    try:
        key = inchi.MolToInchiKey(molecule)
    except Exception:
        key = ""
    if key:
        return "-".join(key.split("-")[:2])
    return "smiles_" + hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:20]


def _detect_sites(
    molecule: Chem.Mol,
    overlap_threshold: float,
) -> tuple[list[Dict], list[Dict]]:
    resolved, _ = resolve_overlapping_sites(
        find_sites_with_metadata(molecule),
        overlap_threshold=float(overlap_threshold),
    )
    transitions: list[Dict] = []
    for site in resolved:
        transitions.extend(expand_pair_site_transitions(site))
    return resolved, transitions


def _no_site_result(
    input_smiles: str,
    canonical_smiles: str,
    molecule: Chem.Mol,
    ph: float,
) -> PredictionResult:
    mapped = mapped_smiles(ensure_heavy_atom_maps(molecule))
    molecule_id = "query_" + hashlib.sha256(
        canonical_smiles.encode("utf-8")
    ).hexdigest()[:16]
    return PredictionResult(
        molecule_id=molecule_id,
        input_smiles=input_smiles,
        canonical_input_smiles=canonical_smiles,
        ph=float(ph),
        status="no_supported_ionizable_sites_detected",
        dominant_microstate_smiles=canonical_smiles,
        dominant_atom_mapped_smiles=mapped,
        dominant_formal_charge=int(
            sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
        ),
        dominant_configuration_probability=1.0,
        dominant_complete_microstate_heuristic_weight=1.0,
        site_predictions=(),
        ranked_tautomers=(),
        warnings=(
            "No ionization transition covered by the current functional-group "
            "policy was detected; the submitted structure is returned unchanged.",
        ),
        provenance={
            "functional_group_detector_schema_version": (
                FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION
            ),
            "stage1_applied": False,
            "stage2_applied": False,
            "stage3_applied": False,
            "marvin_values_used": False,
            "epik_values_used": False,
        },
    )


def _build_query_network(
    *,
    input_smiles: str,
    canonical_smiles: str,
    molecule: Chem.Mol,
    resolved_sites: Sequence[Dict],
    transition_sites: Sequence[Dict],
    ph: float,
    artifacts: ModelArtifacts,
    overlap_threshold: float,
    max_protonation_states: int,
    max_tautomers_per_state: int,
) -> pd.DataFrame:
    """Construct one canonical Stage 1-scored network without experimental labels."""
    stage1_bundle = load_stage1_bundle(
        str(artifacts.stage1_model), require_experimental_only=True
    )
    bundle_detector_versions = {
        str(value)
        for value in stage1_bundle.get(
            "functional_group_detector_schema_versions", []
        )
    }
    current_detector_versions = {FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION}
    if bundle_detector_versions and bundle_detector_versions != current_detector_versions:
        raise ValueError(
            "The current functional-group detector is incompatible with the fitted "
            "Stage 1 model: "
            f"model={sorted(bundle_detector_versions)} "
            f"code={sorted(current_detector_versions)}"
        )
    with artifacts.stage1_metrics.open(encoding="utf-8") as handle:
        stage1_metrics = json.load(handle)
    scaffold_mae = float(stage1_metrics["scaffold"]["mae"])
    feature_columns = set(stage1_bundle["feature_columns"])
    global_calibration = stage1_bundle.get("error_calibration_global", {})
    family_calibration = stage1_bundle.get("error_calibration_by_family", {})
    label_calibration = stage1_bundle.get("error_calibration_by_label", {})

    network = enumerate_joint_protonation_network(
        molecule,
        list(transition_sites),
        max_protonation_states=int(max_protonation_states),
        max_tautomers_per_state=int(max_tautomers_per_state),
    )
    if not network.get("sites") or not network.get("nodes"):
        raise RuntimeError(
            "Ionizable sites were detected, but a molecular microstate network "
            "could not be enumerated for the submitted structure."
        )

    group_key = _molecule_group_key(molecule, canonical_smiles)
    molecule_id = "query_" + hashlib.sha256(
        canonical_smiles.encode("utf-8")
    ).hexdigest()[:16]
    tasks = []
    for site in network["sites"]:
        site["acid_labels"] = sorted(
            PAIR_TYPE_FAMILY_FORMS[site["family"]]["acid_form"]
        )
        site["base_labels"] = sorted(
            PAIR_TYPE_FAMILY_FORMS[site["family"]]["base_form"]
        )
        site["experimental_measurements"] = []
        site["weak_experimental_measurements"] = []
        center_maps = site.get("center_maps", [])
        task_id = f"{molecule_id}:{site['site_id']}"
        site["stage1_task_id"] = task_id
        tasks.append({
            "task_id": task_id,
            "molecule_id": molecule_id,
            "site_id": str(site["site_id"]),
            "smiles": canonical_smiles,
            "candidate_label": str(site["label"]),
            "pka_type_canonical": (
                "acidic" if site["family"] in ACIDIC_FAMILIES else "basic"
            ),
            "atom_index_raw": (
                int(center_maps[0]) - 1 if center_maps else None
            ),
        })
    predictions = predict_stage1_tasks(tasks, str(artifacts.stage1_model), workers=1)

    site_pka_map: Dict[str, float] = {}
    for site in network["sites"]:
        site["stage1_group_feature_present"] = (
            f"group_{training_group_label(site['label'])}" in feature_columns
        )
        site["stage1_scaffold_validation_mae"] = scaffold_mae
        prediction = predictions.get(site.pop("stage1_task_id"))
        if prediction is None:
            raise RuntimeError(
                f"Stage 1 did not return a prediction for {site['site_id']}"
            )
        calibration = label_calibration.get(
            site["label"],
            family_calibration.get(site["family"], global_calibration),
        )
        calibration_rows = int(calibration.get("rows", 0))
        expected_abs_error = float(calibration.get("mae", scaffold_mae))
        error_p90 = float(calibration.get("p90_abs_error", expected_abs_error))
        error_p95 = float(calibration.get("p95_abs_error", error_p90))
        applicability = stage1_applicability(
            canonical_smiles, str(site["label"]), stage1_bundle
        )
        applicability_domain = str(applicability["stage1_applicability_domain"])
        local_pka, local_provenance, confidence, reference_uncertainty = (
            stage1_local_prediction_for_site(
                site, float(prediction), applicability_domain
            )
        )
        if (
            not confidence
            and applicability_domain == "interpolation_high_similarity"
            and error_p90 <= 2.0
        ):
            confidence = "high_empirical_support"
        elif (
            not confidence
            and applicability_domain == "interpolation_limited_support"
            and error_p90 <= 3.0
        ):
            confidence = "medium_empirical_support"
        elif not confidence:
            confidence = "low_empirical_support"
        multiplier = (
            2.0
            if "zero_shot" in applicability_domain
            else 1.5
            if "extrapolation" in applicability_domain
            else 1.0
        )
        half_width = max(
            0.5,
            error_p95 * multiplier,
            float(reference_uncertainty or 0.0),
        )
        site.update({
            "stage1_intrinsic_pka": float(local_pka),
            "local_pka_used": float(local_pka),
            "local_pka_provenance": local_provenance,
            "stage1_raw_model_prediction": float(prediction),
            "local_pka_confidence": confidence,
            "stage1_empirical_calibration_scope": (
                f"label:{site['label']}"
                if site["label"] in label_calibration
                else f"family:{site['family']}"
                if site["family"] in family_calibration
                else "global"
            ),
            "stage1_empirical_calibration_rows": calibration_rows,
            "stage1_expected_abs_error": expected_abs_error,
            "stage1_abs_error_p90": error_p90,
            "stage1_abs_error_p95": error_p95,
            "stage1_prediction_outside_training_filter_range": not (
                -5.0 < float(local_pka) < 40.0
            ),
            **applicability,
            "stage1_predicted_pka_ci_low": float(local_pka) - half_width,
            "stage1_predicted_pka_ci_high": float(local_pka) + half_width,
            "stage1_uncertainty_basis": (
                "transition reference uncertainty and OOF p95; exact label absent"
                if reference_uncertainty is not None
                else "exact-label interpolation OOF p95 widened by applicability domain"
            ),
            "experimental_anchor_pka": None,
        })
        site_pka_map[str(site["site_id"])] = float(local_pka)

    macro_steps: list[Dict] = []
    site_step_map: Dict[str, Dict] = {}
    if (
        len(site_pka_map) == len(network["sites"])
        and not network["protonation_state_enumeration_truncated"]
        and not network["incomplete_site_ids"]
    ):
        macro_steps, site_step_map = coupled_network_thermodynamics(
            network, site_pka_map, float(ph)
        )
    for site in network["sites"]:
        step = site_step_map.get(str(site["site_id"]))
        if step is not None:
            site["rank_associated_macro_step"] = int(step["protonation_step"])
            site["rank_associated_macro_pka"] = float(step["predicted_macro_pka"])
            site["predicted_macro_shift_from_local_pka"] = (
                float(step["predicted_macro_pka"])
                - float(site["local_pka_used"])
            )
    site_lookup = {str(site["site_id"]): site for site in network["sites"]}
    for edge in network["edges"]:
        site = site_lookup[str(edge["site_id"])]
        edge.update({
            "local_pka_used": site["local_pka_used"],
            "local_pka_provenance": site["local_pka_provenance"],
            "local_pka_confidence": site["local_pka_confidence"],
            "context_model": "shared_transition_pka_across_enumerated_context_edges",
        })

    mapped_molecule = ensure_heavy_atom_maps(molecule)
    unresolved = unresolved_ionizable_contexts(
        list(resolved_sites), list(transition_sites)
    )
    limited = bool(
        network["protonation_state_enumeration_truncated"]
        or network["incomplete_site_ids"]
    )
    row = {
        "dataset_schema_version": MOLECULE_NETWORK_DATASET_SCHEMA_VERSION,
        "functional_group_detector_schema_versions_json": _json(
            [FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION]
        ),
        "rdkit_version": rdBase.rdkitVersion,
        "molecule_id": molecule_id,
        "molecule_group_key": group_key,
        "molecule_grouping_rule": (
            "standard_InChIKey_first_two_blocks; protonation block removed; "
            "stereochemistry retained"
        ),
        "representative_input_smiles": canonical_smiles,
        "representative_atom_mapped_smiles": mapped_smiles(mapped_molecule),
        "source_input_smiles_json": _json([input_smiles]),
        "source_files_json": _json(["user_supplied_query"]),
        "source_transition_row_ids_json": _json([]),
        "experimental_transition_count": 0,
        "detected_site_count": len(network["sites"]),
        "experimentally_anchored_site_count": 0,
        "stage1_prior_only_site_count": len(network["sites"]),
        "unattached_experimental_measurement_count": 0,
        "unattached_experimental_measurements_json": _json([]),
        "weak_experimental_measurement_count": 0,
        "weak_experimental_measurements_json": _json([]),
        "unresolved_ionizable_context_count": len(unresolved),
        "unresolved_ionizable_contexts_json": _json(unresolved),
        "network_confidence": (
            "limited" if limited else "stage1_only_unseen_molecule"
        ),
        "protonation_state_count": len(network["nodes"]),
        "protonation_edge_count": len(network["edges"]),
        "protonation_state_enumeration_truncated": bool(
            network["protonation_state_enumeration_truncated"]
        ),
        "protonation_state_cap": int(network["max_protonation_states"]),
        "incomplete_site_ids_json": _json(network["incomplete_site_ids"]),
        "tautomer_cap_per_protonation_state": int(max_tautomers_per_state),
        "sites_json": _json(network["sites"]),
        "microstate_nodes_json": _json(network["nodes"]),
        "microstate_edges_json": _json(network["edges"]),
        "independent_site_macro_pka_steps_json": _json(macro_steps),
        "baseline_macro_pka_steps_json": _json(macro_steps),
        "population_ph": float(ph),
        "thermodynamic_model": (
            "enumerated coupled-coordinate binding polynomial; serial amphoteric "
            "transitions share one coordinate; alternate drawings share "
            "configuration weight"
        ),
        "thermodynamic_caveat": (
            "blind Stage 1 local pKas only; no Stage 2 site-site coupling "
            "correction in this baseline row"
        ),
        "stage1_model_sha256": _sha256(artifacts.stage1_model),
        "stage1_schema_version": stage1_bundle.get("stage1_schema_version"),
        "stage1_target_definition": stage1_bundle.get("target_definition"),
        "stage1_training_measurement_method": stage1_bundle.get(
            "training_measurement_method"
        ),
        "stage1_training_rows": stage1_bundle.get("training_rows"),
        "stage1_training_measurement_count": stage1_bundle.get(
            "training_measurement_count"
        ),
        "stage1_training_dataset_sha256": stage1_bundle.get(
            "training_dataset_sha256"
        ),
        "stage1_trained_on_all_eligible_rows": stage1_bundle.get(
            "trained_on_all_eligible_rows"
        ),
        "stage1_scaffold_validation_mae": scaffold_mae,
        "marvin_values_used": False,
        "epik_values_used": False,
    }
    return pd.DataFrame([row])


def _site_payload(row: pd.Series) -> Dict[str, Any]:
    protonated_probability = _optional_float(
        row.get("stage3_protonated_probability_at_ph")
    )
    predicted_form = str(row.get("stage3_predicted_site_form_at_ph", "unavailable"))
    return {
        "site_id": str(row["site_id"]),
        "site_label": str(row["site_label"]),
        "site_family": str(row["site_family"]),
        "atom_maps": _parse_json_list(row.get("site_atom_maps_json", "[]")),
        "center_atom_maps": _parse_json_list(
            row.get("site_center_maps_json", "[]")
        ),
        "predicted_form": predicted_form,
        "predicted_protonation_state": (
            "protonated"
            if predicted_form == "acid_form"
            else "deprotonated"
            if predicted_form == "base_form"
            else "unavailable"
        ),
        "protonated_probability": protonated_probability,
        "protonated_probability_interval": [
            _optional_float(row.get("stage3_protonated_probability_ci_low")),
            _optional_float(row.get("stage3_protonated_probability_ci_high")),
        ],
        "predicted_macroscopic_pka": _optional_float(
            row.get("stage2_predicted_macro_pka")
        ),
        "predicted_macroscopic_pka_interval_90": [
            _optional_float(row.get("stage3_empirical_macro_pka_interval_90_low")),
            _optional_float(row.get("stage3_empirical_macro_pka_interval_90_high")),
        ],
        "contextual_edge_pka_at_dominant_background": _optional_float(
            row.get("stage3_contextual_edge_pka_at_dominant_background")
        ),
        "overall_confidence": _optional_float(
            row.get("stage3_overall_site_state_confidence")
        ),
        "confidence_tier": str(
            row.get("stage3_overall_site_state_confidence_tier", "unavailable")
        ),
        "stage1_applicability_domain": str(
            row.get("stage1_applicability_domain", "unknown")
        ),
        "stage1_exact_label_training_rows": _optional_int(
            row.get("stage1_exact_label_training_rows")
        ),
        "stage2_applied": bool(row.get("stage2_applied", False)),
        "stage2_prediction_confidence": str(
            row.get("stage2_prediction_confidence", "unknown")
        ),
    }


def _tautomer_payload(row: pd.Series) -> Dict[str, Any]:
    mapped = str(row["atom_mapped_smiles"])
    return {
        "rank": int(row["tautomer_rank_within_configuration"]),
        "smiles": _unmapped_smiles(mapped),
        "atom_mapped_smiles": mapped,
        "conditional_heuristic_weight": _optional_float(
            row.get("tautomer_heuristic_conditional_weight")
        ),
        "complete_microstate_heuristic_weight": _optional_float(
            row.get("complete_microstate_heuristic_population_at_ph")
        ),
        "rdkit_tautomer_score": _optional_float(row.get("rdkit_tautomer_score")),
        "ranking_semantics": str(row.get("ranking_semantics", "")),
    }


def _result_from_outputs(
    *,
    input_smiles: str,
    canonical_smiles: str,
    ph: float,
    molecule_table: pd.DataFrame,
    site_table: pd.DataFrame,
    tautomer_table: pd.DataFrame,
    artifacts: ModelArtifacts,
    persistent_output_dir: Optional[Path],
    unresolved_contexts: Sequence[Dict],
    top_tautomers: int,
) -> PredictionResult:
    row = molecule_table.iloc[0]
    molecule_id = str(row["molecule_id"])
    dominant_mapped_raw = row.get("stage3_dominant_atom_mapped_smiles")
    dominant_mapped = (
        None
        if dominant_mapped_raw is None or pd.isna(dominant_mapped_raw)
        else str(dominant_mapped_raw)
    )
    molecule_sites = site_table[site_table["molecule_id"].astype(str) == molecule_id]
    site_payloads = tuple(
        _site_payload(site)
        for _, site in molecule_sites.sort_values("site_id").iterrows()
    )
    dominant_node = str(row.get("stage3_dominant_configuration_node_id", ""))
    dominant_tautomers = tautomer_table[
        (tautomer_table["molecule_id"].astype(str) == molecule_id)
        & (tautomer_table["origin_node_id"].astype(str) == dominant_node)
    ].sort_values("tautomer_rank_within_configuration")
    if int(top_tautomers) == 0:
        dominant_tautomers = dominant_tautomers.iloc[0:0]
    else:
        dominant_tautomers = dominant_tautomers.head(int(top_tautomers))
    tautomer_payloads = tuple(
        _tautomer_payload(tautomer) for _, tautomer in dominant_tautomers.iterrows()
    )

    warnings = []
    if unresolved_contexts:
        warnings.append(
            "The detector reported unresolved ionizable contexts; inspect provenance "
            "before relying on the complete-state call."
        )
    status = str(row.get("stage3_population_status", "unavailable"))
    if status != "complete_pairwise_coupled_population":
        warnings.append(
            "The protonation network is incomplete or truncated, so whole-molecule "
            "populations are unavailable."
        )
    low_confidence_sites = [
        site["site_id"]
        for site in site_payloads
        if site["confidence_tier"] in {"low", "very_low"}
    ]
    if low_confidence_sites:
        warnings.append(
            "Low-confidence site calls: " + ", ".join(low_confidence_sites)
        )

    provenance: Dict[str, Any] = {
        "prediction_mode": "unseen_molecule_no_experimental_query_labels",
        "functional_group_detector_schema_version": (
            FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION
        ),
        "network_dataset_schema_version": MOLECULE_NETWORK_DATASET_SCHEMA_VERSION,
        "rdkit_version": rdBase.rdkitVersion,
        "stage1_model_sha256": _sha256(artifacts.stage1_model),
        "stage2_model_sha256": _sha256(artifacts.stage2_model),
        "marvin_values_used": False,
        "epik_values_used": False,
    }
    if persistent_output_dir is not None:
        provenance["output_directory"] = str(persistent_output_dir)

    return PredictionResult(
        molecule_id=molecule_id,
        input_smiles=input_smiles,
        canonical_input_smiles=canonical_smiles,
        ph=float(ph),
        status=status,
        dominant_microstate_smiles=_unmapped_smiles(dominant_mapped),
        dominant_atom_mapped_smiles=dominant_mapped,
        dominant_formal_charge=_optional_int(row.get("stage3_dominant_formal_charge")),
        dominant_configuration_probability=_optional_float(
            row.get("stage3_dominant_population")
        ),
        dominant_complete_microstate_heuristic_weight=_optional_float(
            row.get("stage3_dominant_complete_microstate_heuristic_population")
        ),
        site_predictions=site_payloads,
        ranked_tautomers=tautomer_payloads,
        warnings=tuple(warnings),
        provenance=provenance,
    )


def predict(
    smiles: str,
    ph: float = 7.4,
    *,
    artifacts: Optional[ModelArtifacts] = None,
    output_dir: Optional[str | Path] = None,
    overlap_threshold: float = 0.5,
    max_protonation_states: int = DEFAULT_MAX_PROTONATION_STATES,
    max_tautomers_per_state: int = 4,
    max_tautomers_per_configuration: int = (
        DEFAULT_MAX_TAUTOMERS_PER_CONFIGURATION
    ),
    tautomer_score_temperature: float = DEFAULT_TAUTOMER_SCORE_TEMPERATURE,
    top_tautomers: int = 5,
) -> PredictionResult:
    """Predict pKₐ values and the complete microstate of one unseen molecule.

    Parameters
    ----------
    smiles:
        Input structure. Charged or neutral forms are accepted; heavy atoms are
        canonicalized and atom-mapped internally.
    ph:
        pH at which configuration populations and site states are requested.
    artifacts:
        Optional non-default paths to fitted Stage 1/2 and calibration files.
    output_dir:
        If supplied, retain all canonical network, site, tautomer, metrics, and
        provenance artifacts. Otherwise intermediates live in a temporary
        directory and only the structured result is returned.

    Returns
    -------
    PredictionResult
        Dominant complete microstate, per-site pKₐ/probability/confidence rows,
        top ranked tautomers, warnings, and model provenance.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")
    if not math.isfinite(float(ph)):
        raise ValueError("ph must be finite")
    if not 0.0 <= float(overlap_threshold) <= 1.0:
        raise ValueError("overlap_threshold must be between zero and one")
    if int(max_protonation_states) < 1:
        raise ValueError("max_protonation_states must be at least one")
    if int(max_tautomers_per_state) < 1:
        raise ValueError("max_tautomers_per_state must be at least one")
    if int(max_tautomers_per_configuration) < 1:
        raise ValueError("max_tautomers_per_configuration must be at least one")
    if int(top_tautomers) < 0:
        raise ValueError("top_tautomers cannot be negative")

    input_smiles = smiles.strip()
    molecule = Chem.MolFromSmiles(input_smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse the submitted SMILES: {input_smiles!r}")
    canonical_smiles = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    resolved_sites, transition_sites = _detect_sites(molecule, overlap_threshold)
    if not transition_sites:
        return _no_site_result(
            input_smiles, canonical_smiles, molecule, float(ph)
        )

    model_artifacts = artifacts or ModelArtifacts.defaults()
    model_artifacts.validate()
    unresolved = unresolved_ionizable_contexts(
        list(resolved_sites), list(transition_sites)
    )

    persistent_dir = Path(output_dir).resolve() if output_dir is not None else None
    context = (
        nullcontext(str(persistent_dir))
        if persistent_dir is not None
        else tempfile.TemporaryDirectory(prefix="pkate-prediction-")
    )
    if persistent_dir is not None:
        persistent_dir.mkdir(parents=True, exist_ok=True)
    with context as directory:
        work_dir = Path(directory)
        stage1_network_path = work_dir / "query_stage1_network.csv"
        stage2_network_path = work_dir / "query_stage2_network.csv"
        stage2_report_path = work_dir / "stage2_application_metrics.json"
        stage3_molecule_path = work_dir / "query_stage3_molecule.csv"
        stage3_dir = work_dir / "stage3"

        network = _build_query_network(
            input_smiles=input_smiles,
            canonical_smiles=canonical_smiles,
            molecule=molecule,
            resolved_sites=resolved_sites,
            transition_sites=transition_sites,
            ph=float(ph),
            artifacts=model_artifacts,
            overlap_threshold=float(overlap_threshold),
            max_protonation_states=int(max_protonation_states),
            max_tautomers_per_state=int(max_tautomers_per_state),
        )
        network.to_csv(stage1_network_path, index=False)
        apply_stage2(
            network_dataset_path=str(stage1_network_path),
            model_path=str(model_artifacts.stage2_model),
            output_path=str(stage2_network_path),
            allow_compatible_network=True,
            report_path=str(stage2_report_path),
        )
        molecule_table, _ = apply_stage3(
            input_path=str(stage2_network_path),
            stage2_model_path=str(model_artifacts.stage2_model),
            output_path=str(stage3_molecule_path),
            out_dir=str(stage3_dir),
            ph=float(ph),
            stage1_oof_path=str(model_artifacts.stage1_oof),
            stage2_eval_path=str(model_artifacts.stage2_eval),
            stage2_quarantine_path=str(model_artifacts.stage2_quarantine),
            tautomer_score_temperature=float(tautomer_score_temperature),
            max_tautomers_per_configuration=int(
                max_tautomers_per_configuration
            ),
        )
        site_table = pd.read_csv(
            stage3_dir / "stage3_site_predictions.csv", low_memory=False
        )
        try:
            tautomer_table = pd.read_csv(
                stage3_dir / "stage3_ranked_tautomers.csv", low_memory=False
            )
        except pd.errors.EmptyDataError:
            tautomer_table = pd.DataFrame(columns=[
                "molecule_id",
                "origin_node_id",
                "atom_mapped_smiles",
                "tautomer_rank_within_configuration",
            ])
        result = _result_from_outputs(
            input_smiles=input_smiles,
            canonical_smiles=canonical_smiles,
            ph=float(ph),
            molecule_table=molecule_table,
            site_table=site_table,
            tautomer_table=tautomer_table,
            artifacts=model_artifacts,
            persistent_output_dir=persistent_dir,
            unresolved_contexts=unresolved,
            top_tautomers=int(top_tautomers),
        )
        if persistent_dir is not None:
            summary_path = persistent_dir / "prediction.json"
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(result.to_dict(), handle, indent=2, allow_nan=False)
        return result


calculate_pkas = predict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict pKa values and the dominant microstate of one molecule."
    )
    parser.add_argument("smiles", help="Input molecule as a SMILES string")
    parser.add_argument("--ph", type=float, default=7.4, help="Requested pH (default: 7.4)")
    parser.add_argument(
        "--output-dir",
        help="Retain detailed network/site/tautomer artifacts in this directory",
    )
    parser.add_argument(
        "--top-tautomers",
        type=int,
        default=5,
        help="Number of dominant-configuration tautomers in JSON (default: 5)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact rather than indented JSON",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the single-molecule CLI and print its JSON result."""
    args = _parser().parse_args(argv)
    result = predict(
        args.smiles,
        ph=args.ph,
        output_dir=args.output_dir,
        top_tautomers=args.top_tautomers,
    )
    print(
        json.dumps(
            result.to_dict(),
            indent=None if args.compact else 2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
