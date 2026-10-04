"""Canonical, replicate-aware Stage 1 training data from molecule networks."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore

from functional_group_pka_analysis import ACIDIC_FAMILIES
from pka_evidence_policy import EVIDENCE_POLICY_SCHEMA_VERSION, reference_prior


STAGE1_TRAINING_SCHEMA_VERSION = "3.1.0"
TARGET_DEFINITION = (
    "evidence_tiered_median_experimental_pka_for_transition_on_"
    "chemically_single_coordinate_molecule"
)


def _json(value: object) -> str:
    """Serialize nested provenance fields deterministically for stable artifacts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_structure_key(smiles: str) -> str:
    """Return an isomeric canonical key, with a stable hash for invalid SMILES."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "invalid::" + hashlib.sha256(str(smiles).encode("utf-8")).hexdigest()[:20]
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def scaffold_group_key(smiles: str) -> str:
    """Ring systems share a Murcko scaffold; acyclic molecules remain structure-grouped."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "invalid::" + hashlib.sha256(str(smiles).encode("utf-8")).hexdigest()[:20]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=True,
    )
    if scaffold:
        return "murcko::" + scaffold
    return "acyclic::" + canonical_structure_key(smiles)


def _measurement_weight(
    count: int,
    robust_sigma: float,
    medium_fraction: float,
    evidence_tier: str,
) -> float:
    """Weight a target by replicate support, agreement, mapping confidence, and tier."""
    # Replicates add evidence sublinearly; disagreement and uncertain mappings reduce it.
    count_factor = min(1.5, 1.0 + 0.20 * math.log2(max(1, count)))
    dispersion_factor = 1.0 / (1.0 + robust_sigma * robust_sigma)
    confidence_factor = 1.0 - 0.25 * min(1.0, max(0.0, medium_fraction))
    evidence_factor = {"gold": 1.0, "silver": 0.70}.get(str(evidence_tier), 0.25)
    return float(np.clip(
        count_factor * dispersion_factor * confidence_factor * evidence_factor,
        0.10,
        1.5,
    ))


def _truthy(value: object, default: bool = False) -> bool:
    """Interpret common CSV Boolean spellings, using ``default`` for missing values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _safe_int(value: object, default: int = 0) -> int:
    """Convert a CSV scalar to ``int`` without propagating missing or malformed data."""
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _independent_measurements(measurements: List[Dict]) -> List[Dict]:
    """Collapse copied values without counting source-file copies as replicates."""
    chosen: Dict[Tuple[object, ...], Dict] = {}
    for record in measurements:
        value = float(record["experimental_pka"])
        document = str(record.get("source_document_id", "")).strip()
        assay = str(record.get("source_assay_id", "")).strip()
        if document and document.lower() != "nan":
            key = ("document", document, assay, round(value, 6))
        else:
            # In citation-free compilations, an exact copied scalar is not
            # independent merely because it appears in another SDF file.
            key = ("citation_unknown_exact_value", round(value, 6))
        existing = chosen.get(key)
        if existing is None:
            chosen[key] = record
            continue
        rank = {"gold": 2, "silver": 1}.get(str(record.get("evidence_tier", "")), 0)
        existing_rank = {"gold": 2, "silver": 1}.get(
            str(existing.get("evidence_tier", "")), 0
        )
        if rank > existing_rank:
            chosen[key] = record
    return list(chosen.values())


def build_stage1_training_sites(
    network_df: pd.DataFrame,
    max_replicate_range: float = 2.0,
    min_supported_pka: float = -5.0,
    max_supported_pka: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Extract robust edge targets from eligible single-coordinate molecules.

    A serial amphoteric ring is one physical protonation coordinate even though
    it has two adjacent pKa edges. Each measured, transition-resolved edge may
    supervise Stage 1; an unmeasured sibling edge remains prior-only.
    """
    required = {
        "molecule_id",
        "representative_input_smiles",
        "representative_atom_mapped_smiles",
        "detected_site_count",
        "sites_json",
        "marvin_values_used",
        "epik_values_used",
    }
    missing = required - set(network_df.columns)
    if missing:
        raise ValueError(f"Network dataset missing Stage 1 source columns: {sorted(missing)}")
    if not network_df["marvin_values_used"].eq(False).all():
        raise ValueError("Canonical Stage 1 source contains prohibited Marvin values")
    if not network_df["epik_values_used"].eq(False).all():
        raise ValueError("Canonical Stage 1 source contains prohibited Epik values")

    source_network_molecules = int(len(network_df))
    stage1_candidates = []
    single_coordinate_molecules = 0
    for _, molecule in network_df.iterrows():
        try:
            sites = json.loads(str(molecule["sites_json"]))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        coordinate_ids = {
            str(site.get("coupling_group_id") or site.get("site_id") or "")
            for site in sites
        }
        if len(coordinate_ids) != 1 or not sites:
            continue
        single_coordinate_molecules += 1
        measured_sites = [
            site for site in sites if site.get("experimental_measurements", [])
        ]
        # Preserve the historical no-measurement diagnostic once per physical
        # coordinate, but never manufacture a target for an unmeasured sibling
        # edge of a serial coordinate.
        candidate_sites = measured_sites or sites[:1]
        for site in candidate_sites:
            candidate = molecule.copy()
            candidate["detected_site_count"] = 1
            candidate["sites_json"] = _json([site])
            stage1_candidates.append(candidate)
    network_df = pd.DataFrame(stage1_candidates, columns=network_df.columns)

    accepted: List[Dict] = []
    quarantined: List[Dict] = []
    funnel = {
        "network_molecules": source_network_molecules,
        "single_coordinate_molecules": int(single_coordinate_molecules),
        "single_coordinate_transition_candidates": int(len(network_df)),
        "single_site_molecules": 0,
        "single_site_with_measurements": 0,
        "accepted_training_sites": 0,
        "quarantined_training_sites": 0,
    }

    for _, molecule in network_df.iterrows():
        if int(molecule["detected_site_count"]) != 1:
            continue
        funnel["single_site_molecules"] += 1
        sites = json.loads(str(molecule["sites_json"]))
        if len(sites) != 1:
            quarantined.append({
                "molecule_id": str(molecule["molecule_id"]),
                "quarantine_reason": "single_site_count_json_mismatch",
                "detail": f"detected_site_count=1 sites_json_count={len(sites)}",
            })
            continue
        site = sites[0]
        try:
            incomplete_site_ids = json.loads(str(
                molecule.get("incomplete_site_ids_json", "[]")
            ))
        except (TypeError, ValueError, json.JSONDecodeError):
            incomplete_site_ids = ["unreadable_completeness_declaration"]
        if (
            _truthy(molecule.get("protonation_state_enumeration_truncated", False))
            or bool(incomplete_site_ids)
        ):
            quarantined.append({
                "molecule_id": str(molecule["molecule_id"]),
                "site_id": str(site.get("site_id", "")),
                "site_family": str(site.get("family", "")),
                "site_label": str(site.get("label", "")),
                "quarantine_reason": "incomplete_or_truncated_microstate_network",
                "detail": _json(incomplete_site_ids),
            })
            continue
        all_measurements = site.get("experimental_measurements", [])
        measurements = [
            record
            for record in all_measurements
            if _truthy(record.get("stage1_local_supervision_eligible"), default=True)
            and str(record.get("evidence_tier", "legacy_exact")) in {"gold", "silver", "legacy_exact"}
        ]
        unresolved_count = _safe_int(
            molecule.get("unresolved_ionizable_context_count", 0)
        )
        if unresolved_count:
            quarantined.append({
                "molecule_id": str(molecule["molecule_id"]),
                "site_id": str(site.get("site_id", "")),
                "site_family": str(site.get("family", "")),
                "site_label": str(site.get("label", "")),
                "quarantine_reason": "competing_unresolved_ionizable_context",
                "detail": str(molecule.get("unresolved_ionizable_contexts_json", "[]")),
            })
            continue
        measurements = _independent_measurements(measurements)
        values = np.asarray(
            [float(record["experimental_pka"]) for record in measurements],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if not len(values):
            quarantined.append({
                "molecule_id": str(molecule["molecule_id"]),
                "site_id": str(site.get("site_id", "")),
                "quarantine_reason": "no_finite_experimental_measurement",
                "detail": "",
            })
            continue
        funnel["single_site_with_measurements"] += 1

        value_range = float(np.max(values) - np.min(values))
        if value_range > float(max_replicate_range):
            quarantined.append({
                "molecule_id": str(molecule["molecule_id"]),
                "site_id": str(site.get("site_id", "")),
                "site_family": str(site.get("family", "")),
                "site_label": str(site.get("label", "")),
                "measurement_count": int(len(values)),
                "measurement_min": float(np.min(values)),
                "measurement_max": float(np.max(values)),
                "measurement_range": value_range,
                "quarantine_reason": "replicate_range_exceeds_threshold",
                "detail": f"range={value_range:.6g}; threshold={max_replicate_range:.6g}",
            })
            continue

        median = float(np.median(values))
        if not float(min_supported_pka) <= median <= float(max_supported_pka):
            quarantined.append({
                "molecule_id": str(molecule["molecule_id"]),
                "site_id": str(site.get("site_id", "")),
                "site_family": str(site.get("family", "")),
                "site_label": str(site.get("label", "")),
                "measurement_count": int(len(values)),
                "measurement_min": float(np.min(values)),
                "measurement_max": float(np.max(values)),
                "measurement_range": value_range,
                "quarantine_reason": "target_outside_supported_pka_range",
                "detail": (
                    f"median={median:.6g}; supported=[{min_supported_pka:.6g},"
                    f"{max_supported_pka:.6g}]"
                ),
            })
            continue
        mad = float(np.median(np.abs(values - median)))
        robust_sigma = float(1.4826 * mad)
        confidences = [str(record.get("structural_site_assignment_confidence", record.get("site_assignment_confidence", "unknown"))) for record in measurements]
        medium_fraction = sum(value != "high" for value in confidences) / max(1, len(confidences))
        evidence_tiers = [str(record.get("evidence_tier", "legacy_exact")) for record in measurements]
        aggregate_evidence_tier = (
            "gold" if evidence_tiers and all(value == "gold" for value in evidence_tiers)
            else "silver" if evidence_tiers else "legacy_exact"
        )
        center_maps = [int(value) for value in site.get("center_maps", [])]
        smiles = str(molecule["representative_input_smiles"])
        prior = reference_prior(site.get("label"), site.get("family"))
        accepted.append({
            "stage1_training_schema_version": STAGE1_TRAINING_SCHEMA_VERSION,
            "evidence_policy_schema_version": EVIDENCE_POLICY_SCHEMA_VERSION,
            "functional_group_detector_schema_versions_json": str(
                molecule.get("functional_group_detector_schema_versions_json", "[]")
            ),
            "target_definition": TARGET_DEFINITION,
            "molecule_key": str(molecule["molecule_id"]),
            "molecule_id": str(molecule["molecule_id"]),
            "structure_key": canonical_structure_key(smiles),
            "scaffold_group": scaffold_group_key(smiles),
            "smiles": smiles,
            "mapped_smiles": str(molecule["representative_atom_mapped_smiles"]),
            "site_id": str(site["site_id"]),
            "candidate_label": str(site["label"]),
            "final_group_label": str(site["label"]),
            "site_family": str(site["family"]),
            "pka_type_canonical": "acidic" if str(site["family"]) in ACIDIC_FAMILIES else "basic",
            "atom_index_raw": int(center_maps[0] - 1) if center_maps else np.nan,
            "pka_value": median,
            "experimental_anchor_pka": median,
            "experimental_measurement_count": int(len(all_measurements)),
            "experimental_eligible_measurement_count": int(len(values)),
            "experimental_independent_lineage_count": int(len(values)),
            "experimental_measurement_mean": float(np.mean(values)),
            "experimental_measurement_std": float(np.std(values)),
            "experimental_measurement_mad": mad,
            "experimental_measurement_robust_sigma": robust_sigma,
            "experimental_measurement_min": float(np.min(values)),
            "experimental_measurement_max": float(np.max(values)),
            "experimental_measurement_range": value_range,
            "experimental_medium_confidence_fraction": float(medium_fraction),
            "experimental_evidence_tier": aggregate_evidence_tier,
            "stage1_reference_pka": prior.pka if prior is not None else np.nan,
            "stage1_reference_uncertainty": prior.uncertainty if prior is not None else np.nan,
            "stage1_reference_transition_definition": (
                prior.transition_definition if prior is not None else ""
            ),
            "stage1_reference_provenance": prior.provenance if prior is not None else "",
            "experimental_source_site_evidence_json": _json(sorted({
                str(record.get("source_site_evidence", "unknown")) for record in measurements
            })),
            "experimental_source_site_evidence_confidences_json": _json(sorted({
                str(record.get("source_site_evidence_confidence", "unknown")) for record in measurements
            })),
            "sample_weight_raw": _measurement_weight(
                len(values), robust_sigma, medium_fraction, aggregate_evidence_tier
            ),
            "measurement_method": "experimental",
            "target_aggregation": "median",
            "experimental_transition_ids_json": _json(sorted(
                str(record.get("transition_row_id", "")) for record in measurements
            )),
            "experimental_source_files_json": _json(sorted({
                str(record.get("source_file", "")) for record in measurements
            })),
            "experimental_record_indices_json": _json(sorted({
                int(record["record_index"]) for record in measurements if record.get("record_index") is not None
            })),
        })

    training = pd.DataFrame(accepted)
    if not training.empty:
        training = training.sort_values(["molecule_id", "site_id"]).reset_index(drop=True)
    quarantine = pd.DataFrame(quarantined)
    if not training.empty:
        mean_weight = float(training["sample_weight_raw"].mean())
        training["sample_weight"] = training["sample_weight_raw"] / mean_weight
    funnel["accepted_training_sites"] = int(len(training))
    funnel["quarantined_training_sites"] = int(len(quarantine))
    return training, quarantine, funnel


def load_stage1_training_sites(
    network_dataset_path: str,
    max_replicate_range: float = 2.0,
    min_supported_pka: float = -5.0,
    max_supported_pka: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Load a network snapshot and derive its accepted and quarantined Stage 1 sites.

    The returned frames preserve the experimental-only provenance checks performed
    by :func:`build_stage1_training_sites`; the dictionary records the data funnel.
    """
    network_df = pd.read_csv(network_dataset_path, low_memory=False)
    if "stage1_training_measurement_method" in network_df.columns:
        methods = set(network_df["stage1_training_measurement_method"].astype(str).str.lower())
        if methods != {"experimental"}:
            raise ValueError(f"Network Stage 1 provenance is not experimental-only: {sorted(methods)}")
    return build_stage1_training_sites(
        network_df,
        max_replicate_range=max_replicate_range,
        min_supported_pka=min_supported_pka,
        max_supported_pka=max_supported_pka,
    )
