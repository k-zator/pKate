"""Canonical Stage 2 features for correcting Stage-1/network macroscopic pKas."""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore
from rdkit import Chem, rdBase  # type: ignore
from sklearn.isotonic import IsotonicRegression  # type: ignore

from geometry_features import embed_molecule_3d
from ml_features import molecule_descriptors, morgan_bits


STAGE2_TARGET_DEFINITION = (
    "evidence_tiered_exact_or_marginalized_weak_macro_pka_minus_blind_stage1_network_macro_pka"
)
STAGE2_TRAINING_SCHEMA_VERSION = "2.2.0"


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


def _network_is_limited(row: pd.Series) -> bool:
    """Return whether truncation or incomplete sites make a network unsafe to train on."""
    if _truthy(row.get("protonation_state_enumeration_truncated", False)):
        return True
    try:
        incomplete = json.loads(str(row.get("incomplete_site_ids_json", "[]")))
    except (TypeError, ValueError, json.JSONDecodeError):
        # An unreadable completeness declaration is not evidence of a complete
        # candidate network.
        return True
    return bool(incomplete)


def detector_schema_versions(network_df: pd.DataFrame) -> List[str]:
    """Collect functional-group detector versions carried by a network snapshot."""
    versions = set()
    for raw in network_df["functional_group_detector_schema_versions_json"]:
        try:
            parsed = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = [str(raw)]
        versions.update(str(value) for value in parsed if str(value))
    return sorted(versions)


def project_macro_pkas_nonincreasing(values: List[float]) -> List[float]:
    """Least-squares projection onto a non-increasing macro-pKa ladder."""
    if len(values) < 2:
        return [float(value) for value in values]
    x = np.arange(len(values), dtype=float)
    projected = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(
        x,
        np.asarray(values, dtype=float),
    )
    return [float(value) for value in projected]


def project_macro_pkas_nonincreasing_with_fixed(
    values: List[float],
    adjustable: List[bool],
) -> List[float]:
    """Project adjustable ladder entries while leaving fallback entries fixed.

    The Stage-1 binding-polynomial ladder is already non-increasing.  Stage 2
    may adjust only supported families, so unsupported entries act as exact
    upper/lower bounds for each contiguous adjustable block.
    """
    if len(values) != len(adjustable):
        raise ValueError("values and adjustable must have equal length")
    result = [float(value) for value in values]
    start = 0
    while start < len(result):
        if not adjustable[start]:
            start += 1
            continue
        stop = start
        while stop + 1 < len(result) and adjustable[stop + 1]:
            stop += 1
        block = project_macro_pkas_nonincreasing(result[start : stop + 1])
        upper = result[start - 1] if start > 0 else math.inf
        lower = result[stop + 1] if stop + 1 < len(result) else -math.inf
        result[start : stop + 1] = [
            float(np.clip(value, lower, upper)) for value in block
        ]
        start = stop + 1
    return result


def _safe_stats(values: List[float], prefix: str) -> Dict[str, float]:
    """Return fixed-shape summary statistics, using zeros for an empty context."""
    if not values:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_count": float(len(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
    }


def _site_center_indices(mapped_mol: Chem.Mol, site: Dict) -> List[int]:
    """Resolve a site's atom-map identifiers to RDKit atom indices."""
    maps = {int(value) for value in site.get("center_maps", [])}
    return [
        atom.GetIdx()
        for atom in mapped_mol.GetAtoms()
        if atom.GetAtomMapNum() in maps
    ]


def _geometry_context(mapped_smiles: str, sites: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Describe graph and optional 3D separations between ionizable site centers."""
    mapped_mol = Chem.MolFromSmiles(mapped_smiles)
    if mapped_mol is None:
        return {site["site_id"]: {} for site in sites}
    distance = Chem.GetDistanceMatrix(mapped_mol)
    centers = {site["site_id"]: _site_center_indices(mapped_mol, site) for site in sites}

    with rdBase.BlockLogs():
        mol_3d = embed_molecule_3d(mapped_mol)
    coords = None
    mol_centroid = None
    if mol_3d is not None and mol_3d.GetNumConformers():
        conformer = mol_3d.GetConformer()
        coords = np.asarray([
            [conformer.GetAtomPosition(idx).x, conformer.GetAtomPosition(idx).y, conformer.GetAtomPosition(idx).z]
            for idx in range(mapped_mol.GetNumAtoms())
        ])
        heavy = [atom.GetIdx() for atom in mapped_mol.GetAtoms() if atom.GetAtomicNum() > 1]
        if heavy:
            mol_centroid = coords[heavy].mean(axis=0)

    result = {}
    for site in sites:
        site_id = site["site_id"]
        target = centers[site_id]
        graph_distances = []
        euclidean_distances = []
        for other_id, other in centers.items():
            if other_id == site_id or not target or not other:
                continue
            graph_distances.append(min(float(distance[a, b]) for a in target for b in other))
            if coords is not None:
                target_centroid = coords[target].mean(axis=0)
                other_centroid = coords[other].mean(axis=0)
                euclidean_distances.append(float(np.linalg.norm(target_centroid - other_centroid)))

        payload = {
            **_safe_stats(graph_distances, "other_center_graph_distance"),
            **_safe_stats(euclidean_distances, "other_center_3d_distance"),
            "stage2_geometry_embed_success": float(coords is not None),
            "target_center_count": float(len(target)),
        }
        if coords is not None and mol_centroid is not None and target:
            payload["target_center_to_molecule_centroid_3d"] = float(
                np.linalg.norm(coords[target].mean(axis=0) - mol_centroid)
            )
        else:
            payload["target_center_to_molecule_centroid_3d"] = 0.0
        result[site_id] = payload
    return result


def molecule_network_to_site_rows(row: pd.Series) -> List[Dict]:
    """Flatten one complete molecular network into one context record per site.

    Each row retains the blind Stage 1 macro-pKa baseline and, when present, an
    experimental residual target. Incomplete networks deliberately yield no rows.
    """
    if _network_is_limited(row):
        return []
    sites = json.loads(str(row["sites_json"]))
    if not sites:
        return []
    geometry = _geometry_context(str(row["representative_atom_mapped_smiles"]), sites)
    rows = []
    for site in sites:
        site_id = site["site_id"]
        stage1_pka = site.get("stage1_intrinsic_pka", site.get("local_pka_used"))
        macro_pka = site.get("rank_associated_macro_pka")
        if stage1_pka is None or macro_pka is None:
            continue
        other_sites = [candidate for candidate in sites if candidate["site_id"] != site_id]
        other_pkas = [float(candidate["stage1_intrinsic_pka"]) for candidate in other_sites if candidate.get("stage1_intrinsic_pka") is not None]
        gaps = [abs(float(value) - float(stage1_pka)) for value in other_pkas]
        other_family_counts: Dict[str, int] = {}
        for candidate in other_sites:
            family = str(candidate["family"])
            other_family_counts[family] = other_family_counts.get(family, 0) + 1

        experimental_values = [
            float(record["experimental_pka"])
            for record in site.get("experimental_measurements", [])
        ]
        experimental_values = [value for value in experimental_values if np.isfinite(value)]
        experimental_anchor = site.get("experimental_anchor_pka")
        experimental_median = (
            float(np.median(experimental_values)) if experimental_values else np.nan
        )
        experimental_mad = (
            float(np.median(np.abs(np.asarray(experimental_values) - experimental_median)))
            if experimental_values else np.nan
        )
        experimental_confidences = [
            str(record.get("site_assignment_confidence", "unknown"))
            for record in site.get("experimental_measurements", [])
        ]
        experimental_evidence_tiers = [
            str(record.get("evidence_tier", "legacy_exact"))
            for record in site.get("experimental_measurements", [])
        ]
        payload = {
            "molecule_key": str(row["molecule_id"]),
            "molecule_id": str(row["molecule_id"]),
            "site_id": str(site_id),
            "smiles": str(row["representative_input_smiles"]),
            "mapped_smiles": str(row["representative_atom_mapped_smiles"]),
            "site_label": str(site["label"]),
            "site_family": str(site["family"]),
            "stage1_intrinsic_pka": float(stage1_pka),
            "stage1_network_macro_pka": float(macro_pka),
            "stage1_macro_shift": float(macro_pka) - float(stage1_pka),
            "stage1_macro_step": float(site.get("rank_associated_macro_step", 0)),
            "detected_site_count": float(row["detected_site_count"]),
            "protonation_state_count": float(row["protonation_state_count"]),
            "protonation_edge_count": float(row["protonation_edge_count"]),
            "network_truncated": float(bool(row["protonation_state_enumeration_truncated"])),
            "site_atom_count": float(len(site.get("atom_maps", []))),
            "site_center_count": float(len(site.get("center_maps", []))),
            "stage1_group_feature_present": float(bool(site.get("stage1_group_feature_present", False))),
            "other_pka_closest_gap": float(min(gaps)) if gaps else 0.0,
            "other_pka_higher_count": float(sum(value > float(stage1_pka) for value in other_pkas)),
            "other_pka_lower_count": float(sum(value < float(stage1_pka) for value in other_pkas)),
            "other_pka_log10_competition": float(
                math.log10(sum(10.0 ** (value - float(stage1_pka)) for value in other_pkas))
            ) if other_pkas else 0.0,
            "experimental_anchor_pka": float(experimental_anchor) if experimental_anchor is not None else np.nan,
            "experimental_measurement_count": float(len(experimental_values)),
            "experimental_measurement_mean": float(np.mean(experimental_values)) if experimental_values else np.nan,
            "experimental_measurement_std": float(np.std(experimental_values)) if experimental_values else np.nan,
            "experimental_measurement_mad": experimental_mad,
            "experimental_measurement_robust_sigma": 1.4826 * experimental_mad if experimental_values else np.nan,
            "experimental_measurement_min": float(np.min(experimental_values)) if experimental_values else np.nan,
            "experimental_measurement_max": float(np.max(experimental_values)) if experimental_values else np.nan,
            "experimental_measurement_range": (
                float(np.max(experimental_values) - np.min(experimental_values))
                if experimental_values else np.nan
            ),
            "experimental_medium_confidence_fraction": (
                float(sum(value != "high" for value in experimental_confidences))
                / max(1, len(experimental_confidences))
            ),
            "experimental_values_json": json.dumps(experimental_values, separators=(",", ":")),
            "experimental_evidence_tier": (
                "gold"
                if experimental_evidence_tiers and all(value == "gold" for value in experimental_evidence_tiers)
                else "silver"
                if experimental_evidence_tiers
                else "none"
            ),
            "supervision_scope": "exact_site",
            "network_sites_json": str(row["sites_json"]),
            "microstate_nodes_json": str(row.get("microstate_nodes_json", "[]")),
            **_safe_stats(other_pkas, "other_stage1_pka"),
            **geometry.get(site_id, {}),
        }
        for family, count in other_family_counts.items():
            payload[f"other_family_count__{family}"] = float(count)
        if experimental_anchor is not None:
            payload["stage2_delta_target"] = float(experimental_anchor) - float(macro_pka)
        else:
            payload["stage2_delta_target"] = np.nan
        rows.append(payload)
    return rows


def build_stage2_site_table(
    network_df: pd.DataFrame,
    complex_only: bool = True,
    anchored_only: bool = False,
) -> pd.DataFrame:
    """Build the Stage 2 site table, optionally restricting molecules and anchors."""
    work = network_df
    if complex_only:
        work = work[pd.to_numeric(work["detected_site_count"], errors="coerce") > 1]
    rows = []
    for _, row in work.iterrows():
        rows.extend(molecule_network_to_site_rows(row))
    result = pd.DataFrame(rows)
    if anchored_only and not result.empty:
        result = result[result["experimental_anchor_pka"].notna()].copy()
    return result.reset_index(drop=True)


def build_stage2_weak_training_sites(
    network_df: pd.DataFrame,
    full_site_df: pd.DataFrame,
    total_measurement_weight: float = 0.20,
    assignment_temperature: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Marginalize eligible molecule-level pKas over complete candidate sites.

    Weak labels never unlock a family support gate and are not used for model
    validation.  Networks with unresolved ionizable contexts are retained in
    a blocked manifest because marginalizing over an incomplete candidate set
    would simply create another wrong-site label.
    """
    accepted: List[Dict] = []
    blocked: List[Dict] = []
    site_groups = (
        {
            str(key): part.copy()
            for key, part in full_site_df.groupby("molecule_id", sort=False)
        }
        if "molecule_id" in full_site_df.columns
        else {}
    )
    for _, molecule in network_df.iterrows():
        raw = molecule.get("weak_experimental_measurements_json", "[]")
        try:
            measurements = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            measurements = []
        if not measurements:
            continue
        molecule_id = str(molecule["molecule_id"])
        candidates = site_groups.get(molecule_id)
        unresolved_count = _safe_int(
            molecule.get("unresolved_ionizable_context_count", 0)
        )
        network_limited = _network_is_limited(molecule)
        if candidates is None or candidates.empty or unresolved_count or network_limited:
            reason = (
                "incomplete_candidate_space_unresolved_ionizable_context"
                if unresolved_count
                else "incomplete_or_missing_network"
            )
            for measurement in measurements:
                blocked.append({
                    "molecule_id": molecule_id,
                    "experimental_pka": measurement.get("experimental_pka"),
                    "source_file": measurement.get("source_file"),
                    "record_index": measurement.get("record_index"),
                    "blocked_reason": reason,
                    "unresolved_ionizable_contexts_json": molecule.get(
                        "unresolved_ionizable_contexts_json", "[]"
                    ),
                })
            continue

        seen = set()
        for measurement in measurements:
            value = float(measurement["experimental_pka"])
            lineage = (
                str(measurement.get("source_document_id", "")),
                round(value, 6),
            )
            if lineage in seen:
                continue
            seen.add(lineage)
            distance = np.abs(
                candidates["stage1_network_macro_pka"].to_numpy(dtype=float) - value
            )
            likelihood = np.exp(-distance / max(0.25, float(assignment_temperature)))
            probability = likelihood / max(float(likelihood.sum()), 1e-12)
            weak_id = str(measurement.get("transition_row_id", ""))
            for (_, candidate), candidate_probability in zip(candidates.iterrows(), probability):
                row = candidate.to_dict()
                row.update({
                    "experimental_anchor_pka": value,
                    "experimental_measurement_count": 1.0,
                    "experimental_measurement_mean": value,
                    "experimental_measurement_std": 0.0,
                    "experimental_measurement_mad": 0.0,
                    "experimental_measurement_robust_sigma": 0.0,
                    "experimental_measurement_min": value,
                    "experimental_measurement_max": value,
                    "experimental_measurement_range": 0.0,
                    "experimental_medium_confidence_fraction": 1.0,
                    "experimental_values_json": json.dumps([value], separators=(",", ":")),
                    "experimental_evidence_tier": "ambiguous",
                    "supervision_scope": "weak_molecule_macro_marginalized",
                    "weak_measurement_id": weak_id,
                    "weak_candidate_probability": float(candidate_probability),
                    "stage2_delta_target": value - float(candidate["stage1_network_macro_pka"]),
                    "sample_weight_raw": float(total_measurement_weight) * float(candidate_probability),
                    "sample_weight": float(total_measurement_weight) * float(candidate_probability),
                    "stage2_training_schema_version": STAGE2_TRAINING_SCHEMA_VERSION,
                    "target_definition": STAGE2_TARGET_DEFINITION,
                })
                accepted.append(row)
    return pd.DataFrame(accepted), pd.DataFrame(blocked)


def build_stage2_feature_matrix(site_df: pd.DataFrame, fp_bits: int = 512, fp_radius: int = 2) -> pd.DataFrame:
    """Encode site context while excluding experimental values and target leakage."""
    forbidden = {
        column for column in site_df.columns
        if column.startswith("experimental_")
    } | {"stage2_delta_target", "sample_weight", "sample_weight_raw"}
    metadata = {
        "molecule_key", "molecule_id", "site_id", "smiles", "mapped_smiles",
        "site_label", "site_family", "scaffold_group", "quarantine_reason",
        "full_site_position",
        "network_sites_json", "microstate_nodes_json",
    }
    numeric_columns = sorted(
        column
        for column in site_df.columns
        if column not in forbidden | metadata and pd.api.types.is_numeric_dtype(site_df[column])
    )
    numeric = site_df[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).reset_index(drop=True)

    descriptors = pd.DataFrame([molecule_descriptors(smiles) for smiles in site_df["smiles"]])
    fps = np.vstack([morgan_bits(smiles, nbits=fp_bits, radius=fp_radius) for smiles in site_df["smiles"]])
    fp_df = pd.DataFrame(fps, columns=[f"fp_{idx}" for idx in range(fp_bits)])
    label_df = pd.get_dummies(site_df["site_label"].astype(str), prefix="site_label")
    family_df = pd.get_dummies(site_df["site_family"].astype(str), prefix="site_family")
    features = pd.concat(
        [numeric, descriptors.reset_index(drop=True), label_df.reset_index(drop=True), family_df.reset_index(drop=True), fp_df],
        axis=1,
    )
    leaked = forbidden & set(features.columns)
    if leaked:
        raise AssertionError(f"Stage 2 target leakage columns present: {sorted(leaked)}")
    return features


def _measurement_weight(count: int, robust_sigma: float, medium_fraction: float) -> float:
    """Weight an anchor by replicate count, agreement, and assignment confidence."""
    count_factor = min(1.5, 1.0 + 0.20 * math.log2(max(1, count)))
    dispersion_factor = 1.0 / (1.0 + robust_sigma * robust_sigma)
    confidence_factor = 1.0 - 0.25 * min(1.0, max(0.0, medium_fraction))
    return float(np.clip(count_factor * dispersion_factor * confidence_factor, 0.25, 1.5))


def prepare_stage2_training_sites(
    site_df: pd.DataFrame,
    max_replicate_range: float = 2.0,
    min_supported_pka: float = -5.0,
    max_supported_pka: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Select consistent experimental anchors and quarantine conflicts."""
    anchored = site_df[site_df["experimental_anchor_pka"].notna()].copy()
    accepted_rows = []
    quarantine_rows = []
    for row in anchored.to_dict(orient="records"):
        value_range = float(row.get("experimental_measurement_range", np.nan))
        target = float(row["experimental_anchor_pka"])
        reason = None
        detail = ""
        if not np.isfinite(value_range):
            reason = "no_finite_experimental_measurement"
        elif value_range > float(max_replicate_range):
            reason = "replicate_range_exceeds_threshold"
            detail = f"range={value_range:.6g}; threshold={max_replicate_range:.6g}"
        elif not float(min_supported_pka) <= target <= float(max_supported_pka):
            reason = "target_outside_supported_pka_range"
            detail = (
                f"target={target:.6g}; supported=[{min_supported_pka:.6g},"
                f"{max_supported_pka:.6g}]"
            )
        if reason:
            quarantine_rows.append({
                "molecule_id": row["molecule_id"],
                "site_id": row["site_id"],
                "site_family": row["site_family"],
                "site_label": row["site_label"],
                "experimental_anchor_pka": target,
                "experimental_measurement_count": row["experimental_measurement_count"],
                "experimental_measurement_min": row["experimental_measurement_min"],
                "experimental_measurement_max": row["experimental_measurement_max"],
                "experimental_measurement_range": value_range,
                "experimental_values_json": row["experimental_values_json"],
                "quarantine_reason": reason,
                "detail": detail,
            })
            continue
        robust_sigma = float(row["experimental_measurement_robust_sigma"])
        medium_fraction = float(row["experimental_medium_confidence_fraction"])
        row["stage2_training_schema_version"] = STAGE2_TRAINING_SCHEMA_VERSION
        row["target_definition"] = STAGE2_TARGET_DEFINITION
        evidence_factor = 1.0 if str(row.get("experimental_evidence_tier")) == "gold" else 0.70
        row["sample_weight_raw"] = evidence_factor * _measurement_weight(
            int(row["experimental_measurement_count"]), robust_sigma, medium_fraction
        )
        accepted_rows.append(row)

    accepted = pd.DataFrame(accepted_rows)
    quarantine = pd.DataFrame(quarantine_rows)
    if not accepted.empty:
        accepted = accepted.sort_values(["molecule_id", "site_id"]).reset_index(drop=True)
        accepted["sample_weight"] = (
            accepted["sample_weight_raw"] / float(accepted["sample_weight_raw"].mean())
        )
    funnel = {
        "complex_network_sites": int(len(site_df)),
        "experimentally_anchored_sites": int(len(anchored)),
        "accepted_training_sites": int(len(accepted)),
        "quarantined_training_sites": int(len(quarantine)),
    }
    return accepted, quarantine, funnel


def validate_stage2_site_table(site_df: pd.DataFrame) -> None:
    """Reject malformed or physically inconsistent Stage 2 site records."""
    required = {
        "molecule_key",
        "site_id",
        "stage1_intrinsic_pka",
        "stage1_network_macro_pka",
        "stage2_delta_target",
    }
    missing = required - set(site_df)
    if missing:
        raise ValueError(f"Stage 2 site table missing columns: {sorted(missing)}")
    if site_df[["molecule_key", "site_id"]].duplicated().any():
        raise ValueError("Stage 2 site table contains duplicate molecule/site rows")
