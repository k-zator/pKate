#!/usr/bin/env python3
"""Build one-row-per-molecule protonation/tautomer microstate networks.

Experimental pKa values are retained as site-associated macroscopic anchors.
Unmeasured sites receive explicitly labelled Stage-1 priors. A binding
polynomial over the enumerated protonation coordinates estimates how local
transitions combine into macroscopic steps; serial transitions on one
amphoteric group share a coordinate. The result is never labelled experimental.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd  # type: ignore
from rdkit import Chem, rdBase  # type: ignore
from rdkit.Chem import inchi  # type: ignore

from functional_group_pka_analysis import (
    ACIDIC_FAMILIES,
    PAIR_TYPE_FAMILY_FORMS,
    classify_pair_type,
    expand_pair_site_transitions,
    normalize_conjugate_family,
    training_group_label,
)
from microstate_enumerator import (
    DEFAULT_MAX_PROTONATION_STATES,
    ensure_heavy_atom_maps,
    enumerate_joint_protonation_network,
    mapped_smiles,
)
from pka_evidence_policy import reference_prior, unresolved_ionizable_contexts
from substructure_match import find_sites_with_metadata, resolve_overlapping_sites
from stage1_network_inference import (
    load_stage1_bundle,
    predict_stage1_tasks,
    stage1_applicability,
)


DEFAULT_TRANSITIONS = "data/processed/pka_microstate_dataset.csv"
DEFAULT_STAGE1 = "data/processed/ml_models_experimental_only/stage1_intrinsic/stage1_intrinsic_model.pkl"
DEFAULT_STAGE1_METRICS = "data/processed/ml_models_experimental_only/stage1_intrinsic/metrics.json"
DEFAULT_OUTPUT = "data/processed/pka_molecule_microstate_network_dataset.csv"
DEFAULT_QUARANTINE = "data/processed/pka_molecule_microstate_network_quarantine.csv"
MOLECULE_NETWORK_DATASET_SCHEMA_VERSION = "2.9.0"


def stage1_local_prediction_for_site(
    site: Dict, raw_prediction: float, applicability_domain: str
) -> Tuple[float, str, str, Optional[float]]:
    """Choose the deployable local pKa without pretending zero-shot ML is trained.

    An exact-label-absent prediction extrapolates across functional-group
    identities. When an enabled transition-specific reference exists, that
    reference is the safer physical baseline. The raw model prediction remains
    useful only for labels for which no curated reference is available.
    """
    prior = reference_prior(site.get("label"), site.get("family"))
    if applicability_domain == "zero_shot_exact_label_absent" and prior is not None:
        return (
            float(prior.pka),
            "stage1_reference_prior_zero_shot_exact_label_fallback",
            "low_reference_prior_zero_shot_exact_label",
            float(prior.uncertainty),
        )
    confidence = (
        "very_low_zero_shot_exact_label"
        if applicability_domain == "zero_shot_exact_label_absent"
        else ""
    )
    return float(raw_prediction), "stage1_experimental_only_intrinsic_prediction", confidence, None


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _truthy(value: object, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _molecule_group_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid_" + hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:20]
    key = inchi.MolToInchiKey(mol)
    blocks = key.split("-")
    # The final InChIKey block encodes protonation. Keeping the first two
    # blocks preserves connectivity and stereochemistry while merging charge
    # and mobile-proton forms of the same molecule.
    return "-".join(blocks[:2])


def _resolved_pair_sites(mol: Chem.Mol, overlap_threshold: float) -> Tuple[List[Dict], List[Dict]]:
    resolved, _ = resolve_overlapping_sites(
        find_sites_with_metadata(mol),
        overlap_threshold=overlap_threshold,
    )
    pair_sites = []
    for site in resolved:
        pair_sites.extend(expand_pair_site_transitions(site))
    return resolved, pair_sites


def _representative_info(smiles: str, overlap_threshold: float) -> Dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "mol": None, "resolved": [], "pair_sites": []}
    resolved, pair_sites = _resolved_pair_sites(mol, overlap_threshold)
    unresolved_contexts = unresolved_ionizable_contexts(resolved, pair_sites)
    return {
        "smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        "mol": mol,
        "resolved": resolved,
        "pair_sites": pair_sites,
        "unresolved_ionizable_contexts": unresolved_contexts,
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "fragment_count": len(Chem.GetMolFrags(mol)),
    }


def _choose_representative(group: pd.DataFrame, cache: Dict[str, Dict], overlap_threshold: float) -> Optional[Dict]:
    counts = group["input_smiles"].value_counts().to_dict()
    measured_families = set(group["site_family"].astype(str))
    options = []
    for smiles in sorted(counts):
        info = cache.get(smiles)
        if info is None:
            info = _representative_info(smiles, overlap_threshold)
            cache[smiles] = info
        if info.get("mol") is None:
            continue
        represented_families = {site["pair_family"] for site in info["pair_sites"]}
        rank = (
            len(measured_families & represented_families),
            len(info["pair_sites"]),
            int(counts[smiles]),
            -abs(int(info["formal_charge"])),
            -int(info["fragment_count"]),
            info["smiles"],
        )
        options.append((rank, info))
    return max(options, key=lambda item: item[0])[1] if options else None


def _element_topology_query(mol: Chem.Mol) -> Chem.Mol:
    """Create an element-specific query that ignores charge, H count and bond order."""
    rw = Chem.RWMol()
    for atom in mol.GetAtoms():
        query_atom = Chem.Atom(atom.GetAtomicNum())
        query_atom.SetIsotope(atom.GetIsotope())
        rw.AddAtom(query_atom)
    for bond in mol.GetBonds():
        rw.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    params = Chem.AdjustQueryParameters()
    params.makeBondsGeneric = True
    return Chem.AdjustQueryProperties(rw.GetMol(), params)


def _measurement_target_indices(row: pd.Series) -> Tuple[Chem.Mol, set[int]]:
    source = Chem.MolFromSmiles(str(row["input_atom_mapped_smiles"]))
    if source is None:
        raise ValueError("input_atom_mapped_smiles is unreadable")
    target_maps = set(json.loads(str(row["site_atom_maps_json"])))
    target_indices = {
        atom.GetIdx()
        for atom in source.GetAtoms()
        if atom.GetAtomMapNum() in target_maps
    }
    return source, target_indices


def _map_measurement_to_site(
    row: pd.Series,
    representative: Chem.Mol,
    network_sites: Sequence[Dict],
) -> Tuple[Optional[str], str]:
    family = str(row["site_family"])
    candidates = [site for site in network_sites if site["family"] == family]
    if len(candidates) == 1:
        return str(candidates[0]["site_id"]), "unique_family_on_representative"
    family_reclassified = not candidates
    if family_reclassified:
        candidates = list(network_sites)

    # A curator-selected atom is stronger evidence than topology alone.  In a
    # symmetric molecule several same-family sites can have equally good
    # topological mappings; retain the reviewed drawing identity when its full
    # mapped site is present unchanged on the representative network.
    reviewer_atom_index = row.get("override_atom_index_raw")
    if not pd.isna(reviewer_atom_index):
        try:
            reviewed_maps = {
                int(value) for value in json.loads(str(row["site_atom_maps_json"]))
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            reviewed_maps = set()
        direct = [
            site for site in candidates
            if reviewed_maps
            and {int(value) for value in site.get("atom_maps", [])} == reviewed_maps
        ]
        if len(direct) == 1:
            if family_reclassified:
                new_family = str(direct[0]["family"])
                return str(direct[0]["site_id"]), (
                    "reviewer_atom_map_with_family_reclassification:"
                    f"{family}->{new_family}"
                )
            return str(direct[0]["site_id"]), "reviewer_selected_atom_map_on_representative"

    try:
        source, target_indices = _measurement_target_indices(row)
    except Exception as exc:
        return None, f"measurement_map_unreadable:{exc}"
    query = _element_topology_query(source)
    matches = representative.GetSubstructMatches(query, uniquify=False, maxMatches=256)
    if not matches:
        return None, "no_topological_mapping_to_representative"

    scores: Dict[str, float] = {}
    for match in matches:
        mapped_atoms = {int(match[idx]) + 1 for idx in target_indices if idx < len(match)}
        for site in candidates:
            site_atoms = set(int(value) for value in site["atom_maps"])
            union = mapped_atoms | site_atoms
            score = len(mapped_atoms & site_atoms) / max(1, len(union))
            scores[site["site_id"]] = max(scores.get(site["site_id"], 0.0), score)
    best = max(scores.values(), default=0.0)
    winners = sorted(site_id for site_id, score in scores.items() if score == best and score > 0.0)
    if len(winners) == 1:
        if family_reclassified:
            new_family = next(site["family"] for site in candidates if site["site_id"] == winners[0])
            return winners[0], f"topology_mapped_with_family_reclassification:{family}->{new_family}"
        return winners[0], "topology_mapped_between_protonation_forms"
    return None, "topological_site_mapping_ambiguous"


def _log10_add(a: float, b: float) -> float:
    if math.isinf(a) and a < 0:
        return b
    if math.isinf(b) and b < 0:
        return a
    high = max(a, b)
    return high + math.log10(10.0 ** (a - high) + 10.0 ** (b - high))


def independent_site_macro_pkas(site_pkas: Sequence[Tuple[str, float]]) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Return binding-polynomial macro steps and a deterministic site-to-step map."""
    ordered = sorted(((str(site_id), float(pka)) for site_id, pka in site_pkas), key=lambda x: (-x[1], x[0]))
    coefficients = [0.0] + [-math.inf] * len(ordered)
    for _, pka in ordered:
        for count in range(len(ordered), 0, -1):
            coefficients[count] = _log10_add(coefficients[count], coefficients[count - 1] + pka)

    steps = []
    site_map = {}
    for count in range(1, len(coefficients)):
        macro_pka = coefficients[count] - coefficients[count - 1]
        associated_site = ordered[count - 1][0]
        step = {
            "protonation_step": count,
            "from_protonated_site_count": count - 1,
            "to_protonated_site_count": count,
            "predicted_macro_pka": macro_pka,
            "rank_associated_site_id": associated_site,
        }
        steps.append(step)
        site_map[associated_site] = step
    return steps, site_map


def _annotate_node_populations(nodes: List[Dict], site_pka_map: Dict[str, float], ph: float) -> None:
    log_weights = []
    signatures = []
    for node in nodes:
        forms = node.get("site_forms", {})
        signatures.append(tuple(sorted(forms.items())))
        if set(forms) != set(site_pka_map):
            log_weights.append(-math.inf)
            continue
        log_weights.append(sum(site_pka_map[site_id] - ph for site_id, form in forms.items() if form == "acid_form"))
    finite = [value for value in log_weights if math.isfinite(value)]
    if not finite:
        return
    high = max(finite)
    scaled = [0.0 if not math.isfinite(value) else 10.0 ** (value - high) for value in log_weights]
    # Alternate atom-centered drawings of the same acid/base configuration
    # share that configuration's weight. Enumeration count is not an energy
    # model and must not create a spurious statistical factor.
    multiplicity = {signature: signatures.count(signature) for signature in set(signatures)}
    scaled = [
        value / max(1, multiplicity[signature])
        for value, signature in zip(scaled, signatures)
    ]
    total = sum(scaled)
    for node, value in zip(nodes, scaled):
        node["independent_site_population_at_ph"] = value / total if total else 0.0


def coupled_network_thermodynamics(
    network: Dict[str, object],
    site_pka_map: Dict[str, float],
    ph: float,
) -> Tuple[List[Dict], Dict[str, Dict]]:
    """Annotate populations and macro-pKas for the enumerated state graph.

    The binding coefficients are propagated over actual one-proton edges, so
    two serial transitions on an amphoteric ring cannot contribute the
    fictitious fourth configuration of an independent two-switch model.
    """
    if (
        bool(network.get("protonation_state_enumeration_truncated", False))
        or bool(network.get("incomplete_site_ids", []))
    ):
        return [], {}
    nodes = list(network.get("nodes", []))
    edges = list(network.get("edges", []))
    sites = list(network.get("sites", []))
    if set(site_pka_map) != {str(site["site_id"]) for site in sites}:
        return [], {}

    node_lookup = {str(node["node_id"]): node for node in nodes}
    config_for_node = {
        node_id: tuple(sorted((str(key), int(value)) for key, value in node.get("group_levels", {}).items()))
        for node_id, node in node_lookup.items()
    }
    proton_count = {}
    for node_id, config in config_for_node.items():
        count = int(node_lookup[node_id]["protonated_site_count"])
        proton_count.setdefault(config, count)

    adjacency: Dict[Tuple[Tuple[str, int], ...], List[Tuple[Tuple[Tuple[str, int], ...], float]]] = {}
    for edge in edges:
        site_id = str(edge["site_id"])
        acid = config_for_node.get(str(edge["acid_node_id"]))
        base = config_for_node.get(str(edge["base_node_id"]))
        if acid is None or base is None:
            continue
        pka = float(site_pka_map[site_id])
        adjacency.setdefault(base, []).append((acid, pka))
        adjacency.setdefault(acid, []).append((base, -pka))

    if not proton_count:
        return [], {}
    minimum_count = min(proton_count.values())
    roots = sorted(config for config, count in proton_count.items() if count == minimum_count)
    coefficients: Dict[Tuple[Tuple[str, int], ...], float] = {root: 0.0 for root in roots}
    queue = list(roots)
    while queue:
        config = queue.pop(0)
        for neighbor, delta in adjacency.get(config, []):
            proposed = coefficients[config] + delta
            if neighbor not in coefficients:
                coefficients[neighbor] = proposed
                queue.append(neighbor)

    if set(coefficients) != set(proton_count):
        return [], {}
    by_count: Dict[int, float] = {}
    for config, coefficient in coefficients.items():
        count = proton_count[config]
        by_count[count] = _log10_add(by_count.get(count, -math.inf), coefficient)

    ordered_sites = sorted(site_pka_map.items(), key=lambda item: (-float(item[1]), str(item[0])))
    steps = []
    site_map = {}
    counts = sorted(by_count)
    for step_index, (lower_count, upper_count) in enumerate(zip(counts, counts[1:]), start=1):
        if upper_count != lower_count + 1:
            continue
        associated_site = str(ordered_sites[step_index - 1][0])
        macro_pka = by_count[upper_count] - by_count[lower_count]
        step = {
            "protonation_step": step_index,
            "from_protonated_site_count": lower_count,
            "to_protonated_site_count": upper_count,
            "predicted_macro_pka": macro_pka,
            "rank_associated_site_id": associated_site,
        }
        steps.append(step)
        site_map[associated_site] = step

    config_log_weights = {
        config: coefficient - proton_count[config] * float(ph)
        for config, coefficient in coefficients.items()
    }
    high = max(config_log_weights.values())
    scaled = {config: 10.0 ** (value - high) for config, value in config_log_weights.items()}
    total = sum(scaled.values())
    multiplicity = {
        config: sum(candidate == config for candidate in config_for_node.values())
        for config in set(config_for_node.values())
    }
    for node_id, node in node_lookup.items():
        config = config_for_node[node_id]
        population = scaled[config] / total / max(1, multiplicity[config]) if total else 0.0
        node["network_population_at_ph"] = population
        # Compatibility name retained for current Stage 3 readers.
        node["independent_site_population_at_ph"] = population
    return steps, site_map


def build_molecule_dataset(
    transitions_path: str,
    stage1_model_path: str,
    stage1_metrics_path: str,
    output_path: str,
    quarantine_path: str,
    ph: float = 7.4,
    overlap_threshold: float = 0.5,
    max_protonation_states: int = DEFAULT_MAX_PROTONATION_STATES,
    max_tautomers_per_state: int = 4,
    stage1_workers: int = 4,
    allow_stage1_detector_mismatch: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    transitions = pd.read_csv(transitions_path, low_memory=False)
    if set(transitions["measurement_method"].astype(str).str.lower()) != {"experimental"}:
        raise ValueError("Transition dataset must be experimental-only")
    transitions["molecule_group_key"] = transitions["input_smiles"].map(_molecule_group_key)
    detector_versions = sorted(set(
        transitions.get(
            "functional_group_detector_schema_version",
            pd.Series(["unknown"] * len(transitions), index=transitions.index),
        ).astype(str)
    ))

    stage1_bundle = load_stage1_bundle(stage1_model_path, require_experimental_only=True)
    bundle_detector_versions = {
        str(value) for value in stage1_bundle.get("functional_group_detector_schema_versions", [])
    }
    transition_detector_versions = set(detector_versions)
    if (
        bundle_detector_versions
        and bundle_detector_versions != transition_detector_versions
        and not allow_stage1_detector_mismatch
    ):
        raise ValueError(
            "Stage 1 detector provenance does not match the transition dataset: "
            f"bundle={sorted(bundle_detector_versions)} transitions={sorted(transition_detector_versions)}"
        )
    with open(stage1_metrics_path, encoding="utf-8") as handle:
        stage1_metrics = json.load(handle)
    scaffold_mae = float(stage1_metrics["scaffold"]["mae"])
    feature_columns = set(stage1_bundle["feature_columns"])
    global_calibration = stage1_bundle.get("error_calibration_global", {})
    family_calibration = stage1_bundle.get("error_calibration_by_family", {})
    stage1_sha256 = _sha256_file(stage1_model_path)

    cache: Dict[str, Dict] = {}
    work_items = []
    quarantine_rows = []
    stage1_tasks = []

    for group_key, group in transitions.groupby("molecule_group_key", sort=True):
        representative = _choose_representative(group, cache, overlap_threshold)
        if representative is None:
            quarantine_rows.append({
                "molecule_group_key": group_key,
                "reason": "no_readable_representative",
                "experimental_transition_count": int(len(group)),
                "source_transition_row_ids_json": _json(sorted(group["dataset_row_id"].astype(str))),
            })
            continue
        network = enumerate_joint_protonation_network(
            representative["mol"],
            representative["pair_sites"],
            max_protonation_states=max_protonation_states,
            max_tautomers_per_state=max_tautomers_per_state,
        )
        if not network["sites"] or not network["nodes"]:
            quarantine_rows.append({
                "molecule_group_key": group_key,
                "representative_smiles": representative["smiles"],
                "reason": "joint_network_empty",
                "experimental_transition_count": int(len(group)),
                "source_transition_row_ids_json": _json(sorted(group["dataset_row_id"].astype(str))),
            })
            continue

        mapped_rep = ensure_heavy_atom_maps(representative["mol"])
        measurements_by_site: Dict[str, List[Dict]] = {site["site_id"]: [] for site in network["sites"]}
        unattached = []
        weak_measurements = []
        for _, row in group.iterrows():
            site_id, mapping_basis = _map_measurement_to_site(row, representative["mol"], network["sites"])
            measurement = {
                "transition_row_id": str(row["dataset_row_id"]),
                "experimental_pka": float(row["experimental_pka"]),
                "source_file": str(row["source_file"]),
                "record_index": int(row["record_index"]),
                "source_assay_id": str(row.get("source_assay_id", "")),
                "source_document_id": str(row.get("source_document_id", "")),
                "source_molecule_id": str(row.get("source_molecule_id", "")),
                "source_original_smiles": str(row.get("source_original_smiles", "")),
                "structure_provenance": str(row.get("structure_provenance", "unknown")),
                "reported_site_label": str(row["site_label"]),
                "site_assignment_confidence": (
                    "medium"
                    if mapping_basis.startswith("topology_mapped_with_family_reclassification")
                    else str(row["site_assignment_confidence"])
                ),
                "structural_site_assignment_confidence": (
                    "medium"
                    if mapping_basis.startswith("topology_mapped_with_family_reclassification")
                    else str(row.get("structural_site_assignment_confidence", row["site_assignment_confidence"]))
                ),
                "source_site_evidence": str(row.get("source_site_evidence", "unknown")),
                "source_site_evidence_confidence": str(
                    row.get("source_site_evidence_confidence", "unknown")
                ),
                "network_mapping_basis": mapping_basis,
                "measurement_scope": "experimental_macroscopic_site_associated",
                "evidence_tier": str(row.get("evidence_tier", "legacy_exact")),
                "evidence_tier_reason": str(row.get("evidence_tier_reason", "legacy_dataset")),
                "experimental_site_attribution_confidence": str(
                    row.get("experimental_site_attribution_confidence", row.get("source_site_evidence_confidence", "unknown"))
                ),
                "transition_identity_confidence": str(
                    row.get("transition_identity_confidence", "unknown")
                ),
                "measurement_conditions_confidence": str(
                    row.get("measurement_conditions_confidence", "unknown")
                ),
                "reference_plausibility_status": str(
                    row.get("reference_plausibility_status", "not_assessed")
                ),
                "stage1_local_supervision_eligible": _truthy(
                    row.get("stage1_local_supervision_eligible", True), default=True
                ),
                "stage2_exact_site_supervision_eligible": _truthy(
                    row.get("stage2_exact_site_supervision_eligible", True), default=True
                ),
                "stage2_weak_molecule_supervision_eligible": _truthy(
                    row.get("stage2_weak_molecule_supervision_eligible", False)
                ),
            }
            if site_id is None:
                unattached.append(measurement)
                if measurement["stage2_weak_molecule_supervision_eligible"]:
                    weak_measurements.append(measurement)
            elif measurement["stage2_exact_site_supervision_eligible"]:
                measurements_by_site[site_id].append(measurement)
            else:
                measurement["candidate_site_id"] = site_id
                measurement["measurement_scope"] = "experimental_macroscopic_molecule_weak"
                weak_measurements.append(measurement)

        molecule_id = "mol_" + hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:16]
        for site in network["sites"]:
            site_id = site["site_id"]
            site["acid_labels"] = sorted(PAIR_TYPE_FAMILY_FORMS[site["family"]]["acid_form"])
            site["base_labels"] = sorted(PAIR_TYPE_FAMILY_FORMS[site["family"]]["base_form"])
            site["experimental_measurements"] = measurements_by_site[site_id]
            site["weak_experimental_measurements"] = [
                measurement
                for measurement in weak_measurements
                if measurement.get("candidate_site_id") == site_id
            ]
            task_id = f"{molecule_id}:{site_id}"
            center_maps = site.get("center_maps", [])
            pka_type = "acidic" if site["family"] in ACIDIC_FAMILIES else "basic"
            task = {
                "task_id": task_id,
                "molecule_id": molecule_id,
                "site_id": site_id,
                "smiles": representative["smiles"],
                "candidate_label": site["label"],
                "pka_type_canonical": pka_type,
                "atom_index_raw": (int(center_maps[0]) - 1) if center_maps else None,
            }
            stage1_tasks.append(task)
            site["stage1_task_id"] = task_id

        work_items.append({
            "molecule_id": molecule_id,
            "group_key": group_key,
            "group": group,
            "representative": representative,
            "representative_mapped_smiles": mapped_smiles(mapped_rep),
            "network": network,
            "unattached_measurements": unattached,
            "weak_measurements": weak_measurements,
        })

    print(f"molecule_groups={len(work_items)} stage1_all_site_tasks={len(stage1_tasks)}")
    stage1_predictions = predict_stage1_tasks(stage1_tasks, stage1_model_path, stage1_workers)

    rows = []
    for item in work_items:
        group = item["group"]
        network = item["network"]
        site_pka_map: Dict[str, float] = {}
        measured_site_count = 0
        for site in network["sites"]:
            measurements = site["experimental_measurements"]
            group_feature = f"group_{training_group_label(site['label'])}"
            site["stage1_group_feature_present"] = group_feature in feature_columns
            site["stage1_scaffold_validation_mae"] = scaffold_mae
            prediction = stage1_predictions.get(site.pop("stage1_task_id", ""))
            if prediction is None:
                site.update({
                    "stage1_intrinsic_pka": None,
                    "local_pka_used": None,
                    "local_pka_provenance": "missing_stage1_prediction",
                    "local_pka_confidence": "none",
                })
                continue
            label_calibration = stage1_bundle.get("error_calibration_by_label", {})
            calibration = label_calibration.get(
                site["label"], family_calibration.get(site["family"], global_calibration)
            )
            calibration_rows = int(calibration.get("rows", 0))
            expected_abs_error = float(calibration.get("mae", scaffold_mae))
            error_p90 = float(calibration.get("p90_abs_error", expected_abs_error))
            error_p95 = float(calibration.get("p95_abs_error", error_p90))
            applicability = stage1_applicability(
                str(item["representative"]["smiles"]),
                str(site["label"]),
                stage1_bundle,
            )
            applicability_domain = str(applicability["stage1_applicability_domain"])
            local_pka, local_pka_provenance, confidence, reference_uncertainty = (
                stage1_local_prediction_for_site(
                    site, float(prediction), applicability_domain
                )
            )
            if not confidence and applicability_domain == "interpolation_high_similarity" and error_p90 <= 2.0:
                confidence = "high_empirical_support"
            elif not confidence and applicability_domain == "interpolation_limited_support" and error_p90 <= 3.0:
                confidence = "medium_empirical_support"
            elif not confidence:
                confidence = "low_empirical_support"
            uncertainty_multiplier = (
                2.0 if "zero_shot" in applicability_domain
                else 1.5 if "extrapolation" in applicability_domain
                else 1.0
            )
            pka_half_width = max(
                0.5,
                error_p95 * uncertainty_multiplier,
                float(reference_uncertainty or 0.0),
            )
            site.update({
                "stage1_intrinsic_pka": local_pka,
                "local_pka_used": local_pka,
                "local_pka_provenance": local_pka_provenance,
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
                "stage1_prediction_outside_training_filter_range": not (-5.0 < local_pka < 40.0),
                **applicability,
                "stage1_predicted_pka_ci_low": local_pka - pka_half_width,
                "stage1_predicted_pka_ci_high": local_pka + pka_half_width,
                "stage1_uncertainty_basis": (
                    "transition reference uncertainty and OOF p95; exact label absent"
                    if reference_uncertainty is not None
                    else "exact-label interpolation OOF p95 widened by applicability domain"
                ),
            })
            if measurements:
                measured_site_count += 1
                values = sorted(float(record["experimental_pka"]) for record in measurements)
                experimental_anchor = float(np.median(values))
                site.update({
                    "experimental_anchor_pka": experimental_anchor,
                    "experimental_pka_values": values,
                    "experimental_anchor_confidence": "high" if all(record["site_assignment_confidence"] == "high" for record in measurements) else "medium",
                    "experimental_anchor_structural_confidence": (
                        "high"
                        if all(record["structural_site_assignment_confidence"] == "high" for record in measurements)
                        else "medium"
                    ),
                    "experimental_anchor_source_evidence_confidence": (
                        "high"
                        if all(record["source_site_evidence_confidence"] == "high" for record in measurements)
                        else "medium"
                        if any(record["source_site_evidence_confidence"] in {"high", "medium"} for record in measurements)
                        else "low"
                        if any(record["source_site_evidence_confidence"] == "low" for record in measurements)
                        else "none"
                    ),
                    "experimental_anchor_evidence_tier": (
                        "gold"
                        if all(record.get("evidence_tier") == "gold" for record in measurements)
                        else "silver"
                        if all(record.get("evidence_tier") in {"gold", "silver", "legacy_exact"} for record in measurements)
                        else "ambiguous"
                    ),
                    "anchor_caveat": "reported pKa is a site-associated macroscopic label and is not used as a Stage 1 input",
                })
            else:
                site["experimental_anchor_pka"] = None
            site_pka_map[site["site_id"]] = local_pka

        macro_steps = []
        site_step_map = {}
        if (
            len(site_pka_map) == len(network["sites"])
            and not network["protonation_state_enumeration_truncated"]
            and not network["incomplete_site_ids"]
        ):
            macro_steps, site_step_map = coupled_network_thermodynamics(
                network, site_pka_map, ph
            )
        for site in network["sites"]:
            macro_step = site_step_map.get(site["site_id"])
            if macro_step is not None:
                site["rank_associated_macro_step"] = macro_step["protonation_step"]
                site["rank_associated_macro_pka"] = macro_step["predicted_macro_pka"]
                site["predicted_macro_shift_from_local_pka"] = (
                    macro_step["predicted_macro_pka"] - float(site["local_pka_used"])
                )

        site_lookup = {site["site_id"]: site for site in network["sites"]}
        for edge in network["edges"]:
            site = site_lookup[edge["site_id"]]
            edge.update({
                "local_pka_used": site.get("local_pka_used"),
                "local_pka_provenance": site.get("local_pka_provenance"),
                "local_pka_confidence": site.get("local_pka_confidence"),
                "context_model": "shared_transition_pka_across_enumerated_context_edges",
            })

        predicted_only = len(network["sites"]) - measured_site_count
        if network["protonation_state_enumeration_truncated"] or network["incomplete_site_ids"]:
            confidence = "limited"
        elif predicted_only:
            confidence = "mixed_experimental_and_stage1"
        else:
            confidence = "experimental_anchors_complete_for_detected_sites"
        rows.append({
            "dataset_schema_version": MOLECULE_NETWORK_DATASET_SCHEMA_VERSION,
            "functional_group_detector_schema_versions_json": _json(detector_versions),
            "rdkit_version": rdBase.rdkitVersion,
            "molecule_id": item["molecule_id"],
            "molecule_group_key": item["group_key"],
            "molecule_grouping_rule": "standard_InChIKey_first_two_blocks; protonation block removed; stereochemistry retained",
            "representative_input_smiles": item["representative"]["smiles"],
            "representative_atom_mapped_smiles": item["representative_mapped_smiles"],
            "source_input_smiles_json": _json(sorted(set(group["input_smiles"].astype(str)))),
            "source_files_json": _json(sorted(set(group["source_file"].astype(str)))),
            "source_transition_row_ids_json": _json(sorted(set(group["dataset_row_id"].astype(str)))),
            "experimental_transition_count": int(len(group)),
            "detected_site_count": len(network["sites"]),
            "experimentally_anchored_site_count": measured_site_count,
            "stage1_prior_only_site_count": predicted_only,
            "unattached_experimental_measurement_count": len(item["unattached_measurements"]),
            "unattached_experimental_measurements_json": _json(item["unattached_measurements"]),
            "weak_experimental_measurement_count": len(item["weak_measurements"]),
            "weak_experimental_measurements_json": _json(item["weak_measurements"]),
            "unresolved_ionizable_context_count": len(
                item["representative"].get("unresolved_ionizable_contexts", [])
            ),
            "unresolved_ionizable_contexts_json": _json(
                item["representative"].get("unresolved_ionizable_contexts", [])
            ),
            "network_confidence": confidence,
            "protonation_state_count": len(network["nodes"]),
            "protonation_edge_count": len(network["edges"]),
            "protonation_state_enumeration_truncated": network["protonation_state_enumeration_truncated"],
            "protonation_state_cap": int(network["max_protonation_states"]),
            "incomplete_site_ids_json": _json(network["incomplete_site_ids"]),
            "tautomer_cap_per_protonation_state": max_tautomers_per_state,
            "sites_json": _json(network["sites"]),
            "microstate_nodes_json": _json(network["nodes"]),
            "microstate_edges_json": _json(network["edges"]),
            "independent_site_macro_pka_steps_json": _json(macro_steps),
            "baseline_macro_pka_steps_json": _json(macro_steps),
            "population_ph": ph,
            "thermodynamic_model": "enumerated coupled-coordinate binding polynomial; serial amphoteric transitions share one coordinate; alternate drawings share configuration weight",
            "thermodynamic_caveat": "blind Stage 1 local pKas only; no Stage 2 site-site coupling correction in this baseline file",
            "stage1_model_sha256": stage1_sha256,
            "stage1_schema_version": stage1_bundle.get("stage1_schema_version"),
            "stage1_target_definition": stage1_bundle.get("target_definition"),
            "stage1_training_measurement_method": stage1_bundle.get("training_measurement_method"),
            "stage1_training_rows": stage1_bundle.get("training_rows"),
            "stage1_training_measurement_count": stage1_bundle.get("training_measurement_count"),
            "stage1_training_dataset_sha256": stage1_bundle.get("training_dataset_sha256"),
            "stage1_trained_on_all_eligible_rows": stage1_bundle.get("trained_on_all_eligible_rows"),
            "stage1_scaffold_validation_mae": scaffold_mae,
            "marvin_values_used": False,
            "epik_values_used": False,
        })

    dataset = pd.DataFrame(rows).sort_values("molecule_id").reset_index(drop=True)
    quarantine_columns = [
        "molecule_group_key",
        "representative_smiles",
        "reason",
        "experimental_transition_count",
        "source_transition_row_ids_json",
    ]
    quarantine = pd.DataFrame(quarantine_rows, columns=quarantine_columns)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    quarantine.to_csv(quarantine_path, index=False)
    return dataset, quarantine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", default=DEFAULT_TRANSITIONS)
    parser.add_argument("--stage1-model", default=DEFAULT_STAGE1)
    parser.add_argument("--stage1-metrics", default=DEFAULT_STAGE1_METRICS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-protonation-states",
        type=int,
        default=DEFAULT_MAX_PROTONATION_STATES,
    )
    parser.add_argument("--max-tautomers-per-state", type=int, default=4)
    parser.add_argument("--stage1-workers", type=int, default=4)
    parser.add_argument(
        "--allow-stage1-detector-mismatch",
        action="store_true",
        help="Topology-refresh bootstrap only; final network builds must not use this option",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, quarantine = build_molecule_dataset(
        transitions_path=args.transitions,
        stage1_model_path=args.stage1_model,
        stage1_metrics_path=args.stage1_metrics,
        output_path=args.output,
        quarantine_path=args.quarantine,
        ph=args.ph,
        overlap_threshold=args.overlap_threshold,
        max_protonation_states=args.max_protonation_states,
        max_tautomers_per_state=args.max_tautomers_per_state,
        stage1_workers=args.stage1_workers,
        allow_stage1_detector_mismatch=bool(args.allow_stage1_detector_mismatch),
    )
    print(f"molecule_rows={len(dataset)} quarantine_rows={len(quarantine)} output={args.output}")


if __name__ == "__main__":
    main()
