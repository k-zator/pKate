#!/usr/bin/env python3
"""Prepare provenance-safe pKahub supplements for canonical pKate training.

Only experimental pKa scalars and input structures are consumed.  pKahub's
Epik-derived charge transitions and microspecies columns are deliberately
ignored.  Structurally unique one-coordinate molecules become conservative
Stage 1 candidates; multi-coordinate molecules are emitted separately as
unordered macroscopic pKa sets for a Stage 2/3 set-matching objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi

from audit_pkahub_transition_assignability import _molecule_assignment
from functional_group_pka_analysis import ACIDIC_FAMILIES, expand_pair_site_transitions
from microstate_enumerator import ensure_heavy_atom_maps, mapped_smiles
from pka_evidence_policy import (
    EVIDENCE_POLICY_SCHEMA_VERSION,
    assess_measurement_evidence,
)
from stage1_canonical_data import (
    STAGE1_TRAINING_SCHEMA_VERSION,
    TARGET_DEFINITION,
    canonical_structure_key,
    scaffold_group_key,
)
from substructure_match import (
    FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION,
    find_sites_with_metadata,
    resolve_overlapping_sites,
)


DEFAULT_INPUT = "data/raw/external_datasets/pkahub/all_datasets.tsv"
DEFAULT_EXISTING_STAGE1 = (
    "data/processed/ml_models_experimental_only/stage1_intrinsic/"
    "stage1_training_sites.csv"
)
DEFAULT_EXISTING_NETWORK = "data/processed/pka_molecule_microstate_network_dataset.csv"
DEFAULT_OUTPUT_DIR = "data/processed/pkahub"
BENCHMARK_SOURCES = frozenset({"SAMPL6", "SAMPL7", "SAMPL8", "euroSAMPL1"})
PKA_RANGE = (-5.0, 20.0)
MAX_SINGLE_TRANSITION_REPLICATE_RANGE = 2.0
MAX_MACRO_TRANSITIONS = 6


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def molecule_group_key(smiles: str) -> str:
    """Return the protonation-insensitive, stereochemistry-preserving key."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "invalid::" + hashlib.sha256(str(smiles).encode()).hexdigest()[:20]
    value = inchi.MolToInchiKey(mol)
    return "-".join(value.split("-")[:2])


def _existing_keys(
    stage1_path: str | Path,
    network_path: str | Path,
    *,
    excluded_training_cohorts: Sequence[str] = (),
    excluded_source_prefixes: Sequence[str] = (),
) -> set[str]:
    """Return pre-existing identities without treating a supplement as itself.

    A prepared network may already contain the supplement being regenerated.
    Callers can exclude rows whose complete provenance belongs to that same
    supplement, making preparation idempotent while retaining true overlaps
    with canonical or other external sources.
    """
    keys: set[str] = set()
    stage1 = Path(stage1_path)
    if stage1.is_file():
        frame = pd.read_csv(stage1)
        if "training_cohort" in frame and excluded_training_cohorts:
            frame = frame.loc[
                ~frame["training_cohort"].astype(str).isin(excluded_training_cohorts)
            ]
        keys.update(frame["smiles"].astype(str).map(molecule_group_key))
    network = Path(network_path)
    if network.is_file():
        frame = pd.read_csv(network)
        for row in frame.itertuples(index=False):
            sources: list[str] = []
            if hasattr(row, "source_files_json"):
                try:
                    sources = [
                        str(value) for value in json.loads(row.source_files_json)
                    ]
                except (TypeError, ValueError, json.JSONDecodeError):
                    sources = []
            self_only = bool(sources) and bool(excluded_source_prefixes) and all(
                any(source.startswith(prefix) for prefix in excluded_source_prefixes)
                for source in sources
            )
            if not self_only:
                keys.add(molecule_group_key(str(row.representative_input_smiles)))
    return keys


def _independent_values(group: pd.DataFrame) -> np.ndarray:
    """Collapse exact copied scalars before estimating a pKa target."""
    values = pd.to_numeric(group["pka_value"], errors="coerce").dropna()
    return np.asarray(sorted(set(round(float(value), 6) for value in values)), dtype=float)


def _macro_clusters(values: Iterable[float], maximum_span: float = 2.0) -> List[List[float]]:
    """Conservatively cluster repeated macroscopic measurements in one dimension."""
    ordered = sorted(set(round(float(value), 6) for value in values))
    if not ordered:
        return []
    clusters: List[List[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        current = clusters[-1]
        if value - current[-1] <= 1.0 and value - current[0] <= maximum_span:
            current.append(value)
        else:
            clusters.append([value])
    return clusters


def _resolved_transitions(smiles: str) -> tuple[Chem.Mol, list[Dict], list[Dict]]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("invalid_smiles")
    resolved, _ = resolve_overlapping_sites(find_sites_with_metadata(mol), 0.75)
    transitions = [
        transition
        for site in resolved
        for transition in expand_pair_site_transitions(site)
    ]
    return mol, resolved, transitions


def _stage1_row(group: pd.DataFrame, assignment: pd.Series) -> tuple[Dict | None, Dict]:
    smiles = str(assignment["canonical_smiles"])
    mol, _, transitions = _resolved_transitions(smiles)
    values = _independent_values(group)
    base = {
        "molid": str(group.iloc[0]["molid"]),
        "smiles": smiles,
        "measurement_count": int(len(group)),
        "independent_value_count": int(len(values)),
    }
    if len(transitions) != 1:
        return None, {**base, "reason": "not_exactly_one_supported_transition"}
    if not len(values):
        return None, {**base, "reason": "no_finite_pka"}
    value_range = float(np.max(values) - np.min(values))
    if value_range > MAX_SINGLE_TRANSITION_REPLICATE_RANGE:
        return None, {
            **base,
            "reason": "replicate_range_exceeds_threshold",
            "measurement_range": value_range,
        }
    transition = transitions[0]
    pka = float(np.median(values))
    temperatures = pd.to_numeric(group.get("temperature"), errors="coerce").dropna()
    temperature = "" if temperatures.empty else f"{float(np.median(temperatures)):.6g}"
    evidence = assess_measurement_evidence(
        pka=pka,
        label=transition["label"],
        family=transition["pair_family"],
        structural_mapping_confidence="high",
        source_site_evidence_confidence="none",
        unresolved_contexts=[],
        source_temperature=temperature,
        source_original_smiles=smiles,
        input_smiles=smiles,
    )
    if evidence["evidence_tier"] not in {"gold", "silver"}:
        return None, {
            **base,
            "reason": "evidence_policy_not_stage1_eligible",
            "evidence_tier": evidence["evidence_tier"],
            "evidence_tier_reason": evidence["evidence_tier_reason"],
            "reference_plausibility_status": evidence["reference_plausibility_status"],
        }
    mapped = ensure_heavy_atom_maps(mol)
    center = sorted(transition.get("center_atom_set", set()))
    if not center:
        center = sorted(transition.get("atom_set", set()))
    atom_index = int(center[0]) if center else np.nan
    mad = float(np.median(np.abs(values - pka)))
    robust_sigma = float(1.4826 * mad)
    key = molecule_group_key(smiles)
    transition_id = "pkahub_" + hashlib.sha256(
        f"{key}|{transition['label']}".encode()
    ).hexdigest()[:16]
    sources = sorted(set(group["data_source"].astype(str)))
    raw_ids = sorted(set(group["rawdataID"].astype(str)))
    row = {
        "stage1_training_schema_version": STAGE1_TRAINING_SCHEMA_VERSION,
        "evidence_policy_schema_version": EVIDENCE_POLICY_SCHEMA_VERSION,
        "functional_group_detector_schema_versions_json": _json([
            FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION
        ]),
        "target_definition": TARGET_DEFINITION,
        "molecule_key": key,
        "molecule_id": "pkahub_" + hashlib.sha256(key.encode()).hexdigest()[:16],
        "structure_key": canonical_structure_key(smiles),
        "scaffold_group": scaffold_group_key(smiles),
        "smiles": smiles,
        "mapped_smiles": mapped_smiles(mapped),
        "site_id": transition_id,
        "candidate_label": str(transition["label"]),
        "final_group_label": str(transition["label"]),
        "site_family": str(transition["pair_family"]),
        "pka_type_canonical": (
            "acidic" if transition["pair_family"] in ACIDIC_FAMILIES else "basic"
        ),
        "atom_index_raw": atom_index,
        "pka_value": pka,
        "experimental_anchor_pka": pka,
        "experimental_measurement_count": int(len(group)),
        "experimental_eligible_measurement_count": int(len(values)),
        "experimental_independent_lineage_count": int(len(values)),
        "experimental_measurement_mean": float(np.mean(values)),
        "experimental_measurement_std": float(np.std(values)),
        "experimental_measurement_mad": mad,
        "experimental_measurement_robust_sigma": robust_sigma,
        "experimental_measurement_min": float(np.min(values)),
        "experimental_measurement_max": float(np.max(values)),
        "experimental_measurement_range": value_range,
        "experimental_medium_confidence_fraction": 1.0,
        "experimental_evidence_tier": str(evidence["evidence_tier"]),
        "stage1_reference_pka": evidence["reference_pka"],
        "stage1_reference_uncertainty": evidence["reference_uncertainty"],
        "stage1_reference_transition_definition": evidence[
            "reference_transition_definition"
        ],
        "stage1_reference_provenance": evidence["reference_provenance"],
        "experimental_source_site_evidence_json": _json([
            "structurally_unique_transition; source_is_molecule_level"
        ]),
        "experimental_source_site_evidence_confidences_json": _json(["none"]),
        "sample_weight_raw": float(0.70 / (1.0 + robust_sigma * robust_sigma)),
        "sample_weight": float(0.70 / (1.0 + robust_sigma * robust_sigma)),
        "measurement_method": "experimental",
        "target_aggregation": "median_of_unique_reported_values",
        "experimental_transition_ids_json": _json(raw_ids),
        "experimental_source_files_json": _json([
            f"pkahub:{source}" for source in sources
        ]),
        "experimental_record_indices_json": _json([]),
        "training_cohort": "pkahub_clean_single_transition",
        "assignment_teacher_used": False,
        "pkahub_assignment_columns_ignored": True,
        "pkahub_molids_json": _json(sorted(set(group["molid"].astype(str)))),
        "pkahub_data_sources_json": _json(sources),
        "pkahub_primary_sources_json": _json(sorted(set(
            group.get("primary_source", pd.Series(dtype=str)).dropna().astype(str)
        ))),
    }
    return row, {}


def _macro_transition_placeholders(macro: pd.DataFrame) -> pd.DataFrame:
    """Represent unordered macro sets as weak-only network-builder inputs.

    The chosen structural site is merely a carrier that lets the molecule enter
    network construction.  Explicit eligibility flags prohibit it from acting
    as an atom-resolved label; Stage 2 rematches the complete pKa set jointly.
    """
    rows: List[Dict] = []
    for record_index, row in enumerate(macro.itertuples(index=False)):
        mol, _, transitions = _resolved_transitions(str(row.smiles))
        if not transitions:
            continue
        transition = sorted(
            transitions,
            key=lambda item: (
                str(item.get("label", "")),
                tuple(sorted(item.get("atom_set", set()))),
            ),
        )[0]
        mapped = ensure_heavy_atom_maps(mol)
        atom_maps = [
            int(mapped.GetAtomWithIdx(index).GetAtomMapNum())
            for index in sorted(transition.get("atom_set", set()))
        ]
        values = json.loads(str(row.experimental_pka_values_json))
        macro_set_id = "pkahub_macro_" + hashlib.sha256(
            str(row.molecule_group_key).encode()
        ).hexdigest()[:16]
        for value_index, value in enumerate(values):
            rows.append({
                "dataset_schema_version": "pkahub_macro_placeholder_1.0.0",
                "functional_group_detector_schema_version": (
                    FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION
                ),
                "dataset_row_id": f"{macro_set_id}_{value_index}",
                "source_file": "pkahub_macro_set_candidates.tsv",
                "source_dataset": "pkahub",
                "record_index": int(record_index),
                "experimental_pka": float(value),
                "measurement_method": "experimental",
                "measurement_provenance": "pkahub_experimental_scalar_only",
                "source_assay_id": "",
                "source_document_id": "",
                "source_molecule_id": str(row.pkahub_molids_json),
                "source_original_smiles": str(row.smiles),
                "input_smiles": str(row.smiles),
                "input_atom_mapped_smiles": mapped_smiles(mapped),
                "site_label": str(transition["label"]),
                "site_family": str(transition["pair_family"]),
                "site_atom_maps_json": _json(atom_maps),
                "site_assignment_confidence": "structural_carrier_only",
                "structural_site_assignment_confidence": "high",
                "source_site_evidence": "unassigned_molecule_macro_set",
                "source_site_evidence_confidence": "none",
                "evidence_tier": "ambiguous",
                "evidence_tier_reason": "multi_transition_molecule_level_pka_set",
                "experimental_site_attribution_confidence": "none",
                "transition_identity_confidence": "unknown",
                "measurement_conditions_confidence": "low",
                "reference_plausibility_status": "not_assessed_macro_set",
                "stage1_local_supervision_eligible": False,
                "stage2_exact_site_supervision_eligible": False,
                "stage2_weak_molecule_supervision_eligible": True,
                "source_measurement_scope": "unordered_molecule_macroscopic_pka_set",
                "macro_set_id": macro_set_id,
                "macro_set_size": int(len(values)),
                "assignment_teacher_used": False,
                "pkahub_assignment_columns_ignored": True,
                "computational_pka_values_used": False,
                "computed_pka_columns_ignored": True,
                "marvin_values_used": False,
                "epik_values_used": False,
            })
    return pd.DataFrame(rows)


def prepare(
    input_path: str | Path,
    output_dir: str | Path,
    existing_stage1_path: str | Path = DEFAULT_EXISTING_STAGE1,
    existing_network_path: str | Path = DEFAULT_EXISTING_NETWORK,
) -> Dict[str, object]:
    """Prepare Stage 1 and macroscopic pKahub supplements with exclusion ledgers."""
    frame = pd.read_csv(input_path, sep="\t", low_memory=False)
    required = {
        "molid", "smiles", "pka_value", "data_source", "rawdataID",
        "assignment_method",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"pKahub input missing required columns: {missing}")
    frame["pka_value"] = pd.to_numeric(frame["pka_value"], errors="coerce")
    frame = frame[
        frame["pka_value"].between(*PKA_RANGE)
        & ~frame["data_source"].astype(str).isin(BENCHMARK_SOURCES)
    ].copy()

    assignments = pd.DataFrame([
        _molecule_assignment(row.molid, row.smiles)
        for row in frame[["molid", "smiles"]].drop_duplicates("molid").itertuples(
            index=False
        )
    ])
    frame = frame.merge(assignments, on="molid", validate="many_to_one")
    frame["molecule_group_key"] = frame["canonical_smiles"].map(molecule_group_key)
    existing = _existing_keys(
        existing_stage1_path,
        existing_network_path,
        excluded_training_cohorts=("pkahub_clean_single_transition",),
        excluded_source_prefixes=("pkahub_", "pKahub:"),
    )
    overlapping = frame["molecule_group_key"].isin(existing)
    overlap_rows = frame.loc[overlapping, [
        "molid", "canonical_smiles", "molecule_group_key", "data_source"
    ]].drop_duplicates()
    novel = frame.loc[~overlapping].copy()

    stage1_rows: List[Dict] = []
    stage1_quarantine: List[Dict] = []
    clean = novel[novel["status"] == "clean_single_transition"]
    for _, group in clean.groupby("molecule_group_key", sort=True):
        assignment = group.iloc[0]
        row, rejected = _stage1_row(group, assignment)
        if row is not None:
            stage1_rows.append(row)
        else:
            stage1_quarantine.append(rejected)

    macro_rows: List[Dict] = []
    macro_quarantine: List[Dict] = []
    multiple = novel[novel["status"] == "multiple_candidate_transitions"]
    for key, group in multiple.groupby("molecule_group_key", sort=True):
        smiles = str(group.iloc[0]["canonical_smiles"])
        transition_count = int(group.iloc[0]["candidate_transition_count"])
        unresolved = int(group.iloc[0]["unresolved_context_count"])
        clusters = _macro_clusters(group["pka_value"])
        values = [float(np.median(cluster)) for cluster in clusters]
        reason = ""
        if unresolved:
            reason = "unresolved_ionizable_context"
        elif transition_count > MAX_MACRO_TRANSITIONS:
            reason = "candidate_transition_count_exceeds_cap"
        elif not values:
            reason = "no_macroscopic_values"
        elif len(values) > transition_count:
            reason = "more_pka_clusters_than_structural_transitions"
        base = {
            "molecule_group_key": str(key),
            "smiles": smiles,
            "candidate_transition_count": transition_count,
            "experimental_pka_values_json": _json(values),
            "pka_cluster_members_json": _json(clusters),
            "measurement_count": int(len(group)),
            "pkahub_molids_json": _json(sorted(set(group["molid"].astype(str)))),
            "pkahub_data_sources_json": _json(sorted(set(
                group["data_source"].astype(str)
            ))),
            "pkahub_rawdata_ids_json": _json(sorted(set(
                group["rawdataID"].astype(str)
            ))),
            "measurement_method": "experimental",
            "measurement_scope": "unordered_molecule_macroscopic_pka_set",
            "assignment_teacher_used": False,
            "pkahub_assignment_columns_ignored": True,
        }
        if reason:
            macro_quarantine.append({**base, "reason": reason})
        else:
            macro_rows.append(base)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accepted = pd.DataFrame(stage1_rows)
    stage1_rejected = pd.DataFrame(stage1_quarantine)
    macro = pd.DataFrame(macro_rows)
    macro_rejected = pd.DataFrame(macro_quarantine)
    macro_placeholders = _macro_transition_placeholders(macro)
    accepted.to_csv(output / "stage1_supplemental_sites.csv", index=False)
    stage1_rejected.to_csv(output / "stage1_supplemental_quarantine.csv", index=False)
    macro.to_csv(output / "macro_set_candidates.csv", index=False)
    macro_placeholders.to_csv(
        output / "macro_set_transition_placeholders.csv", index=False
    )
    macro_rejected.to_csv(output / "macro_set_quarantine.csv", index=False)
    overlap_rows.to_csv(output / "existing_molecule_overlap.csv", index=False)
    summary = {
        "input_rows_after_range_and_benchmark_filters": int(len(frame)),
        "benchmark_sources_excluded": sorted(BENCHMARK_SOURCES),
        "existing_molecule_keys_excluded": int(overlap_rows["molecule_group_key"].nunique()),
        "stage1_candidate_molecule_keys": int(clean["molecule_group_key"].nunique()),
        "stage1_accepted_rows": int(len(accepted)),
        "stage1_quarantined_rows": int(len(stage1_rejected)),
        "macro_candidate_molecule_keys": int(multiple["molecule_group_key"].nunique()),
        "macro_set_accepted_rows": int(len(macro)),
        "macro_set_transition_placeholder_rows": int(len(macro_placeholders)),
        "macro_set_quarantined_rows": int(len(macro_rejected)),
        "policy": (
            "Epik-derived assignment fields ignored; benchmark sources excluded; "
            "existing molecule identities excluded; unique-transition values are "
            "reference-screened; multi-transition values remain unordered macro sets"
        ),
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--existing-stage1", default=DEFAULT_EXISTING_STAGE1)
    parser.add_argument("--existing-network", default=DEFAULT_EXISTING_NETWORK)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(prepare(
        args.input, args.output_dir, args.existing_stage1, args.existing_network
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
