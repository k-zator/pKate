#!/usr/bin/env python3
"""Build the experimental-only, atom-mapped pKa microstate dataset.

The primary CSV has one row per measured pKa transition.  Each row contains a
complete-molecule conjugate-acid and conjugate-base microstate ensemble.  Rows
without a defensible structural site are written to quarantine instead of being
assigned from the macro-pKa value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd  # type: ignore
from rdkit import Chem, rdBase  # type: ignore
from rdkit.Chem import rdFMCS  # type: ignore

from functional_group_pka_analysis import (
    ACIDIC_FAMILIES,
    SERIAL_AZOLE_PAIR_FAMILIES,
    SITE_OVERRIDE_KEY_COLUMNS,
    _site_override_key,
    classify_pair_type,
    expand_pair_site_transitions,
    normalize_conjugate_family,
    select_serial_azole_transition_by_reference,
)
from microstate_enumerator import ensure_heavy_atom_maps, generate_conjugate_microstates, mapped_smiles
from pka_evidence_policy import (
    EVIDENCE_POLICY_SCHEMA_VERSION,
    assess_measurement_evidence,
    unresolved_ionizable_contexts,
)
from substructure_match import (
    FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION,
    find_sites_with_metadata,
    resolve_overlapping_sites,
)


DEFAULT_RAW_DIR = "data/raw"
DEFAULT_ASSIGNMENTS = "data/processed/functional_group_assignments.csv"
DEFAULT_OVERRIDE_PARTS = (
    "data/processed/PART1_site_resolution_overrides.csv",
    "data/processed/PART2_site_resolution_overrides.csv",
)
DEFAULT_OUTPUT = "data/processed/pka_microstate_dataset.csv"
DEFAULT_QUARANTINE = "data/processed/pka_microstate_dataset_quarantine.csv"
DEFAULT_CLEAN_OVERRIDES = "data/processed/site_resolution_overrides.csv"
DEFAULT_OVERRIDE_QUARANTINE = "data/processed/site_resolution_overrides_quarantine.csv"
DEFAULT_MEASUREMENT_EXCLUSIONS = "data/curation/pka_measurement_exclusions.csv"
DEFAULT_OUTLIER_REVIEW_DECISIONS = "data/curation/stage1_outlier_review_decisions.csv"
DEFAULT_STRUCTURE_REVIEW_DECISIONS = (
    "data/curation/pka_structure_consistency_review_decisions.csv"
)

STRUCTURE_REVIEW_DECISIONS = {
    "keep_current_structure_and_transition",
    "exclude_measurement_structure_pair",
    "exclude_entire_source_record",
    "replace_input_smiles",
    "correct_transition_definition",
    "defer",
}

OUTLIER_REVIEW_EXCLUSION_DECISIONS = {
    "exclude_wrong_site_unknown",
    "exclude_wrong_endpoint",
    "exclude_implausible_pka",
    "unsupported_site",
    # Deferred measurements remain reviewable but must never silently re-enter
    # a release while their chemistry is unresolved.
    "defer",
}
OUTLIER_REVIEW_DECISIONS = OUTLIER_REVIEW_EXCLUSION_DECISIONS | {
    "confirm_assignment",
    "reassign_site",
}

# A molecule-level scalar above this limit cannot safely be interpreted as a
# conjugate-acid pKa for an automatically selected amine-family site without
# atom/site evidence.  Such values can instead describe a different acidic
# site (for example an indole or amide N-H).  This is deliberately a broad
# quarantine guard, not a claim that larger pKas can never exist.
UNANNOTATED_AMINE_PKA_MAX = 14.0

# Detector 2.0 first made these conjugate families available. A structurally
# unique match can propose a site, but cannot prove that an unannotated,
# molecule-level experimental scalar belongs to it.
DETECTOR_EXPANSION_REVIEW_REQUIRED_FAMILIES = {
    "alcohol_oxonium",
    "thiol_thiolium",
    "hydrazine_like",
    "hydroxylamine_like",
    "phosphate_oxyacid",
    "phosphonate_oxyacid",
    "phosphinate_oxyacid",
    "phosphoramidate_oxyacid",
    "alcohol_alkoxide",
}

def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _mol_prop(mol: Chem.Mol, *names: str) -> str:
    for name in names:
        if mol.HasProp(name):
            value = mol.GetProp(name).strip()
            if value:
                return value
    return ""


def _source_metadata(mol: Chem.Mol) -> Dict[str, str]:
    return {
        "source_assay_id": _mol_prop(mol, "assay_web_id", "assay_chembl_id"),
        "source_document_id": _mol_prop(mol, "document_web_id", "document_chembl_id"),
        "source_molecule_id": _mol_prop(mol, "chembl_id", "molecule_chembl_id"),
        "source_temperature": _mol_prop(mol, "temp", "temperature"),
        "source_original_smiles": _mol_prop(mol, "SMILES_original", "original_SMILES"),
    }


def _read_measurement_exclusions(path: Optional[str]) -> pd.DataFrame:
    columns = [
        "scope", "source_file", "assay_id", "record_index", "pka_value",
        "quarantine_reason", "quarantine_detail", "evidence_url",
    ]
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    exclusions = pd.read_csv(path, dtype={"record_index": "Int64"})
    missing = set(columns) - set(exclusions.columns)
    if missing:
        raise ValueError(f"Measurement exclusion table missing columns: {sorted(missing)}")
    invalid = set(exclusions["scope"].astype(str)) - {"assay", "record"}
    if invalid:
        raise ValueError(f"Unsupported measurement exclusion scopes: {sorted(invalid)}")
    return exclusions


def _read_outlier_review_decisions(path: Optional[str]) -> pd.DataFrame:
    required = set(SITE_OVERRIDE_KEY_COLUMNS + ["review_decision"])
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=sorted(required))
    decisions = pd.read_csv(path, low_memory=False)
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"Outlier review decision table missing columns: {sorted(missing)}")
    decisions["review_decision"] = decisions["review_decision"].fillna("").astype(str).str.strip()
    decisions = decisions[decisions["review_decision"] != ""].copy()
    invalid = set(decisions["review_decision"]) - OUTLIER_REVIEW_DECISIONS
    if invalid:
        raise ValueError(f"Unsupported outlier review decisions: {sorted(invalid)}")
    return decisions


def _prepare_outlier_review_lookup(decisions: pd.DataFrame) -> Dict[Tuple[object, ...], pd.Series]:
    lookup: Dict[Tuple[object, ...], pd.Series] = {}
    if decisions.empty:
        return lookup
    ordered = decisions.copy()
    if "reviewed_at" in ordered.columns:
        ordered = ordered.sort_values("reviewed_at")
    for _, decision in ordered.iterrows():
        lookup[_site_override_key(decision)] = decision
    return lookup


def _read_structure_review_decisions(path: Optional[str]) -> pd.DataFrame:
    columns = [
        "source_file", "record_index", "pka_value", "original_smiles",
        "review_decision", "replacement_smiles", "replacement_transition_mode",
        "review_note", "reviewed_at",
    ]
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=columns)
    decisions = pd.read_csv(path, low_memory=False)
    missing = {"source_file", "record_index", "pka_value", "review_decision"} - set(
        decisions.columns
    )
    if missing:
        raise ValueError(f"Structure review decision table missing columns: {sorted(missing)}")
    for column in columns:
        if column not in decisions.columns:
            decisions[column] = ""
    decisions["review_decision"] = decisions["review_decision"].fillna("").astype(str).str.strip()
    decisions = decisions[decisions["review_decision"] != ""].copy()
    invalid = set(decisions["review_decision"]) - STRUCTURE_REVIEW_DECISIONS
    if invalid:
        raise ValueError(f"Unsupported structure review decisions: {sorted(invalid)}")
    return decisions[columns].copy()


def _prepare_structure_review_lookups(
    decisions: pd.DataFrame,
) -> Tuple[Dict[Tuple[str, int, float], pd.Series], Dict[Tuple[str, int], pd.Series]]:
    measurement: Dict[Tuple[str, int, float], pd.Series] = {}
    record: Dict[Tuple[str, int], pd.Series] = {}
    if decisions.empty:
        return measurement, record
    ordered = decisions.sort_values("reviewed_at") if "reviewed_at" in decisions else decisions
    for _, decision in ordered.iterrows():
        source_key = (str(decision["source_file"]), int(decision["record_index"]))
        if str(decision["review_decision"]) == "exclude_entire_source_record":
            record[source_key] = decision
        else:
            measurement[(source_key[0], source_key[1], round(float(decision["pka_value"]), 6))] = decision
    return measurement, record


def _validated_replacement_mol(original: Chem.Mol, replacement_smiles: str) -> Chem.Mol:
    replacement = Chem.MolFromSmiles(str(replacement_smiles))
    if replacement is None:
        raise ValueError("replacement SMILES is not parseable")
    original_heavy = Chem.RemoveHs(original)
    replacement_heavy = Chem.RemoveHs(replacement)
    original_elements = sorted(atom.GetSymbol() for atom in original_heavy.GetAtoms())
    replacement_elements = sorted(atom.GetSymbol() for atom in replacement_heavy.GetAtoms())
    if original_elements != replacement_elements:
        raise ValueError("replacement changes the heavy-atom element inventory")
    mcs = rdFMCS.FindMCS(
        [original_heavy, replacement_heavy],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=5,
    )
    if bool(mcs.canceled) or int(mcs.numAtoms) != int(original_heavy.GetNumAtoms()):
        raise ValueError("replacement changes heavy-atom connectivity")
    return replacement


def _reviewed_site_atom_indices(review: pd.Series) -> set[int]:
    text = _text(review.get("reviewed_site_atom_indices_json")).strip()
    if not text:
        return set()
    try:
        return {int(value) for value in json.loads(text)}
    except (TypeError, ValueError, json.JSONDecodeError):
        return set()


def _select_reviewed_site(review: pd.Series, pair_sites: List[Dict]) -> Optional[Dict]:
    label = _text(review.get("reviewed_site_label")).strip()
    matching = [
        site for site in pair_sites
        if str(site.get("label", "")) == label
        or str(site.get("detector_label", "")) == label
    ]
    preferred_type = _text(review.get("pka_type_canonical")).strip().lower()
    if preferred_type in {"acidic", "basic"}:
        typed = [
            site for site in matching
            if ("acidic" if site.get("pair_family") in ACIDIC_FAMILIES else "basic") == preferred_type
        ]
        if typed:
            matching = typed
    # Reviewing a detector label or its atoms does not identify which edge of
    # a serial acid/base coordinate was measured.  Historical decisions must
    # provide a pKa type, an explicit transition alias, or explicit transition
    # verification before they can supervise one of these edges.
    detector_only = bool(matching) and all(
        str(site.get("label", "")) != label
        and str(site.get("detector_label", "")) == label
        for site in matching
    )
    if (
        detector_only
        and preferred_type not in {"acidic", "basic"}
        and not _truthy(review.get("review_verified_transition_identity", False))
        and all(
            site.get("pair_family") in SERIAL_AZOLE_PAIR_FAMILIES
            for site in matching
        )
    ):
        return None
    reviewed_atoms = _reviewed_site_atom_indices(review)
    if reviewed_atoms:
        exact = [site for site in matching if set(site.get("atom_set", set())) == reviewed_atoms]
        return exact[0] if len(exact) == 1 else None
    return matching[0] if len(matching) == 1 else None


def _review_explicitly_selects_transition(review: Optional[pd.Series], site: Dict) -> bool:
    """Return whether a review names one transition edge, not only its detector."""
    if review is None:
        return False
    reviewed_label = _text(review.get("reviewed_site_label")).strip()
    return bool(
        site.get("pair_family") in SERIAL_AZOLE_PAIR_FAMILIES
        and reviewed_label
        and reviewed_label == str(site.get("label", ""))
        and reviewed_label != str(site.get("detector_label", ""))
    )


def _measurement_exclusion(
    exclusions: pd.DataFrame,
    source_file: str,
    record_index: int,
    pka_value: float,
    source_assay_id: str,
) -> Optional[pd.Series]:
    if exclusions.empty:
        return None
    same_source = exclusions["source_file"].fillna("").astype(str).isin({"", source_file})
    assay_match = (
        exclusions["scope"].eq("assay")
        & exclusions["assay_id"].fillna("").astype(str).eq(source_assay_id)
    )
    record_match = (
        exclusions["scope"].eq("record")
        & exclusions["record_index"].eq(record_index)
        & (
            exclusions["pka_value"].isna()
            | exclusions["pka_value"].astype(float).sub(float(pka_value)).abs().lt(1e-8)
        )
    )
    matching = exclusions[same_source & (assay_match | record_match)]
    return None if matching.empty else matching.iloc[0]


def _source_site_evidence(row: pd.Series, site: Dict) -> Tuple[str, str]:
    """Describe evidence in the experimental source, independently of structure matching."""
    if not pd.isna(row.get("atom_index_raw")) and _index_matches_site(row.get("atom_index_raw"), site):
        return "source_atom_index_matches_selected_site", "high"
    if _text(row.get("site_identifier_raw")).strip():
        return "source_site_identifier_without_verified_atom_mapping", "medium"
    if _text(row.get("pka_type_raw")).strip():
        return "source_ionization_type_only_no_site", "low"
    return "unannotated_molecule_level_pka", "none"


def _stable_id(row: pd.Series, site_maps: Sequence[int]) -> str:
    key = list(_site_override_key(row)) + [tuple(site_maps)]
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:20]


def _read_override_parts(paths: Sequence[str]) -> pd.DataFrame:
    parts = []
    for path in paths:
        if not os.path.exists(path):
            continue
        part = pd.read_csv(path, dtype={"override_group_label": "string"})
        part["override_part"] = os.path.basename(path)
        part["override_part_row"] = range(len(part))
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=SITE_OVERRIDE_KEY_COLUMNS + ["override_group_label"])
    return pd.concat(parts, ignore_index=True, sort=False)


def _prepare_override_lookup(
    overrides: pd.DataFrame,
    assignments: pd.DataFrame,
) -> Tuple[Dict[Tuple[object, ...], pd.Series], List[Dict]]:
    assignment_keys = {_site_override_key(row) for _, row in assignments.iterrows()}
    grouped: Dict[Tuple[object, ...], List[pd.Series]] = defaultdict(list)
    quarantine: List[Dict] = []

    for _, row in overrides.iterrows():
        base = row.to_dict()
        label = _text(row.get("override_group_label")).strip()
        if _text(row.get("pka_source_method")).lower() != "experimental":
            quarantine.append({**base, "quarantine_reason": "non_experimental_override_excluded"})
            continue
        if not label:
            quarantine.append({**base, "quarantine_reason": "missing_override_label"})
            continue
        if label.isdigit():
            quarantine.append({**base, "quarantine_reason": "invalid_numeric_override_label"})
            continue
        key = _site_override_key(row)
        if key not in assignment_keys:
            quarantine.append({**base, "quarantine_reason": "stale_override_key"})
            continue
        grouped[key].append(row)

    lookup: Dict[Tuple[object, ...], pd.Series] = {}
    for key, rows in grouped.items():
        labels = {_text(row.get("override_group_label")).strip() for row in rows}
        if len(labels) > 1:
            for row in rows:
                quarantine.append({**row.to_dict(), "quarantine_reason": "conflicting_duplicate_overrides"})
            continue
        chosen = sorted(rows, key=lambda row: _text(row.get("reviewed_at")))[-1]
        lookup[key] = chosen
        for duplicate in rows[:-1]:
            quarantine.append({**duplicate.to_dict(), "quarantine_reason": "superseded_duplicate_override"})
    return lookup, quarantine


def _load_requested_molecules(sdf_path: str, requested: set[int]) -> Dict[int, Chem.Mol]:
    loaded: Dict[int, Chem.Mol] = {}
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    for record_idx, mol in enumerate(supplier):
        if record_idx in requested and mol is not None:
            loaded[record_idx] = mol
        if len(loaded) == len(requested):
            break
    return loaded


def _resolved_pair_sites(mol: Chem.Mol, overlap_threshold: float) -> Tuple[List[Dict], List[Dict]]:
    candidates = find_sites_with_metadata(mol)
    resolved, _ = resolve_overlapping_sites(candidates, overlap_threshold=overlap_threshold)
    pair_sites = []
    for site in resolved:
        pair_sites.extend(expand_pair_site_transitions(site))
    return resolved, pair_sites


def _index_matches_site(raw_index: object, site: Dict) -> bool:
    if pd.isna(raw_index):
        return False
    try:
        value = int(float(raw_index))
    except (TypeError, ValueError):
        return False
    atom_set = set(site.get("atom_set", set()))
    return value in atom_set or (value - 1) in atom_set


def _select_site(
    row: pd.Series,
    pair_sites: List[Dict],
    override: Optional[pd.Series],
) -> Tuple[Optional[Dict], str, str]:
    preferred_type = _text(row.get("pka_type_canonical")).strip().lower()
    if preferred_type in {"acidic", "basic"}:
        typed = [
            site for site in pair_sites
            if ("acidic" if site.get("pair_family") in ACIDIC_FAMILIES else "basic") == preferred_type
        ]
        if typed:
            pair_sites = typed
    if override is None:
        if len(pair_sites) == 1:
            only = pair_sites[0]
            if (
                preferred_type not in {"acidic", "basic"}
                and only.get("pair_family") in SERIAL_AZOLE_PAIR_FAMILIES
            ):
                selected, reason = select_serial_azole_transition_by_reference(
                    pair_sites, row.get("pka_value")
                )
                if selected is None:
                    return None, reason, "none"
                return selected, reason, "medium"
            return pair_sites[0], "structurally_unique_pair_site", "high"
        if not pair_sites:
            return None, "no_supported_conjugate_pair_site", "none"
        physical_groups = {
            (
                str(site.get("coupling_group", site.get("pair_family", ""))),
                tuple(sorted(site.get("atom_set", set()))),
            )
            for site in pair_sites
        }
        if (
            preferred_type not in {"acidic", "basic"}
            and len(physical_groups) == 1
            and all(
                site.get("pair_family") in SERIAL_AZOLE_PAIR_FAMILIES
                for site in pair_sites
            )
        ):
            selected, reason = select_serial_azole_transition_by_reference(
                pair_sites, row.get("pka_value")
            )
            if selected is not None:
                return selected, reason, "medium"
            return None, reason, "none"
        return None, "multiple_pair_sites_without_override", "none"

    label = _text(override.get("override_group_label")).strip()
    matching = [
        site for site in pair_sites
        if site.get("label") == label or site.get("detector_label") == label
    ]
    reviewer_atom_index = override.get("override_atom_index_raw")
    if pd.isna(reviewer_atom_index):
        reviewer_atom_index = override.get("atom_index_raw")
    if len(matching) > 1 and not pd.isna(reviewer_atom_index):
        atom_matching = [
            site for site in matching
            if _index_matches_site(reviewer_atom_index, site)
        ]
        if len(atom_matching) == 1:
            matching = atom_matching
    detector_only_override = bool(matching) and all(
        site.get("label") != label and site.get("detector_label") == label
        for site in matching
    )
    if (
        detector_only_override
        and preferred_type not in {"acidic", "basic"}
        and all(
            site.get("pair_family") in SERIAL_AZOLE_PAIR_FAMILIES
            for site in matching
        )
    ):
        selected, reason = select_serial_azole_transition_by_reference(
            matching, row.get("pka_value")
        )
        if selected is not None:
            return selected, f"legacy_group_override+{reason}", "medium"
        return None, f"legacy_group_override+{reason}", "none"
    if len(matching) == 1:
        return matching[0], "human_override_revalidated_on_raw_structure", "high"
    if not matching:
        return None, "override_label_not_found_on_current_structure", "none"
    return None, "override_ambiguous_between_same_label_sites", "none"


def _fractions(pka: float, ph: float) -> Tuple[float, float]:
    exponent = max(-300.0, min(300.0, ph - pka))
    acid = 1.0 / (1.0 + 10.0 ** exponent)
    return acid, 1.0 - acid


def _base_quarantine_row(
    row: pd.Series,
    reason: str,
    detail: str = "",
    resolved_sites: Optional[Sequence[Dict]] = None,
    pair_sites: Optional[Sequence[Dict]] = None,
    override: Optional[pd.Series] = None,
    source_metadata: Optional[Dict[str, str]] = None,
    outlier_review: Optional[pd.Series] = None,
    structure_review: Optional[pd.Series] = None,
) -> Dict:
    resolved_sites = list(resolved_sites or [])
    pair_sites = list(pair_sites or [])
    source_metadata = source_metadata or {}
    return {
        "functional_group_detector_schema_version": FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION,
        "source_file": row.get("source_file"),
        "record_index": row.get("record_index"),
        "smiles": row.get("smiles"),
        "pka_value": row.get("pka_value"),
        "pka_source_method": row.get("pka_source_method"),
        "pka_type_reported": row.get("pka_type_raw"),
        "source_assay_id": source_metadata.get("source_assay_id", ""),
        "source_document_id": source_metadata.get("source_document_id", ""),
        "source_molecule_id": source_metadata.get("source_molecule_id", ""),
        "source_temperature": source_metadata.get("source_temperature", ""),
        "source_original_smiles": source_metadata.get("source_original_smiles", ""),
        "outlier_review_decision": "" if outlier_review is None else _text(outlier_review.get("review_decision")),
        "outlier_review_note": "" if outlier_review is None else _text(outlier_review.get("review_note")),
        "outlier_reviewed_at": "" if outlier_review is None else _text(outlier_review.get("reviewed_at")),
        "structure_review_decision": "" if structure_review is None else _text(
            structure_review.get("review_decision")
        ),
        "structure_review_replacement_smiles": "" if structure_review is None else _text(
            structure_review.get("replacement_smiles")
        ),
        "structure_review_note": "" if structure_review is None else _text(
            structure_review.get("review_note")
        ),
        "structure_reviewed_at": "" if structure_review is None else _text(
            structure_review.get("reviewed_at")
        ),
        "override_group_label": "" if override is None else _text(override.get("override_group_label")),
        "override_atom_index_raw": (
            "" if override is None else override.get("override_atom_index_raw", "")
        ),
        "resolved_site_count": len(resolved_sites),
        "resolved_site_labels": "|".join(str(site.get("label", "")) for site in resolved_sites),
        "resolved_pair_site_count": len(pair_sites),
        "resolved_pair_site_labels": "|".join(str(site.get("label", "")) for site in pair_sites),
        "quarantine_reason": reason,
        "quarantine_detail": detail,
    }


def build_dataset(
    raw_dir: str,
    assignments_path: str,
    override_paths: Sequence[str],
    output_path: str,
    quarantine_path: str,
    clean_overrides_path: str,
    override_quarantine_path: str,
    ph: float = 7.4,
    overlap_threshold: float = 0.5,
    max_tautomers: int = 16,
    measurement_exclusions_path: Optional[str] = DEFAULT_MEASUREMENT_EXCLUSIONS,
    outlier_review_decisions_path: Optional[str] = DEFAULT_OUTLIER_REVIEW_DECISIONS,
    structure_review_decisions_path: Optional[str] = DEFAULT_STRUCTURE_REVIEW_DECISIONS,
    allow_unreviewed_detector_expansion: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_csv(assignments_path, low_memory=False)
    assignments = assignments[assignments["pka_source_method"].astype(str).str.lower() == "experimental"].copy()
    assignments.sort_values(["source_file", "record_index", "pka_value"], inplace=True)
    assignments.reset_index(drop=True, inplace=True)

    overrides = _read_override_parts(override_paths)
    override_lookup, override_quarantine_rows = _prepare_override_lookup(overrides, assignments)
    measurement_exclusions = _read_measurement_exclusions(measurement_exclusions_path)
    outlier_review_lookup = _prepare_outlier_review_lookup(
        _read_outlier_review_decisions(outlier_review_decisions_path)
    )
    structure_review_measurements, structure_review_records = _prepare_structure_review_lookups(
        _read_structure_review_decisions(structure_review_decisions_path)
    )

    dataset_rows: List[Dict] = []
    quarantine_rows: List[Dict] = []
    clean_override_rows: List[Dict] = []
    validated_override_keys = set()
    microstate_cache = {}

    for source_file, source_rows in assignments.groupby("source_file", sort=True):
        sdf_path = os.path.join(raw_dir, source_file)
        if not os.path.exists(sdf_path):
            for _, row in source_rows.iterrows():
                quarantine_rows.append(_base_quarantine_row(row, "raw_source_file_missing", sdf_path))
            continue

        requested = {int(value) for value in source_rows["record_index"]}
        molecules = _load_requested_molecules(sdf_path, requested)
        for _, row in source_rows.iterrows():
            record_index = int(row["record_index"])
            mol = molecules.get(record_index)
            if mol is None:
                quarantine_rows.append(_base_quarantine_row(row, "raw_sdf_record_unreadable"))
                continue

            source_metadata = _source_metadata(mol)
            source_structure_smiles = Chem.MolToSmiles(
                mol, canonical=True, isomericSmiles=True
            )
            structure_review = structure_review_records.get((source_file, record_index))
            if structure_review is None:
                structure_review = structure_review_measurements.get((
                    source_file, record_index, round(float(row["pka_value"]), 6)
                ))
            structure_review_decision = (
                "" if structure_review is None
                else _text(structure_review.get("review_decision")).strip()
            )
            if structure_review_decision in {
                "exclude_measurement_structure_pair", "exclude_entire_source_record", "defer"
            }:
                quarantine_rows.append(_base_quarantine_row(
                    row,
                    f"manual_pka_structure_review:{structure_review_decision}",
                    "" if structure_review is None else _text(structure_review.get("review_note")),
                    source_metadata=source_metadata,
                    structure_review=structure_review,
                ))
                continue
            if structure_review_decision == "correct_transition_definition":
                mode = _text(structure_review.get("replacement_transition_mode"))
                quarantine_rows.append(_base_quarantine_row(
                    row,
                    "manual_pka_structure_review:transition_definition_requires_rebuild_support",
                    f"requested_transition_mode={mode}; {_text(structure_review.get('review_note'))}",
                    source_metadata=source_metadata,
                    structure_review=structure_review,
                ))
                continue
            if structure_review_decision == "replace_input_smiles":
                recorded_original = _text(structure_review.get("original_smiles")).strip()
                if recorded_original:
                    recorded_mol = Chem.MolFromSmiles(recorded_original)
                    recorded_canonical = (
                        Chem.MolToSmiles(recorded_mol, canonical=True, isomericSmiles=True)
                        if recorded_mol is not None else ""
                    )
                    if recorded_canonical != source_structure_smiles:
                        quarantine_rows.append(_base_quarantine_row(
                            row,
                            "manual_pka_structure_review:stale_original_structure",
                            f"reviewed={recorded_canonical}; current={source_structure_smiles}",
                            source_metadata=source_metadata,
                            structure_review=structure_review,
                        ))
                        continue
                try:
                    mol = _validated_replacement_mol(
                        mol, _text(structure_review.get("replacement_smiles"))
                    )
                except ValueError as exc:
                    quarantine_rows.append(_base_quarantine_row(
                        row,
                        "manual_pka_structure_review:invalid_replacement_smiles",
                        str(exc),
                        source_metadata=source_metadata,
                        structure_review=structure_review,
                    ))
                    continue
            review = outlier_review_lookup.get(_site_override_key(row))
            review_decision = "" if review is None else _text(review.get("review_decision")).strip()
            if review_decision in OUTLIER_REVIEW_EXCLUSION_DECISIONS:
                quarantine_rows.append(_base_quarantine_row(
                    row,
                    f"manual_outlier_review:{review_decision}",
                    _text(review.get("review_note")),
                    override=override_lookup.get(_site_override_key(row)),
                    source_metadata=source_metadata,
                    outlier_review=review,
                ))
                continue

            exclusion = _measurement_exclusion(
                measurement_exclusions,
                source_file=source_file,
                record_index=record_index,
                pka_value=float(row["pka_value"]),
                source_assay_id=source_metadata["source_assay_id"],
            )
            if exclusion is not None:
                detail = _text(exclusion.get("quarantine_detail"))
                evidence_url = _text(exclusion.get("evidence_url"))
                if evidence_url:
                    detail = f"{detail}; evidence={evidence_url}" if detail else f"evidence={evidence_url}"
                quarantine_rows.append(_base_quarantine_row(
                    row,
                    _text(exclusion.get("quarantine_reason")) or "curated_source_exclusion",
                    detail,
                    override=override_lookup.get(_site_override_key(row)),
                    source_metadata=source_metadata,
                ))
                continue

            resolved, pair_sites = _resolved_pair_sites(mol, overlap_threshold)
            unresolved_contexts = unresolved_ionizable_contexts(resolved, pair_sites)
            override = override_lookup.get(_site_override_key(row))
            if review_decision in {"confirm_assignment", "reassign_site"} and review is not None:
                site = _select_reviewed_site(review, pair_sites)
                assignment_basis = (
                    "manual_outlier_review_confirmed_exact_site"
                    if review_decision == "confirm_assignment"
                    else "manual_outlier_review_exact_site"
                )
                confidence = "high" if site is not None else "none"
            else:
                site, assignment_basis, confidence = _select_site(row, pair_sites, override)
            if site is None:
                if review_decision in {"confirm_assignment", "reassign_site"}:
                    assignment_basis = "manual_outlier_review_site_not_supported_on_current_structure"
                quarantine_rows.append(_base_quarantine_row(
                    row,
                    assignment_basis,
                    resolved_sites=resolved,
                    pair_sites=pair_sites,
                    override=override,
                    source_metadata=source_metadata,
                    outlier_review=review,
                ))
                if override is not None:
                    override_quarantine_rows.append({
                        **override.to_dict(),
                        "quarantine_reason": assignment_basis,
                    })
                continue


            source_evidence, source_evidence_confidence = _source_site_evidence(row, site)
            if review is not None and _truthy(review.get("review_verified_site_attribution", False)):
                source_evidence = "reviewer_verified_source_site_attribution"
                source_evidence_confidence = "high"
            if (
                site.get("pair_family") in DETECTOR_EXPANSION_REVIEW_REQUIRED_FAMILIES
                and review_decision not in {"confirm_assignment", "reassign_site"}
                and override is None
                and source_evidence_confidence != "high"
                and not allow_unreviewed_detector_expansion
            ):
                quarantine_rows.append(_base_quarantine_row(
                    row,
                    "detector_v2_new_family_requires_manual_review",
                    (
                        f"selected_site={site.get('label')}; family={site.get('pair_family')}; "
                        "structurally unique site lacks atom-level source evidence or a terminal manual review"
                    ),
                    resolved_sites=resolved,
                    pair_sites=pair_sites,
                    override=override,
                    source_metadata=source_metadata,
                    outlier_review=review,
                ))
                continue
            if (
                site.get("pair_family") == "amine"
                and float(row["pka_value"]) > UNANNOTATED_AMINE_PKA_MAX
                and source_evidence_confidence == "none"
            ):
                quarantine_rows.append(_base_quarantine_row(
                    row,
                    "unannotated_amine_pka_above_plausibility_guard",
                    (
                        f"selected_site={site.get('label')}; pKa={float(row['pka_value']):.6g}; "
                        f"guard={UNANNOTATED_AMINE_PKA_MAX:.6g}; molecule-level source has no atom/site annotation"
                    ),
                    resolved_sites=resolved,
                    pair_sites=pair_sites,
                    override=override,
                    source_metadata=source_metadata,
                ))
                continue

            # Atom selection and protonation-center ambiguity are different
            # concepts.  Preserve the structural mapping confidence before a
            # multi-center microstate enumeration can lower the legacy field.
            structural_mapping_confidence = confidence
            cache_key = (
                Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
                site["label"],
                tuple(sorted(site["atom_set"])),
                max_tautomers,
            )
            result = microstate_cache.get(cache_key)
            if result is None:
                result = generate_conjugate_microstates(
                    mol,
                    site,
                    family=site["pair_family"],
                    member_form=site["pair_member_form"],
                    max_tautomers=max_tautomers,
                )
                microstate_cache[cache_key] = result
            if result.status != "ok":
                quarantine_rows.append(_base_quarantine_row(
                    row,
                    "microstate_generation_failed",
                    result.note,
                    resolved_sites=resolved,
                    pair_sites=pair_sites,
                    override=override,
                    source_metadata=source_metadata,
                ))
                if override is not None:
                    override_quarantine_rows.append({
                        **override.to_dict(),
                        "quarantine_reason": "microstate_generation_failed",
                        "quarantine_detail": result.note,
                    })
                continue

            if len(result.protonation_center_maps) > 1:
                confidence = "medium"
                assignment_basis += "+multiple_centers_enumerated"

            input_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            evidence = assess_measurement_evidence(
                pka=float(row["pka_value"]),
                label=site["label"],
                family=site["pair_family"],
                structural_mapping_confidence=structural_mapping_confidence,
                source_site_evidence_confidence=source_evidence_confidence,
                unresolved_contexts=unresolved_contexts,
                source_temperature=source_metadata.get("source_temperature", ""),
                source_original_smiles=source_metadata.get("source_original_smiles", ""),
                input_smiles=input_smiles,
                reviewer_verified_site_attribution=(
                    review is not None
                    and _truthy(review.get("review_verified_site_attribution", False))
                ),
                reviewer_verified_transition_identity=(
                    review is not None
                    and _truthy(review.get("review_verified_transition_identity", False))
                ) or _review_explicitly_selects_transition(review, site),
                reviewer_verified_measurement_conditions=(
                    review is not None
                    and _truthy(review.get("review_verified_measurement_conditions", False))
                ),
            )

            acid_fraction, base_fraction = _fractions(float(row["pka_value"]), ph)
            dominant_form = "acid_form" if acid_fraction >= base_fraction else "base_form"
            mapped_input = ensure_heavy_atom_maps(mol)
            all_pair_sites = [
                {
                    "label": candidate["label"],
                    "detector_label": candidate.get("detector_label", candidate["label"]),
                    "family": candidate["pair_family"],
                    "member_form": candidate["pair_member_form"],
                    "coupling_group": candidate.get("coupling_group", candidate["pair_family"]),
                    "acid_level": candidate.get("acid_level", 1),
                    "base_level": candidate.get("base_level", 0),
                    "atom_maps": [
                        mapped_input.GetAtomWithIdx(idx).GetAtomMapNum()
                        for idx in sorted(candidate["atom_set"])
                    ],
                }
                for candidate in pair_sites
            ]
            fragments = len(Chem.GetMolFrags(mol))
            dataset_rows.append({
                "dataset_schema_version": "1.8.0",
                "functional_group_detector_schema_version": FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION,
                "evidence_policy_schema_version": EVIDENCE_POLICY_SCHEMA_VERSION,
                "rdkit_version": rdBase.rdkitVersion,
                "dataset_row_id": _stable_id(row, result.site_atom_maps),
                "source_file": source_file,
                "record_index": record_index,
                "source_exact_duplicate_record_indices": _text(row.get("source_exact_duplicate_record_indices")),
                "experimental_pka": float(row["pka_value"]),
                "pka_type_reported": _text(row.get("pka_type_raw")),
                "measurement_method": "experimental",
                "measurement_provenance": "raw_sdf:pKa",
                **source_metadata,
                "raw_structure_smiles": source_structure_smiles,
                "standardized_structure_smiles": input_smiles,
                "structure_provenance": (
                    "manual_pka_structure_review_replacement_overlay"
                    if structure_review_decision == "replace_input_smiles"
                    else "source_contains_original_smiles_and_transformed_drawing"
                    if source_metadata.get("source_original_smiles")
                    else "curated_fixed_sdf"
                    if source_file.startswith("FIXED_")
                    else "source_sdf_as_supplied"
                ),
                "input_smiles": input_smiles,
                "input_atom_mapped_smiles": mapped_smiles(mapped_input),
                "input_formal_charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
                "fragment_count": fragments,
                "is_multicomponent": fragments > 1,
                "site_label": site["label"],
                "site_detector_label": site.get("detector_label", site["label"]),
                "site_family": site["pair_family"],
                "input_site_member_form": site["pair_member_form"],
                "site_coupling_group": site.get("coupling_group", site["pair_family"]),
                "site_acid_level": site.get("acid_level", 1),
                "site_base_level": site.get("base_level", 0),
                "site_atom_maps_json": _json(result.site_atom_maps),
                "protonation_center_atom_maps_json": _json(result.protonation_center_maps),
                "all_resolved_pair_sites_json": _json(all_pair_sites),
                "all_resolved_detector_labels_json": _json(sorted({
                    str(candidate.get("label", "")) for candidate in resolved
                })),
                "unresolved_ionizable_contexts_json": _json(unresolved_contexts),
                "unresolved_ionizable_context_count": len(unresolved_contexts),
                "other_pair_sites_held_as_drawn": max(0, len(pair_sites) - 1),
                "site_assignment_basis": assignment_basis,
                "site_assignment_confidence": confidence,
                "structural_site_assignment_confidence": structural_mapping_confidence,
                "microstate_center_confidence": confidence,
                "source_site_evidence": source_evidence,
                "source_site_evidence_confidence": source_evidence_confidence,
                **{
                    key: _json(value)
                    if key == "unresolved_ionizable_context_labels"
                    else value
                    for key, value in evidence.items()
                },
                "outlier_review_decision": review_decision,
                "outlier_review_note": "" if review is None else _text(review.get("review_note")),
                "outlier_reviewed_at": "" if review is None else _text(review.get("reviewed_at")),
                "structure_review_decision": structure_review_decision,
                "structure_review_replacement_smiles": (
                    "" if structure_review is None else _text(
                        structure_review.get("replacement_smiles")
                    )
                ),
                "structure_review_note": "" if structure_review is None else _text(
                    structure_review.get("review_note")
                ),
                "structure_reviewed_at": "" if structure_review is None else _text(
                    structure_review.get("reviewed_at")
                ),
                "override_applied": override is not None,
                "override_atom_index_raw": (
                    "" if override is None else override.get("override_atom_index_raw", "")
                ),
                "acid_formal_charge": result.acid_charge,
                "base_formal_charge": result.base_charge,
                "reference_acid_atom_mapped_smiles": result.acid_microstates[0],
                "reference_base_atom_mapped_smiles": result.base_microstates[0],
                "acid_microstates_json": _json(result.acid_microstates),
                "base_microstates_json": _json(result.base_microstates),
                "acid_microstate_count": len(result.acid_microstates),
                "base_microstate_count": len(result.base_microstates),
                "acid_tautomer_enumeration_truncated": result.acid_tautomer_enumeration_truncated,
                "base_tautomer_enumeration_truncated": result.base_tautomer_enumeration_truncated,
                "microstate_generation_status": result.status,
                "microstate_generation_note": result.note,
                "distribution_ph": ph,
                "acid_fraction_at_ph": acid_fraction,
                "base_fraction_at_ph": base_fraction,
                "dominant_site_form_at_ph": dominant_form,
                "dominant_site_form_fraction_at_ph": max(acid_fraction, base_fraction),
                "distribution_assumption": "isolated Henderson-Hasselbalch transition; other sites held as drawn",
                "marvin_values_used": False,
                "epik_values_used": False,
            })

            if override is not None:
                key = _site_override_key(row)
                validated_override_keys.add(key)
                clean_override_rows.append({
                    **override.to_dict(),
                    "override_site_family": site["pair_family"],
                    "override_site_atom_maps_json": _json(result.site_atom_maps),
                    "override_protonation_center_atom_maps_json": _json(result.protonation_center_maps),
                    "validation_status": "valid_current_structure",
                })

    quarantined_override_keys = {
        _site_override_key(pd.Series(row))
        for row in override_quarantine_rows
        if all(column in row for column in SITE_OVERRIDE_KEY_COLUMNS)
    }
    for key, override in override_lookup.items():
        if key not in validated_override_keys:
            if key not in quarantined_override_keys:
                override_quarantine_rows.append({
                    **override.to_dict(),
                    "quarantine_reason": "override_not_used_by_release_dataset",
                })

    dataset = pd.DataFrame(dataset_rows)
    quarantine = pd.DataFrame(quarantine_rows)
    clean_overrides = pd.DataFrame(clean_override_rows).drop_duplicates(
        subset=SITE_OVERRIDE_KEY_COLUMNS, keep="last"
    ) if clean_override_rows else pd.DataFrame()
    override_quarantine = pd.DataFrame(override_quarantine_rows)

    for frame, path in (
        (dataset, output_path),
        (quarantine, quarantine_path),
        (clean_overrides, clean_overrides_path),
        (override_quarantine, override_quarantine_path),
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return dataset, quarantine, clean_overrides, override_quarantine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--assignments", default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--override-parts", nargs="*", default=list(DEFAULT_OVERRIDE_PARTS))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--clean-overrides", default=DEFAULT_CLEAN_OVERRIDES)
    parser.add_argument("--override-quarantine", default=DEFAULT_OVERRIDE_QUARANTINE)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--max-tautomers", type=int, default=16)
    parser.add_argument("--measurement-exclusions", default=DEFAULT_MEASUREMENT_EXCLUSIONS)
    parser.add_argument("--outlier-review-decisions", default=DEFAULT_OUTLIER_REVIEW_DECISIONS)
    parser.add_argument("--structure-review-decisions", default=DEFAULT_STRUCTURE_REVIEW_DECISIONS)
    parser.add_argument(
        "--allow-unreviewed-detector-expansion",
        action="store_true",
        help="Candidate-discovery only: bypass the new-family review gate; never use for a release",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, quarantine, clean, override_quarantine = build_dataset(
        raw_dir=args.raw_dir,
        assignments_path=args.assignments,
        override_paths=args.override_parts,
        output_path=args.output,
        quarantine_path=args.quarantine,
        clean_overrides_path=args.clean_overrides,
        override_quarantine_path=args.override_quarantine,
        ph=args.ph,
        overlap_threshold=args.overlap_threshold,
        max_tautomers=args.max_tautomers,
        measurement_exclusions_path=args.measurement_exclusions,
        outlier_review_decisions_path=args.outlier_review_decisions,
        structure_review_decisions_path=args.structure_review_decisions,
        allow_unreviewed_detector_expansion=bool(args.allow_unreviewed_detector_expansion),
    )
    if args.allow_unreviewed_detector_expansion:
        print("WARNING: detector-expansion review gate bypassed; output is candidate-discovery data, not a release")
    print(f"dataset_rows={len(dataset)}")
    print(f"quarantined_rows={len(quarantine)}")
    print(f"validated_overrides={len(clean)}")
    print(f"quarantined_overrides={len(override_quarantine)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
