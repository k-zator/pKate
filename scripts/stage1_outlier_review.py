#!/usr/bin/env python3
"""Build and interactively review every large Stage 1 scaffold-OOF error.

The queue is one row per original experimental transition, not merely one row
per aggregated training target. Decisions are saved immediately and keyed to
the raw SDF measurement, so the release builder can enforce them on every
subsequent rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote, urlparse

import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore

from build_release_microstate_dataset import _resolved_pair_sites, _select_reviewed_site
from functional_group_pka_analysis import (
    SITE_OVERRIDE_KEY_COLUMNS,
    _site_override_key,
)
from site_resolution_review import RawMolCache, _render_review_png, _try_open_image
from substructure_match import FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION, maximal_site_rank_key


DEFAULT_OOF = "data/processed/ml_models_experimental_only/stage1_intrinsic/scaffold_oof_predictions.csv"
DEFAULT_TRANSITIONS = "data/processed/pka_microstate_dataset.csv"
DEFAULT_ASSIGNMENTS = "data/processed/functional_group_assignments.csv"
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_DECISIONS = "data/curation/stage1_outlier_review_decisions.csv"
DEFAULT_DECISION_HISTORY = "data/curation/stage1_outlier_review_decision_history.csv"
DEFAULT_QUEUE = "data/processed/dataset_audit/stage1_outlier_review_queue.csv"
DEFAULT_REOPEN_REPORT = "data/processed/dataset_audit/stage1_detector_change_rereview.csv"
DEFAULT_DETECTOR_EXPANSION_QUEUE = (
    "data/processed/dataset_audit/stage1_detector_expansion_review_queue.csv"
)
DEFAULT_EVIDENCE_QUALITY_QUEUE = (
    "data/processed/dataset_audit/stage1_evidence_quality_review_queue.csv"
)
DEFAULT_IMAGE_DIR = "data/processed/dataset_audit/stage1_outlier_review_images"
DEFAULT_HTML = "data/processed/dataset_audit/stage1_outlier_review_index.html"
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765
DEFAULT_THRESHOLD = 3.0

TERMINAL_DECISIONS = {
    "confirm_assignment",
    "reassign_site",
    "exclude_wrong_site_unknown",
    "exclude_wrong_endpoint",
    "exclude_implausible_pka",
    "unsupported_site",
}

NEWLY_SUPPORTED_PAIR_FAMILIES = {
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

DECISION_COLUMNS = SITE_OVERRIDE_KEY_COLUMNS + [
    "review_id",
    "review_decision",
    "reviewed_site_label",
    "reviewed_site_atom_indices_json",
    "review_verified_site_attribution",
    "review_verified_transition_identity",
    "review_verified_measurement_conditions",
    "review_note",
    "reviewed_at",
    "model_abs_error",
    "model_prediction",
    "review_model_sha256",
]


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _safe_json_list(value: object) -> List:
    try:
        result = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return result if isinstance(result, list) else []


def _review_id(key_row: pd.Series) -> str:
    return "outlier_" + hashlib.sha256(repr(_site_override_key(key_row)).encode("utf-8")).hexdigest()[:20]


def _assignment_lookup(assignments: pd.DataFrame) -> Dict[Tuple[str, int, float], pd.Series]:
    experimental = assignments[
        assignments["pka_source_method"].astype(str).str.lower().eq("experimental")
    ].copy()
    lookup: Dict[Tuple[str, int, float], pd.Series] = {}
    for _, row in experimental.iterrows():
        key = (str(row["source_file"]), int(row["record_index"]), round(float(row["pka_value"]), 6))
        lookup[key] = row
    return lookup


def _candidate_payloads(
    mol: Chem.Mol,
    current_label: str,
    current_atom_maps_json: object,
    overlap_threshold: float,
) -> List[Dict]:
    resolved, pair_sites = _resolved_pair_sites(mol, overlap_threshold)
    pair_keys = {
        (str(site.get("label", "")), tuple(sorted(int(v) for v in site.get("atom_set", set()))))
        for site in pair_sites
    }
    current_atoms = {int(value) - 1 for value in _safe_json_list(current_atom_maps_json)}
    ordered = sorted(
        resolved,
        key=lambda site: (
            str(site.get("label", "")) == current_label
            and (not current_atoms or set(site.get("atom_set", set())) == current_atoms),
            maximal_site_rank_key(site),
        ),
        reverse=True,
    )
    payloads = []
    for index, site in enumerate(ordered):
        atoms = sorted(int(value) for value in site.get("atom_set", set()))
        key = (str(site.get("label", "")), tuple(atoms))
        payloads.append({
            "candidate_index": int(index),
            "label": str(site.get("label", "")),
            "family": str(site.get("pair_family", site.get("family", ""))),
            "site_type": str(site.get("type", "")),
            "atom_indices": atoms,
            "atom_text": ",".join(str(value) for value in atoms),
            "priority": int(site.get("priority", 0)),
            "specificity": int(site.get("specificity", 0)),
            "pair_supported": key in pair_keys,
            "is_current": str(site.get("label", "")) == current_label
            and (not current_atoms or set(atoms) == current_atoms),
            "is_auto": False,
            "atom_index_hit": False,
        })
    return payloads


def _load_decisions(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        decisions = pd.read_csv(path, low_memory=False)
    else:
        decisions = pd.DataFrame(columns=DECISION_COLUMNS)
    for column in DECISION_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = ""
    for column in (
        "review_id",
        "review_decision",
        "reviewed_site_label",
        "reviewed_site_atom_indices_json",
        "review_note",
        "reviewed_at",
        "review_model_sha256",
    ):
        decisions[column] = decisions[column].fillna("").astype(str)
    return decisions[DECISION_COLUMNS].copy()


def _decision_lookup(decisions: pd.DataFrame) -> Dict[Tuple[object, ...], pd.Series]:
    lookup: Dict[Tuple[object, ...], pd.Series] = {}
    for _, row in decisions.iterrows():
        lookup[_site_override_key(row)] = row
    return lookup


def build_outlier_review_queue(
    oof_path: str = DEFAULT_OOF,
    transitions_path: str = DEFAULT_TRANSITIONS,
    assignments_path: str = DEFAULT_ASSIGNMENTS,
    decisions_path: str = DEFAULT_DECISIONS,
    raw_dir: str = DEFAULT_RAW_DIR,
    threshold: float = DEFAULT_THRESHOLD,
    overlap_threshold: float = 0.5,
) -> pd.DataFrame:
    oof = pd.read_csv(oof_path, low_memory=False)
    outliers = oof[oof["abs_error"].astype(float) > float(threshold)].copy()
    outliers = outliers.sort_values("abs_error", ascending=False).reset_index(drop=True)
    transitions = pd.read_csv(transitions_path, low_memory=False)
    transition_lookup = {
        str(row["dataset_row_id"]): row for _, row in transitions.iterrows()
    }
    assignments = pd.read_csv(assignments_path, low_memory=False)
    assignments_by_source = _assignment_lookup(assignments)
    decisions = _decision_lookup(_load_decisions(decisions_path))
    mol_cache = RawMolCache(raw_dir)
    model_sha = _sha256(oof_path)

    rows: List[Dict] = []
    for rank, (_, model_row) in enumerate(outliers.iterrows(), start=1):
        transition_ids = [str(value) for value in _safe_json_list(model_row["experimental_transition_ids_json"])]
        for transition_id in transition_ids:
            transition = transition_lookup.get(transition_id)
            if transition is None:
                continue
            source_key = (
                str(transition["source_file"]),
                int(transition["record_index"]),
                round(float(transition["experimental_pka"]), 6),
            )
            assignment = assignments_by_source.get(source_key)
            if assignment is None:
                continue
            key_payload = {column: assignment.get(column) for column in SITE_OVERRIDE_KEY_COLUMNS}
            key_series = pd.Series(key_payload)
            review_id = _review_id(key_series)
            decision = decisions.get(_site_override_key(key_series))
            mol = mol_cache.get(source_key[0], source_key[1])
            candidates = _candidate_payloads(
                mol,
                current_label=str(transition["site_label"]),
                current_atom_maps_json=transition["site_atom_maps_json"],
                overlap_threshold=overlap_threshold,
            )
            decision_name = "" if decision is None else str(decision.get("review_decision", ""))
            rows.append({
                **key_payload,
                "review_id": review_id,
                "review_trigger": "stage1_scaffold_oof_abs_error_gt_threshold",
                "outlier_rank": rank,
                "review_status": decision_name or "pending",
                "review_note": "" if decision is None else str(decision.get("review_note", "")),
                "reviewed_at": "" if decision is None else str(decision.get("reviewed_at", "")),
                "review_model_sha256": model_sha,
                "stage1_molecule_id": str(model_row["molecule_id"]),
                "stage1_site_id": str(model_row["site_id"]),
                "stage1_structure_key": str(model_row["structure_key"]),
                "aggregated_experimental_pka": float(model_row["pka_value"]),
                "source_record_pka": float(transition["experimental_pka"]),
                "predicted_pka": float(model_row["pred_intrinsic_pka"]),
                "abs_error": float(model_row["abs_error"]),
                "current_site_label": str(transition["site_label"]),
                "current_site_family": str(transition["site_family"]),
                "current_site_atom_maps_json": str(transition["site_atom_maps_json"]),
                "current_protonation_center_maps_json": str(transition["protonation_center_atom_maps_json"]),
                "current_assignment_basis": str(transition["site_assignment_basis"]),
                "structural_site_assignment_confidence": str(
                    transition.get("structural_site_assignment_confidence", transition["site_assignment_confidence"])
                ),
                "source_site_evidence": str(transition.get("source_site_evidence", "unknown")),
                "source_site_evidence_confidence": str(
                    transition.get("source_site_evidence_confidence", "unknown")
                ),
                "source_assay_id": str(transition.get("source_assay_id", "")),
                "source_document_id": str(transition.get("source_document_id", "")),
                "source_molecule_id": str(transition.get("source_molecule_id", "")),
                "dataset_row_id": transition_id,
                "candidate_count": len(candidates),
                "candidate_labels": "|".join(str(candidate["label"]) for candidate in candidates),
                "candidate_sites_json": _json(candidates),
            })
    queue = pd.DataFrame(rows)
    if not queue.empty:
        queue = queue.sort_values(
            ["abs_error", "outlier_rank", "source_file", "record_index"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
    return queue


def build_detector_expansion_review_queue(
    release_dataset_path: str,
    assignments_path: str = DEFAULT_ASSIGNMENTS,
    decisions_path: str = DEFAULT_DECISIONS,
    raw_dir: str = DEFAULT_RAW_DIR,
    overlap_threshold: float = 0.5,
) -> pd.DataFrame:
    """Build review rows for newly admitted, unannotated detector-v2 families."""
    release = pd.read_csv(release_dataset_path, low_memory=False)
    override_applied = release.get(
        "override_applied", pd.Series(False, index=release.index)
    ).fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    eligible = release[
        release["site_family"].astype(str).isin(NEWLY_SUPPORTED_PAIR_FAMILIES)
        & release["site_assignment_basis"].astype(str).eq("structurally_unique_pair_site")
        & ~override_applied
    ].copy()
    assignments = pd.read_csv(assignments_path, low_memory=False)
    assignments_by_source = _assignment_lookup(assignments)
    decision_lookup = _decision_lookup(_load_decisions(decisions_path))
    mol_cache = RawMolCache(raw_dir)
    rows: List[Dict] = []

    for rank, (_, transition) in enumerate(
        eligible.sort_values(["site_family", "experimental_pka", "source_file", "record_index"]).iterrows(),
        start=1,
    ):
        source_key = (
            str(transition["source_file"]),
            int(transition["record_index"]),
            round(float(transition["experimental_pka"]), 6),
        )
        assignment = assignments_by_source.get(source_key)
        if assignment is None:
            continue
        key_payload = {column: assignment.get(column) for column in SITE_OVERRIDE_KEY_COLUMNS}
        key_series = pd.Series(key_payload)
        review_id = _review_id(key_series)
        decision = decision_lookup.get(_site_override_key(key_series))
        mol = mol_cache.get(source_key[0], source_key[1])
        candidates = _candidate_payloads(
            mol,
            current_label=str(transition["site_label"]),
            current_atom_maps_json=transition["site_atom_maps_json"],
            overlap_threshold=overlap_threshold,
        )
        decision_name = "" if decision is None else str(decision.get("review_decision", ""))
        rows.append({
            **key_payload,
            "review_id": review_id,
            "review_trigger": "detector_v2_new_family_stage1_candidate",
            "outlier_rank": 100000 + rank,
            "review_status": decision_name or "pending",
            "review_note": "" if decision is None else str(decision.get("review_note", "")),
            "reviewed_at": "" if decision is None else str(decision.get("reviewed_at", "")),
            "review_model_sha256": f"functional_group_detector:{FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION}",
            "stage1_molecule_id": "",
            "stage1_site_id": "",
            "stage1_structure_key": str(transition["input_smiles"]),
            "aggregated_experimental_pka": float(transition["experimental_pka"]),
            "source_record_pka": float(transition["experimental_pka"]),
            "predicted_pka": float("nan"),
            "abs_error": float("nan"),
            "current_site_label": str(transition["site_label"]),
            "current_site_family": str(transition["site_family"]),
            "current_site_atom_maps_json": str(transition["site_atom_maps_json"]),
            "current_protonation_center_maps_json": str(transition["protonation_center_atom_maps_json"]),
            "current_assignment_basis": str(transition["site_assignment_basis"]),
            "structural_site_assignment_confidence": str(
                transition.get("structural_site_assignment_confidence", transition["site_assignment_confidence"])
            ),
            "source_site_evidence": str(transition.get("source_site_evidence", "unknown")),
            "source_site_evidence_confidence": str(
                transition.get("source_site_evidence_confidence", "unknown")
            ),
            "source_assay_id": str(transition.get("source_assay_id", "")),
            "source_document_id": str(transition.get("source_document_id", "")),
            "source_molecule_id": str(transition.get("source_molecule_id", "")),
            "dataset_row_id": str(transition["dataset_row_id"]),
            "candidate_count": len(candidates),
            "candidate_labels": "|".join(str(candidate["label"]) for candidate in candidates),
            "candidate_sites_json": _json(candidates),
        })
    return pd.DataFrame(rows)


def build_evidence_quality_review_queue(
    release_dataset_path: str = DEFAULT_TRANSITIONS,
    assignments_path: str = DEFAULT_ASSIGNMENTS,
    decisions_path: str = DEFAULT_DECISIONS,
    raw_dir: str = DEFAULT_RAW_DIR,
    overlap_threshold: float = 0.5,
) -> pd.DataFrame:
    """Build a review queue for measurements retained only as weak labels."""
    release = pd.read_csv(release_dataset_path, low_memory=False)
    if "evidence_tier" not in release.columns:
        return pd.DataFrame()
    ambiguous = release[release["evidence_tier"].astype(str).eq("ambiguous")].copy()
    assignments = pd.read_csv(assignments_path, low_memory=False)
    assignments_by_source = _assignment_lookup(assignments)
    decision_lookup = _decision_lookup(_load_decisions(decisions_path))
    mol_cache = RawMolCache(raw_dir)
    rows: List[Dict] = []
    ambiguous = ambiguous.sort_values(
        ["unresolved_ionizable_context_count", "reference_z_distance", "source_file", "record_index"],
        ascending=[False, False, True, True],
        na_position="last",
    )
    for rank, (_, transition) in enumerate(ambiguous.iterrows(), start=1):
        source_key = (
            str(transition["source_file"]),
            int(transition["record_index"]),
            round(float(transition["experimental_pka"]), 6),
        )
        assignment = assignments_by_source.get(source_key)
        if assignment is None:
            continue
        key_payload = {column: assignment.get(column) for column in SITE_OVERRIDE_KEY_COLUMNS}
        key_series = pd.Series(key_payload)
        review_id = _review_id(key_series)
        decision = decision_lookup.get(_site_override_key(key_series))
        verified = bool(
            decision is not None
            and str(decision.get("review_verified_site_attribution", "")).lower() in {"true", "1", "yes"}
            and str(decision.get("review_verified_transition_identity", "")).lower() in {"true", "1", "yes"}
        )
        decision_name = "" if decision is None else str(decision.get("review_decision", ""))
        status = (
            decision_name
            if decision_name in TERMINAL_DECISIONS - {"confirm_assignment", "reassign_site"}
            or verified
            else "pending"
        )
        mol = mol_cache.get(source_key[0], source_key[1])
        candidates = _candidate_payloads(
            mol,
            current_label=str(transition["site_label"]),
            current_atom_maps_json=transition["site_atom_maps_json"],
            overlap_threshold=overlap_threshold,
        )
        rows.append({
            **key_payload,
            "review_id": review_id,
            "review_trigger": "evidence_quality_ambiguous",
            "outlier_rank": rank,
            "review_status": status,
            "review_note": "" if decision is None else str(decision.get("review_note", "")),
            "reviewed_at": "" if decision is None else str(decision.get("reviewed_at", "")),
            "review_model_sha256": "evidence-policy:" + str(
                transition.get("evidence_policy_schema_version", "unknown")
            ),
            "stage1_molecule_id": "",
            "stage1_site_id": "",
            "stage1_structure_key": str(transition["input_smiles"]),
            "aggregated_experimental_pka": float(transition["experimental_pka"]),
            "source_record_pka": float(transition["experimental_pka"]),
            "predicted_pka": transition.get("reference_pka", float("nan")),
            "abs_error": (
                abs(float(transition["experimental_pka"]) - float(transition["reference_pka"]))
                if pd.notna(transition.get("reference_pka")) else float("nan")
            ),
            "current_site_label": str(transition["site_label"]),
            "current_site_family": str(transition["site_family"]),
            "current_site_atom_maps_json": str(transition["site_atom_maps_json"]),
            "current_protonation_center_maps_json": str(transition["protonation_center_atom_maps_json"]),
            "current_assignment_basis": str(transition["site_assignment_basis"]),
            "structural_site_assignment_confidence": str(
                transition.get("structural_site_assignment_confidence", transition["site_assignment_confidence"])
            ),
            "source_site_evidence": str(transition.get("source_site_evidence", "unknown")),
            "source_site_evidence_confidence": str(
                transition.get("source_site_evidence_confidence", "unknown")
            ),
            "evidence_tier_reason": str(transition.get("evidence_tier_reason", "")),
            "unresolved_ionizable_contexts_json": str(
                transition.get("unresolved_ionizable_contexts_json", "[]")
            ),
            "reference_transition_definition": str(
                transition.get("reference_transition_definition", "")
            ),
            "source_original_smiles": str(transition.get("source_original_smiles", "")),
            "source_assay_id": str(transition.get("source_assay_id", "")),
            "source_document_id": str(transition.get("source_document_id", "")),
            "source_molecule_id": str(transition.get("source_molecule_id", "")),
            "dataset_row_id": str(transition["dataset_row_id"]),
            "candidate_count": len(candidates),
            "candidate_labels": "|".join(str(candidate["label"]) for candidate in candidates),
            "candidate_sites_json": _json(candidates),
        })
    return pd.DataFrame(rows)


def build_detector_reopen_review_queue(
    report_path: str = DEFAULT_REOPEN_REPORT,
    assignments_path: str = DEFAULT_ASSIGNMENTS,
    decisions_path: str = DEFAULT_DECISIONS,
    raw_dir: str = DEFAULT_RAW_DIR,
    overlap_threshold: float = 0.5,
) -> pd.DataFrame:
    """Turn detector-change reopen records into ordinary clickable reviews."""
    if not os.path.exists(report_path):
        return pd.DataFrame()
    report = pd.read_csv(report_path, low_memory=False)
    assignments = pd.read_csv(assignments_path, low_memory=False)
    assignments_by_source = _assignment_lookup(assignments)
    decision_lookup = _decision_lookup(_load_decisions(decisions_path))
    mol_cache = RawMolCache(raw_dir)
    rows: List[Dict] = []
    for rank, (_, reopened) in enumerate(report.iterrows(), start=1):
        source_key = (
            str(reopened["source_file"]),
            int(reopened["record_index"]),
            round(float(reopened["pka_value"]), 6),
        )
        assignment = assignments_by_source.get(source_key)
        if assignment is None:
            continue
        key_payload = {column: assignment.get(column) for column in SITE_OVERRIDE_KEY_COLUMNS}
        key_series = pd.Series(key_payload)
        decision = decision_lookup.get(_site_override_key(key_series))
        mol = mol_cache.get(source_key[0], source_key[1])
        candidates = _candidate_payloads(
            mol, current_label="", current_atom_maps_json="[]",
            overlap_threshold=overlap_threshold,
        )
        current = next((candidate for candidate in candidates if candidate["pair_supported"]), None)
        if current is None and candidates:
            current = candidates[0]
        if current is not None:
            current["is_current"] = True
        decision_name = "" if decision is None else str(decision.get("review_decision", ""))
        pka_type = str(assignment.get("pka_type_raw", ""))
        source_evidence = "source_ionization_type_only_no_site" if pka_type and pka_type != "nan" else "unannotated_molecule_level_pka"
        source_confidence = "low" if source_evidence.startswith("source_ionization") else "none"
        rows.append({
            **key_payload,
            "review_id": _review_id(key_series),
            "review_trigger": "detector_schema_change_reopened_decision",
            "outlier_rank": 90000 + rank,
            "review_status": decision_name or "pending",
            "review_note": "" if decision is None else str(decision.get("review_note", "")),
            "reviewed_at": "" if decision is None else str(decision.get("reviewed_at", "")),
            "review_model_sha256": f"functional_group_detector:{FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION}",
            "stage1_molecule_id": "",
            "stage1_site_id": "",
            "stage1_structure_key": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
            "aggregated_experimental_pka": float(reopened["pka_value"]),
            "source_record_pka": float(reopened["pka_value"]),
            "predicted_pka": float("nan"),
            "abs_error": float("nan"),
            "current_site_label": "" if current is None else str(current["label"]),
            "current_site_family": "" if current is None else str(current["family"]),
            "current_site_atom_maps_json": _json(
                [] if current is None else [int(value) + 1 for value in current["atom_indices"]]
            ),
            "current_protonation_center_maps_json": "[]",
            "current_assignment_basis": "detector_schema_change_requires_manual_rereview",
            "structural_site_assignment_confidence": "none",
            "source_site_evidence": source_evidence,
            "source_site_evidence_confidence": source_confidence,
            "source_assay_id": "",
            "source_document_id": "",
            "source_molecule_id": "",
            "dataset_row_id": "",
            "candidate_count": len(candidates),
            "candidate_labels": "|".join(str(candidate["label"]) for candidate in candidates),
            "candidate_sites_json": _json(candidates),
        })
    return pd.DataFrame(rows)


def _refresh_review_status(queue: pd.DataFrame, decisions_path: str) -> pd.DataFrame:
    decisions = _decision_lookup(_load_decisions(decisions_path))
    work = queue.copy()
    for index, row in work.iterrows():
        decision = decisions.get(_site_override_key(row))
        decision_name = "" if decision is None else str(decision.get("review_decision", ""))
        if str(row.get("review_trigger", "")) == "evidence_quality_ambiguous" and decision is not None:
            verified = (
                str(decision.get("review_verified_site_attribution", "")).lower() in {"true", "1", "yes"}
                and str(decision.get("review_verified_transition_identity", "")).lower() in {"true", "1", "yes"}
            )
            structural_only = decision_name in {"confirm_assignment", "reassign_site"} and not verified
            work.at[index, "review_status"] = "pending" if structural_only else decision_name or "pending"
        else:
            work.at[index, "review_status"] = decision_name or "pending"
        work.at[index, "review_note"] = "" if decision is None else str(decision.get("review_note", ""))
        work.at[index, "reviewed_at"] = "" if decision is None else str(decision.get("reviewed_at", ""))
    return work


def _runtime_candidates(row: pd.Series) -> List[Dict]:
    payloads = _safe_json_list(row["candidate_sites_json"])
    return [{**payload, "atom_set": set(payload.get("atom_indices", []))} for payload in payloads]


def _optional_number(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not yet modeled"
    return f"{number:.{digits}f}" if pd.notna(number) else "not yet modeled"


def _queue_rank_label(row: pd.Series) -> str:
    trigger = str(row.get("review_trigger", ""))
    if trigger == "detector_schema_change_reopened_decision":
        return "reopened after detector change"
    if trigger == "detector_v2_new_family_stage1_candidate":
        return "new detector family"
    return f"#{int(row['outlier_rank'])}"


def _save_decisions(path: str, decisions: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    work = decisions[DECISION_COLUMNS].copy()
    work = work.drop_duplicates(subset=SITE_OVERRIDE_KEY_COLUMNS, keep="last")
    temporary = path + ".tmp"
    work.to_csv(temporary, index=False)
    os.replace(temporary, path)


def reopen_reviews_for_detector_change(
    decisions_path: str,
    raw_dir: str,
    history_path: str = DEFAULT_DECISION_HISTORY,
    report_path: str = DEFAULT_REOPEN_REPORT,
    overlap_threshold: float = 0.5,
) -> pd.DataFrame:
    """Reopen decisions made impossible or newly actionable by detector v2.

    Prior decisions are appended to an immutable history table before the
    current record is marked ``defer``. Confirmed assignments that remain
    atom-exact are never reopened merely because another site was detected.
    """
    decisions = _load_decisions(decisions_path)
    mol_cache = RawMolCache(raw_dir)
    now = datetime.now(timezone.utc).isoformat()
    report_rows: List[Dict] = []
    history_rows: List[Dict] = []

    for index, decision in decisions.iterrows():
        decision_name = str(decision.get("review_decision", ""))
        if decision_name not in TERMINAL_DECISIONS:
            continue
        mol = mol_cache.get(str(decision["source_file"]), int(decision["record_index"]))
        _, pair_sites = _resolved_pair_sites(mol, overlap_threshold)
        new_sites = [
            site for site in pair_sites
            if str(site.get("pair_family", "")) in NEWLY_SUPPORTED_PAIR_FAMILIES
        ]
        invalid_exact = (
            decision_name in {"confirm_assignment", "reassign_site"}
            and _select_reviewed_site(decision, pair_sites) is None
        )
        newly_actionable = (
            decision_name in {"exclude_wrong_site_unknown", "unsupported_site"}
            and bool(new_sites)
        )
        if not invalid_exact and not newly_actionable:
            continue

        reason = (
            "reviewed_exact_site_no_longer_exists_after_detector_change"
            if invalid_exact
            else "newly_supported_conjugate_pair_detected"
        )
        candidate_payload = [
            {
                "label": str(site.get("label", "")),
                "family": str(site.get("pair_family", "")),
                "atom_indices": sorted(int(value) for value in site.get("atom_set", set())),
            }
            for site in (pair_sites if invalid_exact else new_sites)
        ]
        history_rows.append({
            **decision.to_dict(),
            "superseded_at": now,
            "superseded_reason": reason,
            "replacement_detector_schema_version": FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION,
        })
        old_note = str(decision.get("review_note", ""))
        if old_note.lower() == "nan":
            old_note = ""
        old_label = str(decision.get("reviewed_site_label", ""))
        if old_label.lower() == "nan":
            old_label = ""
        decisions.at[index, "review_decision"] = "defer"
        decisions.at[index, "reviewed_site_label"] = ""
        decisions.at[index, "reviewed_site_atom_indices_json"] = "[]"
        decisions.at[index, "review_note"] = (
            f"Reopened for detector {FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION}: {reason}; "
            f"prior_decision={decision_name}; prior_label={old_label or 'none'}"
            + (f"; prior_note={old_note}" if old_note else "")
        )
        decisions.at[index, "reviewed_at"] = now
        report_rows.append({
            "review_id": str(decision["review_id"]),
            "source_file": str(decision["source_file"]),
            "record_index": int(decision["record_index"]),
            "smiles": str(decision["smiles"]),
            "pka_value": float(decision["pka_value"]),
            "prior_decision": decision_name,
            "prior_site_label": old_label,
            "reopen_reason": reason,
            "new_candidate_sites_json": _json(candidate_payload),
            "detector_schema_version": FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION,
        })

    report = pd.DataFrame(report_rows)
    if not report.empty:
        _save_decisions(decisions_path, decisions)
        history_columns = DECISION_COLUMNS + [
            "superseded_at", "superseded_reason", "replacement_detector_schema_version",
        ]
        old_history = (
            pd.read_csv(history_path, low_memory=False)
            if os.path.exists(history_path)
            else pd.DataFrame(columns=history_columns)
        )
        history = pd.concat([old_history, pd.DataFrame(history_rows)], ignore_index=True)
        history = history.drop_duplicates(
            subset=["review_id", "reviewed_at", "superseded_reason"], keep="last"
        )
        Path(history_path).parent.mkdir(parents=True, exist_ok=True)
        history.to_csv(history_path, index=False)
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(report_path, index=False)
    return report


def _upsert_decision(
    decisions: pd.DataFrame,
    row: pd.Series,
    decision: str,
    site_label: str = "",
    atom_indices: Optional[Sequence[int]] = None,
    note: str = "",
    evidence_attestations: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    key = _site_override_key(row)
    keep = pd.Series(
        [_site_override_key(existing) != key for _, existing in decisions.iterrows()],
        index=decisions.index,
        dtype=bool,
    )
    work = decisions.loc[keep].copy()
    payload = {column: row.get(column) for column in SITE_OVERRIDE_KEY_COLUMNS}
    evidence_attestations = evidence_attestations or {}
    payload.update({
        "review_id": row["review_id"],
        "review_decision": decision,
        "reviewed_site_label": site_label,
        "reviewed_site_atom_indices_json": _json(list(atom_indices or [])),
        "review_verified_site_attribution": bool(
            evidence_attestations.get("site_attribution", False)
        ),
        "review_verified_transition_identity": bool(
            evidence_attestations.get("transition_identity", False)
        ),
        "review_verified_measurement_conditions": bool(
            evidence_attestations.get("measurement_conditions", False)
        ),
        "review_note": note,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "model_abs_error": float(row["abs_error"]),
        "model_prediction": float(row["predicted_pka"]),
        "review_model_sha256": str(row["review_model_sha256"]),
    })
    return pd.concat([work, pd.DataFrame([payload])], ignore_index=True)


def _render_row(
    row: pd.Series,
    mol_cache: RawMolCache,
    image_dir: str,
) -> str:
    mol = mol_cache.get(str(row["source_file"]), int(row["record_index"]))
    candidates = _runtime_candidates(row)
    render_row = row.copy()
    render_row["selected_group_label"] = row["current_site_label"]
    render_row["auto_selected_group_label"] = row["current_site_label"]
    trigger = str(row.get("review_trigger", ""))
    if trigger == "detector_schema_change_reopened_decision":
        render_row["review_reason"] = (
            f"reopened after detector schema change; source-evidence={row['source_site_evidence']}"
        )
    elif trigger == "detector_v2_new_family_stage1_candidate":
        render_row["review_reason"] = (
            f"new detector-v2 Stage-1 candidate; source-evidence={row['source_site_evidence']}"
        )
    else:
        render_row["review_reason"] = (
            f"OOF |error|={_optional_number(row['abs_error'], 2)}; "
            f"pred={_optional_number(row['predicted_pka'], 2)}; "
            f"source-evidence={row['source_site_evidence']}"
        )
    out_path = os.path.join(image_dir, f"{row['review_id']}.png")
    return _render_review_png(mol, render_row, candidates, out_path)


def _write_html(queue: pd.DataFrame, html_path: str, image_dir: str) -> None:
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for _, row in queue.iterrows():
        image_rel = os.path.relpath(
            os.path.join(image_dir, f"{row['review_id']}.png"),
            start=os.path.dirname(html_path) or ".",
        )
        candidates = _safe_json_list(row["candidate_sites_json"])
        candidate_text = "<br>".join(
            html.escape(
                f"[{candidate['candidate_index']}] {candidate['label']} atoms={candidate['atom_indices']} "
                f"{'pair-supported' if candidate['pair_supported'] else 'not-supported'}"
            )
            for candidate in candidates
        )
        cards.append(f"""
        <article>
          <h2>{html.escape(_queue_rank_label(row))} {html.escape(str(row['review_id']))}</h2>
          <p><b>experimental {float(row['source_record_pka']):.3f}</b> · predicted {_optional_number(row['predicted_pka'])} · error {_optional_number(row['abs_error'])}</p>
          <p>{html.escape(str(row['source_file']))} record {int(row['record_index'])} · assay {html.escape(str(row['source_assay_id']))}</p>
          <p>current: <b>{html.escape(str(row['current_site_label']))}</b> · source evidence: {html.escape(str(row['source_site_evidence']))} · status: {html.escape(str(row['review_status']))}</p>
          <img src="{html.escape(image_rel)}" loading="lazy">
          <p class="candidates">{candidate_text}</p>
        </article>
        """)
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage 1 pKa outlier review</title>
<style>
body{{font:15px system-ui,sans-serif;margin:24px;background:#f5f5f5;color:#222}}
header{{position:sticky;top:0;background:#fff;padding:12px;border:1px solid #ddd;z-index:2}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px;margin-top:16px}}
article{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px}}
img{{width:100%;height:auto;border:1px solid #eee}} h2{{font-size:17px;margin:0 0 6px}}
.candidates{{font-family:ui-monospace,monospace;font-size:12px}}
</style></head><body>
<header><b>Stage 1 outlier review pack</b> — {len(queue)} source measurements. Record decisions with <code>python scripts/stage1_outlier_review.py</code>.</header>
<main class="grid">{''.join(cards)}</main></body></html>"""
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(document)


def _current_candidate(row: pd.Series, candidates: List[Dict]) -> Optional[Dict]:
    return next((value for value in candidates if value.get("is_current")), None)


def _decision_for_action(
    decisions: pd.DataFrame,
    row: pd.Series,
    action: str,
    form: Dict[str, List[str]],
) -> pd.DataFrame:
    note = form.get("note", [""])[0].strip()
    attestations = {
        "site_attribution": "verify_site_attribution" in form,
        "transition_identity": "verify_transition_identity" in form,
        "measurement_conditions": "verify_measurement_conditions" in form,
    }
    if (
        str(row.get("review_trigger", "")) == "evidence_quality_ambiguous"
        and action in {"confirm_assignment", "reassign_site"}
        and not (attestations["site_attribution"] and attestations["transition_identity"])
    ):
        raise ValueError(
            "Ambiguous evidence requires explicit source-site and transition verification, "
            "or an exclusion/unsupported-site decision"
        )
    candidates = _runtime_candidates(row)
    if action == "confirm_assignment":
        raw_index = form.get("candidate_index", [""])[0]
        if raw_index:
            try:
                candidate_index = int(raw_index)
            except ValueError as exc:
                raise ValueError("The selected candidate index is invalid") from exc
            selected = next(
                (value for value in candidates if int(value["candidate_index"]) == candidate_index),
                None,
            )
            if selected is None:
                raise ValueError("The selected candidate is not part of this record")
            if not selected.get("is_current"):
                decision = "reassign_site" if selected.get("pair_supported") else "unsupported_site"
                return _upsert_decision(
                    decisions, row, decision, str(selected["label"]), selected["atom_indices"], note,
                    evidence_attestations=attestations,
                )
        current = _current_candidate(row, candidates)
        atom_indices = (
            current["atom_indices"]
            if current is not None
            else [int(value) - 1 for value in _safe_json_list(row["current_site_atom_maps_json"])]
        )
        return _upsert_decision(
            decisions, row, action, str(row["current_site_label"]), atom_indices, note,
            evidence_attestations=attestations,
        )
    if action == "reassign_site":
        raw_index = form.get("candidate_index", [""])[0]
        try:
            candidate_index = int(raw_index)
        except ValueError as exc:
            raise ValueError("Select a candidate before reassigning") from exc
        candidate = next(
            (value for value in candidates if int(value["candidate_index"]) == candidate_index),
            None,
        )
        if candidate is None:
            raise ValueError("The selected candidate is not part of this record")
        decision = "reassign_site" if candidate.get("pair_supported") else "unsupported_site"
        return _upsert_decision(
            decisions, row, decision, str(candidate["label"]), candidate["atom_indices"], note,
            evidence_attestations=attestations,
        )
    if action == "unsupported_site":
        label = form.get("unsupported_label", [""])[0].strip()
        raw_atom = form.get("unsupported_atom", [""])[0].strip()
        if not label or not raw_atom:
            raise ValueError("Unsupported-site decisions require a label and zero-based atom index")
        try:
            atom_index = int(raw_atom)
        except ValueError as exc:
            raise ValueError("Unsupported-site atom index must be an integer") from exc
        return _upsert_decision(
            decisions, row, action, label, [atom_index], note,
            evidence_attestations=attestations,
        )
    if action in {
        "exclude_wrong_site_unknown",
        "exclude_wrong_endpoint",
        "exclude_implausible_pka",
        "defer",
    }:
        return _upsert_decision(
            decisions, row, action, note=note, evidence_attestations=attestations
        )
    raise ValueError(f"Unsupported review action: {action}")


def _web_review_page(
    queue: pd.DataFrame,
    row: Optional[pd.Series],
    status_by_id: Dict[str, str],
    message: str = "",
) -> str:
    queue_review_ids = set(queue["review_id"].astype(str))
    terminal_count = sum(
        status_by_id.get(review_id, "") in TERMINAL_DECISIONS
        for review_id in queue_review_ids
    )
    deferred_count = sum(
        status_by_id.get(review_id, "") == "defer"
        for review_id in queue_review_ids
    )
    total = len(queue)
    pending_count = total - terminal_count
    progress = 100.0 if total == 0 else 100.0 * terminal_count / total
    if row is None:
        if pending_count == 0:
            heading = "Stage 1 review complete"
            explanation = "Every queued source measurement has a terminal decision. The corrected release can now be rebuilt."
        else:
            heading = "This review pass is complete"
            explanation = (
                f"{pending_count} measurement(s) remain deferred. Restart the reviewer when you are ready "
                "to give them terminal decisions; do not rebuild the final dataset yet."
            )
        main = f"""
        <section class="finished">
          <h1>{heading}</h1>
          <p>{explanation}</p>
        </section>
        """
    else:
        candidates = _safe_json_list(row["candidate_sites_json"])
        choices = []
        for candidate in candidates:
            flags = []
            if candidate.get("is_current"):
                flags.append("CURRENT")
            flags.append("SUPPORTED" if candidate.get("pair_supported") else "UNSUPPORTED")
            choices.append(f"""
            <label class="candidate {'current' if candidate.get('is_current') else ''}">
              <input type="radio" name="candidate_index" value="{int(candidate['candidate_index'])}">
              <span><b>[{int(candidate['candidate_index'])}] {html.escape(str(candidate['label']))}</b>
              · {html.escape(str(candidate['family']))} · atoms {html.escape(str(candidate['atom_indices']))}
              <em>{' · '.join(flags)}</em></span>
            </label>""")
        group_size = int((queue["stage1_structure_key"] == row["stage1_structure_key"]).sum())
        source_assay = str(row.get("source_assay_id", ""))
        source_document = str(row.get("source_document_id", ""))
        source_molecule = str(row.get("source_molecule_id", ""))
        main = f"""
        <main>
          <section class="facts">
            <div><small>Experimental</small><strong>{float(row['source_record_pka']):.3f}</strong></div>
            <div><small>Stage-1 OOF prediction</small><strong>{_optional_number(row['predicted_pka'])}</strong></div>
            <div class="error"><small>Absolute error</small><strong>{_optional_number(row['abs_error'])}</strong></div>
            <div><small>Review trigger</small><strong>{html.escape(_queue_rank_label(row))}</strong></div>
          </section>
          <section class="record">
            <div class="structure">
              <img src="/image/{quote(str(row['review_id']))}.png" alt="Atom-indexed molecular structure">
            </div>
            <div class="details">
              <h2>{html.escape(str(row['current_site_label']))}</h2>
              <dl>
                <dt>Current site atoms</dt><dd>{html.escape(str(row['current_site_atom_maps_json']))} (one-based maps)</dd>
                <dt>Assignment basis</dt><dd>{html.escape(str(row['current_assignment_basis']))}</dd>
                <dt>Source evidence</dt><dd>{html.escape(str(row['source_site_evidence']))} ({html.escape(str(row['source_site_evidence_confidence']))})</dd>
                <dt>Evidence-tier issue</dt><dd>{html.escape(str(row.get('evidence_tier_reason', '')))}</dd>
                <dt>Unresolved contexts</dt><dd>{html.escape(str(row.get('unresolved_ionizable_contexts_json', '[]')))}</dd>
                <dt>Reference transition</dt><dd>{html.escape(str(row.get('reference_transition_definition', '')))}</dd>
                <dt>Original SMILES</dt><dd class="mono">{html.escape(str(row.get('source_original_smiles', '')))}</dd>
                <dt>Raw source</dt><dd>{html.escape(str(row['source_file']))} · record {int(row['record_index'])}</dd>
                <dt>Identifiers</dt><dd>assay {html.escape(source_assay)} · document {html.escape(source_document)} · molecule {html.escape(source_molecule)}</dd>
                <dt>Same Stage-1 structure</dt><dd>{group_size} source measurement{'s' if group_size != 1 else ''}; decisions remain record-specific</dd>
                <dt>SMILES</dt><dd class="mono">{html.escape(str(row['smiles']))}</dd>
                <dt>Review ID</dt><dd class="mono">{html.escape(str(row['review_id']))}</dd>
              </dl>
            </div>
          </section>
          <form method="post" action="/decision">
            <input type="hidden" name="review_id" value="{html.escape(str(row['review_id']))}">
            <section class="choices">
              <h3>Detected candidates <span>zero-based atom indices</span></h3>
              {''.join(choices)}
            </section>
            <section class="evidence-attestations">
              <h3>Independent evidence checks <span>leave unchecked unless the source itself was inspected</span></h3>
              <label><input type="checkbox" name="verify_site_attribution"> Source explicitly attributes this pKa to the selected site</label>
              <label><input type="checkbox" name="verify_transition_identity"> Acid/base charge transition was verified, not inferred from the processed drawing</label>
              <label><input type="checkbox" name="verify_measurement_conditions"> Solvent/scale/temperature metadata are adequate for this training use</label>
            </section>
            <label class="note">Review note <input name="note" placeholder="Evidence or reasoning (recommended)"></label>
            <section class="primary-actions">
              <button name="action" value="confirm_assignment" class="confirm">Confirm structural site selection</button>
              <button name="action" value="reassign_site" class="reassign">Reassign to selected candidate</button>
            </section>
            <section class="reject-actions">
              <button name="action" value="exclude_wrong_site_unknown">Wrong site; correct site unknown</button>
              <button name="action" value="exclude_wrong_endpoint">Wrong/non-pKa endpoint</button>
              <button name="action" value="exclude_implausible_pka">Implausible/incorrect pKa</button>
              <button name="action" value="defer">Defer to a later session</button>
            </section>
            <details>
              <summary>Detected chemistry is missing or unsupported</summary>
              <div class="unsupported">
                <input name="unsupported_label" placeholder="site label">
                <input name="unsupported_atom" placeholder="zero-based atom index" inputmode="numeric">
                <button name="action" value="unsupported_site">Record unsupported site + quarantine</button>
              </div>
            </details>
          </form>
        </main>
        """
    alert = f'<div class="alert">{html.escape(message)}</div>' if message else ""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 1 pKa outlier review</title>
<style>
:root{{--ink:#20242b;--muted:#68707c;--line:#d9dee7;--paper:#fff;--bg:#eef1f5;--blue:#285bb5;--green:#177245;--red:#a63030}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:3;background:#172033;color:white;padding:13px 24px;display:flex;gap:24px;align-items:center}}
header b{{font-size:18px}} .progress{{flex:1;max-width:560px;height:12px;background:#465067;border-radius:8px;overflow:hidden}}
.progress i{{display:block;height:100%;background:#5fd19a;width:{progress:.3f}%}} header span{{white-space:nowrap}}
.alert{{margin:18px auto -4px;max-width:1280px;padding:10px 14px;background:#ffe8e6;border:1px solid #dc8c86;border-radius:6px}}
main,.finished{{max-width:1280px;margin:18px auto;padding:0 18px}} .facts{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px}}
.facts div{{background:var(--paper);border:1px solid var(--line);padding:10px 14px;border-radius:7px}} .facts small{{display:block;color:var(--muted)}}
.facts strong{{font-size:24px}} .facts .error strong{{color:var(--red)}} .record{{display:grid;grid-template-columns:minmax(480px,2fr) minmax(330px,1fr);gap:14px}}
.structure,.details,form,.finished{{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:14px}} img{{width:100%;height:auto;display:block}}
h2{{margin:0 0 8px}} dl{{display:grid;grid-template-columns:130px 1fr;gap:7px 12px;margin:0}} dt{{color:var(--muted)}} dd{{margin:0;overflow-wrap:anywhere}}
.mono{{font:12px ui-monospace,monospace}} form{{margin-top:14px}} h3{{margin:0 0 8px}} h3 span{{font-size:12px;color:var(--muted);font-weight:normal}}
.choices{{display:grid;gap:7px}} .candidate{{display:flex;gap:8px;border:1px solid var(--line);padding:9px;border-radius:6px;cursor:pointer}}
.candidate.current{{border-color:#6d96df;background:#f3f7ff}} .candidate em{{font-size:11px;color:var(--blue);font-style:normal}}
.evidence-attestations{{display:grid;gap:7px;margin-top:14px;padding:11px;border:1px solid #d8c48e;background:#fffbea;border-radius:6px}}
.evidence-attestations label{{display:flex;gap:8px;align-items:flex-start}} .evidence-attestations input{{margin-top:3px}}
.note{{display:grid;grid-template-columns:110px 1fr;align-items:center;margin:14px 0}} input{{padding:9px;border:1px solid #aeb6c4;border-radius:5px}}
button{{border:0;border-radius:5px;padding:10px 13px;cursor:pointer;font-weight:650}} .primary-actions,.reject-actions{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}}
.confirm{{background:var(--green);color:white}} .reassign{{background:var(--blue);color:white}} .reject-actions button{{background:#eee1e1;color:#722}}
details{{margin-top:12px}} .unsupported{{display:flex;gap:8px;margin-top:8px}} .unsupported button{{background:#6d4c8d;color:white}}
.finished{{text-align:center;margin-top:80px}} @media(max-width:850px){{.record{{grid-template-columns:1fr}}.facts{{grid-template-columns:1fr 1fr}}dl{{grid-template-columns:1fr}}}}
</style></head><body>
<header><b>Stage 1 pKa review</b><div class="progress"><i></i></div><span>{terminal_count}/{total} complete · {pending_count} remaining · {deferred_count} deferred</span></header>
{alert}{main}
</body></html>"""


def serve_web_reviewer(
    queue: pd.DataFrame,
    decisions_path: str,
    image_dir: str,
    host: str = DEFAULT_WEB_HOST,
    port: int = DEFAULT_WEB_PORT,
) -> None:
    """Serve a local, persistent click-through reviewer with no extra dependencies."""
    rows_by_id = {str(row["review_id"]): row for _, row in queue.iterrows()}
    lock = threading.Lock()
    session_skipped: set[str] = set()

    def state() -> Tuple[pd.DataFrame, Dict[str, str]]:
        decisions = _load_decisions(decisions_path)
        status = {
            str(row.get("review_id", "")): str(row.get("review_decision", ""))
            for _, row in decisions.iterrows()
        }
        return decisions, status

    def next_row(status: Dict[str, str]) -> Optional[pd.Series]:
        for _, candidate in queue.iterrows():
            review_id = str(candidate["review_id"])
            if status.get(review_id, "") not in TERMINAL_DECISIONS and review_id not in session_skipped:
                return candidate
        return None

    class ReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            print(f"review-web: {fmt % args}")

        def _respond(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path.startswith("/image/") and parsed.path.endswith(".png"):
                review_id = os.path.basename(parsed.path)[:-4]
                if review_id not in rows_by_id:
                    self._respond(b"Not found", "text/plain; charset=utf-8", 404)
                    return
                image_path = os.path.join(image_dir, f"{review_id}.png")
                if not os.path.exists(image_path):
                    self._respond(b"Image not found", "text/plain; charset=utf-8", 404)
                    return
                self._respond(Path(image_path).read_bytes(), mimetypes.guess_type(image_path)[0] or "image/png")
                return
            if parsed.path != "/":
                self._respond(b"Not found", "text/plain; charset=utf-8", 404)
                return
            _, status = state()
            requested = parse_qs(parsed.query).get("review_id", [""])[0]
            row = rows_by_id.get(requested) if requested else next_row(status)
            page = _web_review_page(queue, row, status)
            self._respond(page.encode("utf-8"), "text/html; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/decision":
                self._respond(b"Not found", "text/plain; charset=utf-8", 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                form = parse_qs(self.rfile.read(length).decode("utf-8"), keep_blank_values=True)
                review_id = form.get("review_id", [""])[0]
                action = form.get("action", [""])[0]
                row = rows_by_id.get(review_id)
                if row is None:
                    raise ValueError("Unknown review record")
                with lock:
                    decisions, _ = state()
                    updated = _decision_for_action(decisions, row, action, form)
                    _save_decisions(decisions_path, updated)
                    if action == "defer":
                        session_skipped.add(review_id)
                self.send_response(303)
                self.send_header("Location", "/")
                self.end_headers()
            except ValueError as exc:
                _, status = state()
                row = rows_by_id.get(form.get("review_id", [""])[0]) if "form" in locals() else None
                if row is None:
                    row = next_row(status)
                page = _web_review_page(queue, row, status, str(exc))
                self._respond(page.encode("utf-8"), "text/html; charset=utf-8", 400)

    server = ThreadingHTTPServer((host, int(port)), ReviewHandler)
    print(f"Stage 1 reviewer: http://{host}:{port}")
    print("Decisions are saved immediately. Stop with Ctrl-C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _print_review(row: pd.Series, candidates: List[Dict], image_path: str) -> None:
    print("=" * 100)
    print(
        f"rank={_queue_rank_label(row)} {row['review_id']}  "
        f"experimental={float(row['source_record_pka']):.3f} "
        f"predicted={_optional_number(row['predicted_pka'])} error={_optional_number(row['abs_error'])}"
    )
    print(f"source={row['source_file']}#{int(row['record_index'])} assay={row['source_assay_id']} document={row['source_document_id']}")
    print(f"SMILES={row['smiles']}")
    print(
        f"current={row['current_site_label']} family={row['current_site_family']} "
        f"atoms={row['current_site_atom_maps_json']} basis={row['current_assignment_basis']}"
    )
    print(f"source evidence={row['source_site_evidence']} ({row['source_site_evidence_confidence']})")
    print(f"image={image_path}")
    for candidate in candidates:
        flags = []
        if candidate.get("is_current"):
            flags.append("CURRENT")
        flags.append("PAIR" if candidate.get("pair_supported") else "UNSUPPORTED")
        print(
            f"  [{candidate['candidate_index']}] {candidate['label']} "
            f"family={candidate['family']} atoms={candidate['atom_indices']} {' '.join(flags)}"
        )
    print("commands:")
    print("  c [note]       confirm current site and value")
    print("  r <index> [note] reassign to an exact detected candidate")
    print("  w [note]       wrong site, correct site unknown -> quarantine")
    print("  e [note]       wrong/non-molecular endpoint -> quarantine")
    print("  v [note]       experimental pKa is implausible/incorrect -> quarantine")
    print("  u <label> <atom-index>  missing/unsupported site -> quarantine for rule work")
    print("  d [note]       defer, keeping it in future review queues")
    print("  s              skip for now without recording | q quit")


def review_outliers(
    queue: pd.DataFrame,
    decisions_path: str,
    raw_dir: str,
    image_dir: str,
    html_path: str,
    include_reviewed: bool = False,
    max_records: int = 0,
    export_only: bool = False,
    render_all: bool = False,
    open_images: bool = True,
) -> int:
    decisions = _load_decisions(decisions_path)
    pending = queue.copy()
    if not include_reviewed:
        pending = pending[~pending["review_status"].isin(TERMINAL_DECISIONS)].copy()
    if max_records > 0:
        pending = pending.head(int(max_records)).copy()
    mol_cache = RawMolCache(raw_dir)

    if render_all:
        for _, row in queue.iterrows():
            _render_row(row, mol_cache, image_dir)
        _write_html(queue, html_path, image_dir)
        print(f"Saved visual review index: {html_path}")

    if export_only:
        return len(pending)

    for _, row in pending.iterrows():
        candidates = _runtime_candidates(row)
        image_path = _render_row(row, mol_cache, image_dir)
        if open_images:
            _try_open_image(image_path)
        _print_review(row, candidates, image_path)
        while True:
            raw = input("outlier-review> ").strip()
            if not raw:
                continue
            command, *rest = raw.split(maxsplit=1)
            command = command.lower()
            tail = rest[0].strip() if rest else ""
            if command in {"q", "quit", "exit"}:
                _save_decisions(decisions_path, decisions)
                return len(pending)
            if command in {"s", "skip"}:
                break
            if command == "c":
                current_candidate = next(
                    (value for value in candidates if value.get("is_current")),
                    None,
                )
                decisions = _upsert_decision(
                    decisions, row, "confirm_assignment",
                    site_label=str(row["current_site_label"]),
                    atom_indices=(
                        current_candidate["atom_indices"]
                        if current_candidate is not None
                        else [int(value) - 1 for value in _safe_json_list(row["current_site_atom_maps_json"])]
                    ),
                    note=tail,
                )
            elif command == "r":
                choice_text, *note_parts = tail.split(maxsplit=1)
                if not choice_text.isdigit():
                    print("  use: r <candidate-index> [note]")
                    continue
                candidate = next(
                    (value for value in candidates if int(value["candidate_index"]) == int(choice_text)),
                    None,
                )
                if candidate is None:
                    print("  invalid candidate index")
                    continue
                decision_name = "reassign_site" if candidate.get("pair_supported") else "unsupported_site"
                decisions = _upsert_decision(
                    decisions, row, decision_name,
                    site_label=str(candidate["label"]), atom_indices=candidate["atom_indices"],
                    note=note_parts[0] if note_parts else "",
                )
            elif command in {"w", "e", "v", "d"}:
                decision_name = {
                    "w": "exclude_wrong_site_unknown",
                    "e": "exclude_wrong_endpoint",
                    "v": "exclude_implausible_pka",
                    "d": "defer",
                }[command]
                decisions = _upsert_decision(decisions, row, decision_name, note=tail)
            elif command == "u":
                parts = tail.split()
                if len(parts) < 2:
                    print("  use: u <site-label> <zero-based-atom-index>")
                    continue
                try:
                    atom_index = int(parts[1])
                except ValueError:
                    print("  atom index must be an integer")
                    continue
                decisions = _upsert_decision(
                    decisions, row, "unsupported_site",
                    site_label=parts[0], atom_indices=[atom_index],
                    note="manual missing-site annotation",
                )
            else:
                print("  unrecognized command")
                continue
            _save_decisions(decisions_path, decisions)
            print("  decision saved")
            break
    return len(pending)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", default=DEFAULT_OOF)
    parser.add_argument("--transitions", default=DEFAULT_TRANSITIONS)
    parser.add_argument("--assignments", default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS)
    parser.add_argument("--decision-history", default=DEFAULT_DECISION_HISTORY)
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--detector-expansion-queue", default=DEFAULT_DETECTOR_EXPANSION_QUEUE)
    parser.add_argument("--evidence-quality-queue", default=DEFAULT_EVIDENCE_QUALITY_QUEUE)
    parser.add_argument(
        "--detector-expansion-release",
        default="",
        help="Candidate release CSV used to discover new detector-v2 Stage-1 families",
    )
    parser.add_argument("--reopen-report", default=DEFAULT_REOPEN_REPORT)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--html", default=DEFAULT_HTML)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--include-reviewed", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--render-all", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--web", action="store_true", help="Run the local click-through reviewer")
    parser.add_argument("--host", default=DEFAULT_WEB_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument(
        "--reopen-detector-gaps",
        action="store_true",
        help="Audit completed reviews and reopen only records affected by newly supported detector families",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reopen_detector_gaps:
        reopened = reopen_reviews_for_detector_change(
            decisions_path=args.decisions,
            raw_dir=args.raw_dir,
            history_path=args.decision_history,
            report_path=args.reopen_report,
            overlap_threshold=args.overlap_threshold,
        )
        print(
            f"reopened_for_detector_change={len(reopened)} "
            f"detector_schema_version={FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION}"
        )
        if not reopened.empty:
            print(f"reopen_report={args.reopen_report}")
            print(f"decision_history={args.decision_history}")
    queue = build_outlier_review_queue(
        oof_path=args.oof,
        transitions_path=args.transitions,
        assignments_path=args.assignments,
        decisions_path=args.decisions,
        raw_dir=args.raw_dir,
        threshold=args.threshold,
        overlap_threshold=args.overlap_threshold,
    )
    if args.detector_expansion_release:
        expansion = build_detector_expansion_review_queue(
            release_dataset_path=args.detector_expansion_release,
            assignments_path=args.assignments,
            decisions_path=args.decisions,
            raw_dir=args.raw_dir,
            overlap_threshold=args.overlap_threshold,
        )
        Path(args.detector_expansion_queue).parent.mkdir(parents=True, exist_ok=True)
        expansion.to_csv(args.detector_expansion_queue, index=False)
        print(f"detector_expansion_queue_rows={len(expansion)}")
        print(f"detector_expansion_queue={args.detector_expansion_queue}")
    elif os.path.exists(args.detector_expansion_queue):
        expansion = pd.read_csv(args.detector_expansion_queue, low_memory=False)
    else:
        expansion = pd.DataFrame()
    if not expansion.empty:
        queue = pd.concat([queue, expansion], ignore_index=True, sort=False)
        queue = queue.drop_duplicates(subset=["review_id"], keep="first")
    evidence_quality = build_evidence_quality_review_queue(
        release_dataset_path=args.transitions,
        assignments_path=args.assignments,
        decisions_path=args.decisions,
        raw_dir=args.raw_dir,
        overlap_threshold=args.overlap_threshold,
    )
    Path(args.evidence_quality_queue).parent.mkdir(parents=True, exist_ok=True)
    evidence_quality.to_csv(args.evidence_quality_queue, index=False)
    print(f"evidence_quality_queue_rows={len(evidence_quality)}")
    print(f"evidence_quality_queue={args.evidence_quality_queue}")
    if not evidence_quality.empty:
        # Evidence review takes precedence over an older structural-only OOF
        # decision for the same source record.
        queue = pd.concat([evidence_quality, queue], ignore_index=True, sort=False)
        queue = queue.drop_duplicates(subset=["review_id"], keep="first")
    reopened_queue = build_detector_reopen_review_queue(
        report_path=args.reopen_report,
        assignments_path=args.assignments,
        decisions_path=args.decisions,
        raw_dir=args.raw_dir,
        overlap_threshold=args.overlap_threshold,
    )
    if not reopened_queue.empty:
        queue = pd.concat([queue, reopened_queue], ignore_index=True, sort=False)
        queue = queue.drop_duplicates(subset=["review_id"], keep="first")
    queue = _refresh_review_status(queue, args.decisions)
    if not queue.empty:
        trigger_order = queue["review_trigger"].map({
            "evidence_quality_ambiguous": 0,
            "stage1_scaffold_oof_abs_error_gt_threshold": 1,
            "detector_schema_change_reopened_decision": 2,
            "detector_v2_new_family_stage1_candidate": 3,
        }).fillna(2)
        terminal_order = queue["review_status"].isin(TERMINAL_DECISIONS).astype(int)
        queue = (
            queue.assign(_terminal_order=terminal_order, _trigger_order=trigger_order)
            .sort_values(
                ["_terminal_order", "_trigger_order", "abs_error", "outlier_rank"],
                ascending=[True, True, False, True],
                na_position="last",
            )
            .drop(columns=["_terminal_order", "_trigger_order"])
            .reset_index(drop=True)
        )
    Path(args.queue).parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(args.queue, index=False)
    if args.web:
        mol_cache = RawMolCache(args.raw_dir)
        for _, row in queue.iterrows():
            image_path = os.path.join(args.image_dir, f"{row['review_id']}.png")
            if not os.path.exists(image_path):
                _render_row(row, mol_cache, args.image_dir)
        serve_web_reviewer(
            queue,
            decisions_path=args.decisions,
            image_dir=args.image_dir,
            host=args.host,
            port=args.port,
        )
        return
    pending = review_outliers(
        queue,
        decisions_path=args.decisions,
        raw_dir=args.raw_dir,
        image_dir=args.image_dir,
        html_path=args.html,
        include_reviewed=bool(args.include_reviewed),
        max_records=args.max_records,
        export_only=bool(args.export_only),
        render_all=bool(args.render_all),
        open_images=not bool(args.no_open),
    )
    status_counts = queue["review_status"].value_counts().to_dict() if not queue.empty else {}
    print(f"queue_rows={len(queue)} pending_rows={pending} status_counts={status_counts}")
    print(f"queue={args.queue}")
    print(f"decisions={args.decisions}")


if __name__ == "__main__":
    main()
