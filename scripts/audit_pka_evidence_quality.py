#!/usr/bin/env python3
"""Audit canonical pKa evidence tiers, chemical blockers, and copied lineages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore


DEFAULT_TRANSITIONS = "data/processed/pka_microstate_dataset.csv"
DEFAULT_NETWORK = "data/processed/pka_molecule_microstate_network_stage3_predictions.csv"
DEFAULT_STAGE1_TRAINING = (
    "data/processed/ml_models_experimental_only/stage1_intrinsic/stage1_training_sites.csv"
)
DEFAULT_OUT_DIR = "data/processed/dataset_audit/evidence_quality"
DEFAULT_STAGE2_WEAK = (
    "data/processed/ml_models_experimental_only/stage2_network_context/"
    "weak_molecule_training_rows.csv"
)
DEFAULT_STAGE2_BLOCKED = (
    "data/processed/ml_models_experimental_only/stage2_network_context/"
    "weak_molecule_blocked.csv"
)


def _canonical_smiles(value: object) -> str:
    mol = Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else str(value)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_counts(frame: pd.DataFrame, column: str) -> Dict[str, int]:
    if column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("missing").astype(str).value_counts().items()
    }


def _truthy(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _stage3_site_confidence_counts(network: pd.DataFrame) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if "sites_json" not in network:
        return counts
    for raw in network["sites_json"]:
        try:
            sites = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for site in sites:
            tier = str(site.get("stage3_overall_site_state_confidence_tier", "missing"))
            counts[tier] = counts.get(tier, 0) + 1
    return counts


def _stage3_calibration_summary(network: pd.DataFrame) -> Dict:
    confidences = []
    sources: Dict[str, int] = {}
    scopes: Dict[str, int] = {}
    if "sites_json" not in network:
        return {"available_site_calls": 0, "sources": {}, "scopes": {}}
    for raw in network["sites_json"]:
        try:
            sites = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        for site in sites:
            confidence = site.get("stage3_empirical_pka_side_call_confidence")
            if confidence is not None:
                confidences.append(float(confidence))
            source = site.get("stage3_empirical_calibration_source")
            scope = site.get("stage3_empirical_calibration_scope")
            if source:
                sources[str(source)] = sources.get(str(source), 0) + 1
            if scope:
                scopes[str(scope)] = scopes.get(str(scope), 0) + 1
    series = pd.Series(confidences, dtype=float)
    return {
        "available_site_calls": int(len(series)),
        "minimum": float(series.min()) if len(series) else None,
        "median": float(series.median()) if len(series) else None,
        "maximum": float(series.max()) if len(series) else None,
        "sources": sources,
        "scopes": scopes,
        "semantics": (
            "scaffold-held-out scalar-pKa-side proxy, not direct experimental "
            "microstate-state validation"
        ),
    }


def _lineage_duplicates(transitions: pd.DataFrame) -> pd.DataFrame:
    work = transitions.copy()
    work["canonical_input_smiles"] = work["input_smiles"].map(_canonical_smiles)
    document = work.get("source_document_id", pd.Series("", index=work.index)).fillna("").astype(str)
    assay = work.get("source_assay_id", pd.Series("", index=work.index)).fillna("").astype(str)
    value = pd.to_numeric(work["experimental_pka"], errors="coerce").round(6).astype(str)
    label = work.get("site_label", pd.Series("", index=work.index)).fillna("").astype(str)
    work["lineage_key"] = [
        f"document::{doc}::{assay_id}::{pka}::{site_label}"
        if doc and doc.lower() != "nan"
        else f"citation_unknown::{smiles}::{pka}::{site_label}"
        for doc, assay_id, pka, site_label, smiles in zip(
            document, assay, value, label, work["canonical_input_smiles"]
        )
    ]
    rows: List[Dict] = []
    for lineage_key, part in work.groupby("lineage_key", sort=False):
        if len(part) < 2:
            continue
        rows.append({
            "lineage_key": lineage_key,
            "row_count": int(len(part)),
            "independent_lineage_count": 1,
            "experimental_pka_values_json": json.dumps(sorted(set(
                pd.to_numeric(part["experimental_pka"], errors="coerce").dropna().tolist()
            ))),
            "source_files_json": json.dumps(sorted(set(part["source_file"].astype(str)))),
            "record_indices_json": json.dumps(sorted(set(
                pd.to_numeric(part["record_index"], errors="coerce").dropna().astype(int).tolist()
            ))),
            "dataset_row_ids_json": json.dumps(sorted(set(part["dataset_row_id"].astype(str)))),
        })
    return pd.DataFrame(rows).sort_values(
        "row_count", ascending=False
    ).reset_index(drop=True) if rows else pd.DataFrame()


def run_audit(
    transition_path: str,
    network_path: str,
    stage1_training_path: str,
    out_dir: str,
    stage2_weak_path: str = DEFAULT_STAGE2_WEAK,
    stage2_blocked_path: str = DEFAULT_STAGE2_BLOCKED,
) -> Dict:
    transitions = pd.read_csv(transition_path, low_memory=False)
    network = pd.read_csv(network_path, low_memory=False) if os.path.exists(network_path) else pd.DataFrame()
    stage1 = (
        pd.read_csv(stage1_training_path, low_memory=False)
        if os.path.exists(stage1_training_path) else pd.DataFrame()
    )
    stage2_weak = (
        pd.read_csv(stage2_weak_path, low_memory=False)
        if os.path.exists(stage2_weak_path) else pd.DataFrame()
    )
    stage2_blocked = (
        pd.read_csv(stage2_blocked_path, low_memory=False)
        if os.path.exists(stage2_blocked_path) else pd.DataFrame()
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    tier_by_label = (
        transitions.groupby(["site_family", "site_label", "evidence_tier"], dropna=False)
        .size().rename("row_count").reset_index()
        .sort_values(["site_family", "site_label", "evidence_tier"])
    )
    conflicts = transitions[
        transitions.get("reference_plausibility_status", pd.Series("", index=transitions.index))
        .astype(str).isin({"conflict", "review"})
    ].copy()
    conflict_columns = [column for column in [
        "dataset_row_id", "source_file", "record_index", "input_smiles",
        "source_original_smiles", "experimental_pka", "site_label", "site_family",
        "evidence_tier", "evidence_tier_reason", "reference_plausibility_status",
        "reference_pka", "reference_uncertainty", "reference_z_distance",
        "reference_transition_definition", "unresolved_ionizable_contexts_json",
    ] if column in conflicts]
    conflicts = conflicts[conflict_columns].sort_values(
        "reference_z_distance", ascending=False, na_position="last"
    )
    duplicates = _lineage_duplicates(transitions)

    tier_by_label.to_csv(os.path.join(out_dir, "evidence_tier_by_site_label.csv"), index=False)
    conflicts.to_csv(os.path.join(out_dir, "reference_plausibility_review.csv"), index=False)
    duplicates.to_csv(os.path.join(out_dir, "source_lineage_duplicate_groups.csv"), index=False)

    ambiguous = transitions[transitions["evidence_tier"].astype(str).eq("ambiguous")]
    summary = {
        "audit_schema_version": "1.0.0",
        "interpretation": (
            "Evidence-policy and internal-reference audit. Reference conflicts are review "
            "signals, not proof that an experimental value is wrong."
        ),
        "transition_dataset": {
            "path": transition_path,
            "sha256": _sha256(transition_path),
            "rows": int(len(transitions)),
            "evidence_tiers": _value_counts(transitions, "evidence_tier"),
            "reference_plausibility": _value_counts(
                transitions, "reference_plausibility_status"
            ),
            "transition_identity_confidence": _value_counts(
                transitions, "transition_identity_confidence"
            ),
            "experimental_site_attribution_confidence": _value_counts(
                transitions, "experimental_site_attribution_confidence"
            ),
            "measurement_conditions_confidence": _value_counts(
                transitions, "measurement_conditions_confidence"
            ),
            "unresolved_context_rows": int(
                (pd.to_numeric(
                    transitions.get("unresolved_ionizable_context_count", 0), errors="coerce"
                ).fillna(0) > 0).sum()
            ),
            "source_structure_charge_changed_rows": int(
                transitions.get("source_structure_charge_changed", pd.Series(False, index=transitions.index))
                .map(_truthy).sum()
            ),
            "ambiguous_rows_retained_as_weak_labels": int(len(ambiguous)),
            "stage1_exact_eligible_rows": int(
                transitions.get("stage1_local_supervision_eligible", pd.Series(False, index=transitions.index))
                .map(_truthy).sum()
            ),
        },
        "source_lineage": {
            "duplicate_groups": int(len(duplicates)),
            "rows_in_duplicate_groups": int(duplicates.get("row_count", pd.Series(dtype=int)).sum()),
            "policy": "copied values count as one lineage for Stage 1 weighting",
        },
        "stage1_training": {
            "path": stage1_training_path if not stage1.empty else "not_available",
            "rows": int(len(stage1)),
            "source_measurement_rows": int(
                pd.to_numeric(
                    stage1.get("experimental_measurement_count", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "independent_measurement_lineages": int(
                pd.to_numeric(
                    stage1.get("experimental_independent_lineage_count", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0).sum()
            ),
            "evidence_tiers": _value_counts(stage1, "experimental_evidence_tier"),
            "site_labels": int(stage1["candidate_label"].nunique()) if "candidate_label" in stage1 else 0,
        },
        "stage3_network": {
            "path": network_path if not network.empty else "not_available",
            "molecules": int(len(network)),
            "overall_site_state_confidence_tiers": _stage3_site_confidence_counts(network),
            "empirical_pka_side_calibration": _stage3_calibration_summary(network),
        },
        "stage2_weak_supervision": {
            "candidate_rows": int(len(stage2_weak)),
            "independent_weak_measurements_used": int(
                stage2_weak["weak_measurement_id"].nunique()
                if "weak_measurement_id" in stage2_weak else 0
            ),
            "molecules_used": int(
                stage2_weak["molecule_id"].nunique()
                if "molecule_id" in stage2_weak else 0
            ),
            "blocked_measurements": int(len(stage2_blocked)),
            "blocked_molecules": int(
                stage2_blocked["molecule_id"].nunique()
                if "molecule_id" in stage2_blocked else 0
            ),
            "blocked_reasons": _value_counts(stage2_blocked, "blocked_reason"),
            "interpretation": (
                "Weak labels are deployment-only; exact-site validation excludes them."
            ),
        },
        "outputs": {
            "tier_by_label": os.path.join(out_dir, "evidence_tier_by_site_label.csv"),
            "plausibility_review": os.path.join(out_dir, "reference_plausibility_review.csv"),
            "lineage_duplicates": os.path.join(out_dir, "source_lineage_duplicate_groups.csv"),
        },
    }
    with open(os.path.join(out_dir, "evidence_quality_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", default=DEFAULT_TRANSITIONS)
    parser.add_argument("--network", default=DEFAULT_NETWORK)
    parser.add_argument("--stage1-training", default=DEFAULT_STAGE1_TRAINING)
    parser.add_argument("--stage2-weak", default=DEFAULT_STAGE2_WEAK)
    parser.add_argument("--stage2-blocked", default=DEFAULT_STAGE2_BLOCKED)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(
        args.transitions, args.network, args.stage1_training, args.out_dir,
        args.stage2_weak, args.stage2_blocked,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
