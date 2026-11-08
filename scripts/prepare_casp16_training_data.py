#!/usr/bin/env python3
"""Prepare experimental CASP16 ligand pKa values for pKate training.

The CASP16 files associate ligands with pharmaceutical protein targets and, for
some compounds, crystal structures.  They do not state that the pKa values were
measured in the bound state.  This module therefore treats the experimental
``PKA1``/``PKA2`` values as aqueous ligand evidence.  ADMET Predictor columns
are explicitly ignored.  A value becomes a Stage 1 candidate only when its
reported ACID/BASE class uniquely selects one supported structural transition;
otherwise a complete multi-site molecule may contribute only an unordered,
weak Stage 2 macro-pKa set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem

from functional_group_pka_analysis import ACIDIC_FAMILIES
from microstate_enumerator import ensure_heavy_atom_maps, mapped_smiles
from pka_evidence_policy import EVIDENCE_POLICY_SCHEMA_VERSION, assess_measurement_evidence
from prepare_pkahub_training_data import _existing_keys, _resolved_transitions, molecule_group_key
from stage1_canonical_data import (
    STAGE1_TRAINING_SCHEMA_VERSION,
    TARGET_DEFINITION,
    canonical_structure_key,
    scaffold_group_key,
)
from substructure_match import FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "data/raw/external_datasets/casp16/chymasecatgSI.txt"
)
DEFAULT_EXISTING_STAGE1 = (
    PROJECT_ROOT
    / "data/processed/ml_models_experimental_only/stage1_intrinsic/"
    "stage1_training_sites.csv"
)
DEFAULT_EXISTING_NETWORK = (
    PROJECT_ROOT / "data/processed/pka_molecule_microstate_network_dataset.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/casp16"
SOURCE_DOI = "10.5281/zenodo.16762332"
TRAINING_COHORT = "casp16_experimental_single_transition"
PKA_RANGE = (-5.0, 20.0)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _measurements(row: pd.Series) -> tuple[list[Dict[str, object]], list[Dict[str, object]]]:
    accepted: list[Dict[str, object]] = []
    rejected: list[Dict[str, object]] = []
    for slot in (1, 2):
        value_text = str(row.get(f"PKA{slot}", "")).strip()
        class_text = str(row.get(f"PKA{slot}_CLASS", "")).strip().lower()
        if not value_text or value_text.lower() in {"nan", "nd", "none"}:
            continue
        try:
            value = float(value_text)
        except ValueError:
            rejected.append({
                "slot": slot,
                "raw_value": value_text,
                "reported_class": class_text,
                "reason": "non_exact_or_non_numeric_experimental_pka",
            })
            continue
        if not np.isfinite(value) or not PKA_RANGE[0] <= value <= PKA_RANGE[1]:
            rejected.append({
                "slot": slot,
                "raw_value": value_text,
                "reported_class": class_text,
                "reason": "experimental_pka_outside_supported_range",
            })
            continue
        if class_text not in {"acid", "base"}:
            rejected.append({
                "slot": slot,
                "raw_value": value_text,
                "reported_class": class_text,
                "reason": "missing_or_unknown_acid_base_class",
            })
            continue
        accepted.append({"slot": slot, "pka": value, "reported_class": class_text})
    return accepted, rejected


def _stage1_row(
    source: pd.Series,
    canonical_smiles: str,
    transition: Dict[str, object],
    measurement: Dict[str, object],
) -> Dict[str, object]:
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError("invalid canonical CASP16 SMILES")
    mapped = ensure_heavy_atom_maps(mol)
    center = sorted(transition.get("center_atom_set", set()))
    if not center:
        center = sorted(transition.get("atom_set", set()))
    atom_index = int(center[0]) if center else np.nan
    pka = float(measurement["pka"])
    family = str(transition["pair_family"])
    label = str(transition["label"])
    evidence = assess_measurement_evidence(
        pka=pka,
        label=label,
        family=family,
        structural_mapping_confidence="high",
        source_site_evidence_confidence="medium",
        unresolved_contexts=[],
        source_original_smiles=str(source["structure"]),
        input_smiles=canonical_smiles,
    )
    key = molecule_group_key(canonical_smiles)
    compound_id = str(source["compound id"])
    site_id = "casp16_" + hashlib.sha256(
        f"{compound_id}|{label}|{measurement['slot']}".encode()
    ).hexdigest()[:16]
    source_record = f"CASP16_chymasecatgSI_compound_{compound_id}_PKA{measurement['slot']}"
    weight = 0.50 if evidence["evidence_tier"] in {"gold", "silver"} else 0.25
    return {
        "stage1_training_schema_version": STAGE1_TRAINING_SCHEMA_VERSION,
        "evidence_policy_schema_version": EVIDENCE_POLICY_SCHEMA_VERSION,
        "functional_group_detector_schema_versions_json": _json([
            FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION
        ]),
        "target_definition": TARGET_DEFINITION,
        "molecule_key": key,
        "molecule_id": "casp16_" + hashlib.sha256(key.encode()).hexdigest()[:16],
        "structure_key": canonical_structure_key(canonical_smiles),
        "scaffold_group": scaffold_group_key(canonical_smiles),
        "smiles": canonical_smiles,
        "mapped_smiles": mapped_smiles(mapped),
        "site_id": site_id,
        "candidate_label": label,
        "final_group_label": label,
        "site_family": family,
        "pka_type_canonical": (
            "acidic" if measurement["reported_class"] == "acid" else "basic"
        ),
        "atom_index_raw": atom_index,
        "pka_value": pka,
        "experimental_anchor_pka": pka,
        "experimental_measurement_count": 1,
        "experimental_eligible_measurement_count": 1,
        "experimental_independent_lineage_count": 1,
        "experimental_measurement_mean": pka,
        "experimental_measurement_std": 0.0,
        "experimental_measurement_mad": 0.0,
        "experimental_measurement_robust_sigma": 0.0,
        "experimental_measurement_min": pka,
        "experimental_measurement_max": pka,
        "experimental_measurement_range": 0.0,
        "experimental_medium_confidence_fraction": 1.0,
        "experimental_evidence_tier": str(evidence["evidence_tier"]),
        "stage1_reference_pka": evidence["reference_pka"],
        "stage1_reference_uncertainty": evidence["reference_uncertainty"],
        "stage1_reference_transition_definition": evidence[
            "reference_transition_definition"
        ],
        "stage1_reference_provenance": evidence["reference_provenance"],
        "experimental_source_site_evidence_json": _json([
            "reported_acid_base_class_uniquely_matches_supported_transition"
        ]),
        "experimental_source_site_evidence_confidences_json": _json(["medium"]),
        "sample_weight_raw": weight,
        "sample_weight": weight,
        "measurement_method": "experimental",
        "target_aggregation": "single_reported_value",
        "experimental_transition_ids_json": _json([source_record]),
        "experimental_source_files_json": _json([
            "CASP16:chymasecatgSI.txt"
        ]),
        "experimental_record_indices_json": _json([int(source.name) + 2]),
        "training_cohort": TRAINING_COHORT,
        "assignment_teacher_used": False,
        "computational_pka_values_used": False,
        "computed_pka_columns_ignored": True,
        "casp16_compound_id": compound_id,
        "casp16_associated_pdb": str(source.get("pdb", "")),
        "casp16_series": str(source.get("series", "")),
        "source_document_id": SOURCE_DOI,
        "pka_environment_semantics": "aqueous_or_assay_ligand_pka_not_bound_state",
    }


def _macro_placeholders(macro: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, object]] = []
    for record_index, row in enumerate(macro.itertuples(index=False)):
        mol, _, transitions = _resolved_transitions(str(row.smiles))
        if not transitions:
            continue
        carrier = sorted(
            transitions,
            key=lambda item: (
                str(item.get("label", "")), tuple(sorted(item.get("atom_set", set())))
            ),
        )[0]
        mapped = ensure_heavy_atom_maps(mol)
        atom_maps = [
            int(mapped.GetAtomWithIdx(index).GetAtomMapNum())
            for index in sorted(carrier.get("atom_set", set()))
        ]
        values = json.loads(str(row.experimental_pka_values_json))
        macro_set_id = "casp16_macro_" + hashlib.sha256(
            str(row.molecule_group_key).encode()
        ).hexdigest()[:16]
        for value_index, value in enumerate(values):
            rows.append({
                "dataset_schema_version": "casp16_macro_placeholder_1.0.0",
                "functional_group_detector_schema_version": (
                    FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION
                ),
                "dataset_row_id": f"{macro_set_id}_{value_index}",
                "source_file": "CASP16:chymasecatgSI.txt",
                "source_dataset": "casp16",
                "record_index": int(record_index),
                "experimental_pka": float(value),
                "measurement_method": "experimental",
                "measurement_provenance": "casp16_experimental_pka_scalar_only",
                "source_assay_id": f"CASP16_chymase_compound_{row.casp16_compound_id}",
                "source_document_id": SOURCE_DOI,
                "source_molecule_id": str(row.casp16_compound_id),
                "source_original_smiles": str(row.smiles),
                "input_smiles": str(row.smiles),
                "input_atom_mapped_smiles": mapped_smiles(mapped),
                "structure_provenance": (
                    "associated_protein_complex_pdb_not_bound_pka_label:"
                    f"{row.casp16_associated_pdb}"
                ),
                "site_label": str(carrier["label"]),
                "site_family": str(carrier["pair_family"]),
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
                "computational_pka_values_used": False,
                "computed_pka_columns_ignored": True,
                "marvin_values_used": False,
                "epik_values_used": False,
                "admet_predictor_values_used": False,
            })
    return pd.DataFrame(rows)


def prepare(
    input_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    existing_stage1_path: str | Path = DEFAULT_EXISTING_STAGE1,
    existing_network_path: str | Path = DEFAULT_EXISTING_NETWORK,
) -> Dict[str, object]:
    """Prepare site-resolved and weak CASP16 cohorts with full quarantine ledgers."""
    source_path = Path(input_path)
    frame = pd.read_csv(source_path, sep=";", low_memory=False)
    required = {
        "compound id", "structure", "pdb", "series",
        "PKA1", "PKA1_CLASS", "PKA2", "PKA2_CLASS",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CASP16 table missing columns: {missing}")
    predicted_columns = sorted(
        column for column in frame.columns if "ADMET Predictor" in str(column)
    )
    existing = _existing_keys(
        existing_stage1_path,
        existing_network_path,
        excluded_training_cohorts=(TRAINING_COHORT,),
        excluded_source_prefixes=("CASP16:",),
    )
    accepted: List[Dict[str, object]] = []
    quarantined: List[Dict[str, object]] = []
    macro_rows: List[Dict[str, object]] = []
    overlap_rows: List[Dict[str, object]] = []
    numeric_molecules = 0
    numeric_values = 0
    associated_complexes = 0

    for source_index, row in frame.iterrows():
        measurements, rejected_measurements = _measurements(row)
        compound_id = str(row["compound id"])
        for rejected in rejected_measurements:
            quarantined.append({
                "casp16_compound_id": compound_id,
                "source_row": int(source_index) + 2,
                **rejected,
            })
        if not measurements:
            continue
        numeric_molecules += 1
        numeric_values += len(measurements)
        associated_pdb = "" if pd.isna(row.get("pdb")) else str(row.get("pdb", ""))
        associated_complexes += bool(associated_pdb)
        mol = Chem.MolFromSmiles(str(row["structure"]))
        if mol is None:
            quarantined.append({
                "casp16_compound_id": compound_id,
                "source_row": int(source_index) + 2,
                "reason": "invalid_smiles",
            })
            continue
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        key = molecule_group_key(smiles)
        if key in existing:
            overlap_rows.append({
                "casp16_compound_id": compound_id,
                "molecule_group_key": key,
                "smiles": smiles,
                "reason": "existing_canonical_or_external_molecule_identity",
            })
            continue
        try:
            _, _, transitions = _resolved_transitions(smiles)
        except ValueError as exc:
            quarantined.append({
                "casp16_compound_id": compound_id,
                "source_row": int(source_index) + 2,
                "smiles": smiles,
                "reason": str(exc),
            })
            continue
        acidic = [
            transition for transition in transitions
            if str(transition["pair_family"]) in ACIDIC_FAMILIES
        ]
        basic = [
            transition for transition in transitions
            if str(transition["pair_family"]) not in ACIDIC_FAMILIES
        ]
        for measurement in measurements:
            matches = acidic if measurement["reported_class"] == "acid" else basic
            if len(matches) == 1:
                accepted.append(_stage1_row(row, smiles, matches[0], measurement))
            else:
                quarantined.append({
                    "casp16_compound_id": compound_id,
                    "source_row": int(source_index) + 2,
                    "smiles": smiles,
                    "slot": int(measurement["slot"]),
                    "pka": float(measurement["pka"]),
                    "reported_class": str(measurement["reported_class"]),
                    "matching_supported_transition_count": int(len(matches)),
                    "reason": (
                        "no_supported_transition_for_reported_class"
                        if not matches else "reported_class_does_not_resolve_site"
                    ),
                })
        class_coverage = all(
            (measurement["reported_class"] == "acid" and acidic)
            or (measurement["reported_class"] == "base" and basic)
            for measurement in measurements
        )
        if len(transitions) > 1 and len(measurements) <= len(transitions) and class_coverage:
            macro_rows.append({
                "molecule_group_key": key,
                "smiles": smiles,
                "candidate_transition_count": int(len(transitions)),
                "experimental_pka_values_json": _json(sorted({
                    float(measurement["pka"]) for measurement in measurements
                })),
                "reported_classes_json": _json(sorted({
                    str(measurement["reported_class"]) for measurement in measurements
                })),
                "measurement_count": int(len(measurements)),
                "casp16_compound_id": compound_id,
                "casp16_associated_pdb": associated_pdb,
                "measurement_method": "experimental",
                "measurement_scope": "unordered_molecule_macroscopic_pka_set",
                "assignment_teacher_used": False,
                "computational_pka_values_used": False,
                "computed_pka_columns_ignored": True,
            })

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    accepted_frame = pd.DataFrame(accepted)
    quarantine_frame = pd.DataFrame(quarantined)
    macro_frame = pd.DataFrame(macro_rows)
    placeholders = _macro_placeholders(macro_frame)
    pd.DataFrame(overlap_rows).to_csv(output / "existing_molecule_overlap.csv", index=False)
    accepted_frame.to_csv(output / "stage1_supplemental_sites.csv", index=False)
    quarantine_frame.to_csv(output / "stage1_supplemental_quarantine.csv", index=False)
    macro_frame.to_csv(output / "macro_set_candidates.csv", index=False)
    placeholders.to_csv(output / "macro_set_transition_placeholders.csv", index=False)
    summary = {
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_doi": SOURCE_DOI,
        "input_molecules": int(len(frame)),
        "molecules_with_numeric_experimental_pka": int(numeric_molecules),
        "numeric_experimental_pka_values": int(numeric_values),
        "numeric_molecules_with_associated_complex_pdb": int(associated_complexes),
        "stage1_candidate_rows": int(len(accepted_frame)),
        "stage1_candidate_molecules": int(
            accepted_frame["molecule_id"].nunique() if not accepted_frame.empty else 0
        ),
        "macro_set_candidate_molecules": int(len(macro_frame)),
        "macro_set_placeholder_rows": int(len(placeholders)),
        "quarantine_rows": int(len(quarantine_frame)),
        "existing_molecule_overlap_rows": int(len(overlap_rows)),
        "predicted_columns_ignored": predicted_columns,
        "pka_environment_semantics": "aqueous_or_assay_ligand_pka_not_bound_state",
        "policy": (
            "Use only numeric experimental PKA1/PKA2 values; require ACID/BASE "
            "class to uniquely select a supported Stage 1 transition; preserve "
            "multi-site values only as weak unordered macro sets; never consume "
            "ADMET Predictor pKa or charge fields"
        ),
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--existing-stage1", default=str(DEFAULT_EXISTING_STAGE1))
    parser.add_argument("--existing-network", default=str(DEFAULT_EXISTING_NETWORK))
    args = parser.parse_args(argv)
    print(json.dumps(prepare(
        args.input, args.output_dir, args.existing_stage1, args.existing_network
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
