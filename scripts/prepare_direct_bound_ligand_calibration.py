#!/usr/bin/env python3
"""Prepare and gate experimental bound-ligand pKa-shift calibration.

The registry intentionally mixes usable and documented-but-unusable evidence.
This module admits only direct experimental free/bound point labels with an
exact protein/ligand structure and resolved ionization-center atom.  It then
requires enough independent proteins and chemistries for grouped validation.
With the currently curated literature the gate is expected to stay closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from protein_environment_correction import (
    FEATURE_COLUMNS,
    LigandSelector,
    environment_descriptors,
    read_structure_atoms,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = (
    PROJECT_ROOT / "data/curation/direct_bound_ligand_pka_candidates.csv"
)
DEFAULT_STRUCTURES = (
    PROJECT_ROOT
    / "data/raw/external_datasets/protein_environment/direct_bound_ligand"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/processed/ml_models_experimental_only/protein_environment/"
    "direct_bound_ligand"
)
SCHEMA_VERSION = "direct_bound_ligand_calibration_1.0.0"
MINIMUM_ROWS = 12
MINIMUM_PROTEIN_GROUPS = 6
MINIMUM_CHEMISTRIES = 3
REQUIRED_COLUMNS = {
    "case_id", "protein_group", "ligand_name", "free_pka", "bound_pka",
    "bound_qualifier", "delta_pka", "environment_chemistry",
    "transition_semantics", "source_doi", "source_pmid", "label_origin",
    "computational_teacher", "coordinate_pdb", "ligand_chain",
    "ligand_resname", "ligand_residue_id", "site_atom_name",
    "structure_relation", "site_mapping_status", "candidate_for_calibration",
    "exclusion_reason",
}


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def read_registry(path: str | Path = DEFAULT_REGISTRY) -> pd.DataFrame:
    """Read the curated evidence ledger and enforce its provenance contract."""
    table = pd.read_csv(path, keep_default_na=False, dtype={
        "coordinate_pdb": str,
        "ligand_residue_id": str,
        "source_pmid": str,
    })
    missing = sorted(REQUIRED_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(f"Direct bound-ligand registry missing columns: {missing}")
    if table["case_id"].duplicated().any():
        raise ValueError("Direct bound-ligand case IDs must be unique")
    if table["label_origin"].str.lower().ne("experimental").any():
        raise ValueError("Direct bound-ligand labels must be experimental")
    if table["computational_teacher"].map(_clean).ne("").any():
        raise ValueError("Computational-teacher labels cannot calibrate this layer")
    if (
        table["source_doi"].map(_clean).eq("")
        & table["source_pmid"].map(_clean).eq("")
    ).any():
        raise ValueError("Every direct bound-ligand row needs a DOI or PMID")
    for column in ("free_pka", "bound_pka", "delta_pka"):
        table[column] = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(table[column]).all():
            raise ValueError(f"Direct bound-ligand {column} contains nonfinite values")
    expected = table["bound_pka"] - table["free_pka"]
    if not np.allclose(expected, table["delta_pka"], atol=0.011, rtol=0.0):
        raise ValueError("Direct bound-ligand delta_pka must equal bound_pka-free_pka")
    return table


def _candidate_reason(row: Mapping[str, object]) -> str:
    if not _truthy(row["candidate_for_calibration"]):
        return _clean(row["exclusion_reason"]) or "not nominated for calibration"
    if _clean(row["bound_qualifier"]).lower() != "exact":
        return "bound pKa is not an exact point label"
    if _clean(row["structure_relation"]) != "exact_protein_and_ligand":
        return "structure is not an exact protein-and-ligand match"
    if _clean(row["site_mapping_status"]) != "resolved":
        return "ionization-center mapping is unresolved"
    required = (
        "coordinate_pdb", "ligand_chain", "ligand_resname",
        "ligand_residue_id", "site_atom_name",
    )
    missing = [name for name in required if not _clean(row[name])]
    if missing:
        return f"missing coordinate mapping fields: {','.join(missing)}"
    return ""


def build_feature_table(
    registry_path: str | Path = DEFAULT_REGISTRY,
    structures_dir: str | Path = DEFAULT_STRUCTURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map strict candidates to coordinates and project environment features."""
    registry = read_registry(registry_path)
    structures = Path(structures_dir)
    included: list[Dict[str, object]] = []
    excluded: list[Dict[str, object]] = []
    for source_row in registry.to_dict(orient="records"):
        reason = _candidate_reason(source_row)
        if reason:
            excluded.append({
                "case_id": source_row["case_id"],
                "ligand_name": source_row["ligand_name"],
                "reason": reason,
                "evidence_retained_in_registry": True,
            })
            continue
        pdb = _clean(source_row["coordinate_pdb"]).upper()
        candidates = [
            structures / f"{pdb}.pdb",
            structures / f"{pdb}.cif",
            structures / f"{pdb}.cif.gz",
        ]
        structure = next((path for path in candidates if path.is_file()), None)
        if structure is None:
            excluded.append({
                "case_id": source_row["case_id"],
                "ligand_name": source_row["ligand_name"],
                "reason": f"exact structure {pdb} has not been acquired",
                "evidence_retained_in_registry": True,
            })
            continue
        atoms = read_structure_atoms(structure)
        selector = LigandSelector(
            chain=_clean(source_row["ligand_chain"]),
            residue_name=_clean(source_row["ligand_resname"]),
            residue_id=_clean(source_row["ligand_residue_id"]),
        )
        ligand_atoms = [atom for atom in atoms if selector.matches(atom)]
        center_atoms = [
            atom for atom in ligand_atoms
            if atom.atom_name == _clean(source_row["site_atom_name"])
        ]
        if len(center_atoms) != 1:
            excluded.append({
                "case_id": source_row["case_id"],
                "ligand_name": source_row["ligand_name"],
                "reason": (
                    f"expected one mapped site atom but found {len(center_atoms)} "
                    f"in first structure model"
                ),
                "evidence_retained_in_registry": True,
            })
            continue
        descriptors = environment_descriptors(
            atoms,
            center_atoms[0].coordinate,
            chemistry=_clean(source_row["environment_chemistry"]),
            ligand_coordinates=[atom.coordinate for atom in ligand_atoms],
            ligand_selector=selector,
        )
        included.append({
            **source_row,
            **descriptors,
            "structure_path": str(structure),
            "mapped_site_x": center_atoms[0].x,
            "mapped_site_y": center_atoms[0].y,
            "mapped_site_z": center_atoms[0].z,
            "calibration_target": float(source_row["delta_pka"]),
        })
    return pd.DataFrame(included), pd.DataFrame(excluded)


def _new_estimator() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.75,
        random_state=79,
        n_jobs=1,
    )


def _grouped_oof(table: pd.DataFrame) -> np.ndarray:
    groups = table["protein_group"].astype(str).to_numpy()
    unique = np.unique(groups)
    folds = min(5, len(unique))
    if folds < 2:
        raise ValueError("Grouped OOF calibration needs at least two protein groups")
    x = table.loc[:, FEATURE_COLUMNS]
    y = table["calibration_target"].to_numpy(dtype=float)
    result = np.full(len(table), np.nan, dtype=float)
    for train_index, test_index in GroupKFold(folds).split(x, y, groups):
        model = _new_estimator()
        model.fit(x.iloc[train_index], y[train_index])
        result[test_index] = model.predict(x.iloc[test_index])
    if not np.isfinite(result).all():
        raise RuntimeError("Direct bound-ligand OOF prediction is incomplete")
    return result


def prepare_and_train(
    registry_path: str | Path = DEFAULT_REGISTRY,
    structures_dir: str | Path = DEFAULT_STRUCTURES,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> Dict[str, object]:
    """Prepare evidence, apply independence gates, and fit only if validated."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    registry = read_registry(registry_path)
    table, exclusions = build_feature_table(registry_path, structures_dir)
    rows = len(table)
    groups = int(table["protein_group"].nunique()) if rows else 0
    chemistries = int(table["environment_chemistry"].nunique()) if rows else 0
    count_gate = bool(
        rows >= MINIMUM_ROWS
        and groups >= MINIMUM_PROTEIN_GROUPS
        and chemistries >= MINIMUM_CHEMISTRIES
    )
    artifact_path = output / "direct_bound_ligand_model.pkl"
    selection_gate = False
    metrics: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "registry_sha256": _sha256(registry_path),
        "registry_rows": int(len(registry)),
        "strict_structure_mapped_candidate_rows": int(rows),
        "excluded_or_unmapped_rows": int(len(exclusions)),
        "protein_groups": groups,
        "chemistries": chemistries,
        "minimum_rows": MINIMUM_ROWS,
        "minimum_protein_groups": MINIMUM_PROTEIN_GROUPS,
        "minimum_chemistries": MINIMUM_CHEMISTRIES,
        "count_and_diversity_gate_passed": count_gate,
        "grouped_oof_improvement_gate_passed": False,
        "deployed": False,
        "deployed_training_rows": 0,
        "model_path": None,
        "data_boundary": (
            "direct experimental free/bound ligand pKa pairs only; exact point "
            "labels and exact protein-ligand coordinate mappings; no computational "
            "teacher values and no proxy structures"
        ),
        "evaluation_boundary": (
            "calibration candidates are never scored as the locked external "
            "binding-shift or metalloprotein benchmarks"
        ),
    }
    if count_gate:
        target = table["calibration_target"].to_numpy(dtype=float)
        oof = _grouped_oof(table)
        table["oof_predicted_delta_pka"] = oof
        model_mae = float(mean_absolute_error(target, oof))
        zero_mae = float(mean_absolute_error(target, np.zeros(len(target))))
        group_errors = table.assign(_absolute=np.abs(target - oof)).groupby(
            "protein_group"
        )["_absolute"].mean()
        zero_group_errors = table.assign(_absolute=np.abs(target)).groupby(
            "protein_group"
        )["_absolute"].mean()
        model_group_mae = float(group_errors.mean())
        zero_group_mae = float(zero_group_errors.mean())
        selection_gate = bool(
            model_mae < zero_mae and model_group_mae < zero_group_mae
        )
        metrics.update({
            "grouped_oof_mae": model_mae,
            "zero_shift_mae": zero_mae,
            "group_balanced_oof_mae": model_group_mae,
            "group_balanced_zero_shift_mae": zero_group_mae,
            "grouped_oof_improvement_gate_passed": selection_gate,
        })
        if selection_gate:
            estimator = _new_estimator()
            estimator.fit(table.loc[:, FEATURE_COLUMNS], target)
            artifact = {
                "schema_version": SCHEMA_VERSION,
                "feature_columns": list(FEATURE_COLUMNS),
                "model": estimator,
                "training_rows": int(rows),
                "protein_groups": groups,
                "chemistries": chemistries,
                "oof_p90_abs_error": float(np.quantile(np.abs(target - oof), 0.90)),
                "prediction_clip": [
                    float(np.quantile(target, 0.01)),
                    float(np.quantile(target, 0.99)),
                ],
                "label_semantics": "bound_pka_minus_free_aqueous_pka",
            }
            with artifact_path.open("wb") as handle:
                pickle.dump(artifact, handle)
            metrics.update({
                "deployed": True,
                "deployed_training_rows": int(rows),
                "model_path": str(artifact_path),
            })
    if not selection_gate and artifact_path.exists():
        artifact_path.unlink()
    table.to_csv(output / "direct_bound_ligand_features.csv", index=False)
    exclusions.to_csv(output / "direct_bound_ligand_exclusions.csv", index=False)
    (output / "calibration_gate.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--structures", default=str(DEFAULT_STRUCTURES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(prepare_and_train(
        args.registry, args.structures, args.output
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
