#!/usr/bin/env python3
"""Train the optional protein-environment residual correction layer.

The target is an experimental protein-residue pKa minus a documented aqueous
model-compound reference.  Validation groups are connected components of the
PKAD-R 30%-sequence-identity links, preventing close homologues from appearing
on both sides of a fold.  The fitted model is a deliberately regularized,
fixed-hyperparameter random forest over a small auditable descriptor set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold

from protein_environment_correction import (
    DEFAULT_ENVIRONMENT_MODEL,
    FEATURE_COLUMNS,
    PROTEIN_ENVIRONMENT_FEATURE_SCHEMA_VERSION,
    PROTEIN_ENVIRONMENT_METHOD,
    PROTEIN_ENVIRONMENT_SCHEMA_VERSION,
    environment_descriptors,
    read_structure_atoms,
    residue_site_center,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "data/raw/external_datasets/protein_environment/pkadr/PKAD-R-250211.json"
)
DEFAULT_STRUCTURES = DEFAULT_SOURCE.parent / "structures"
DEFAULT_OUTPUT = PROJECT_ROOT / DEFAULT_ENVIRONMENT_MODEL
DEFAULT_PKAD3_CANDIDATES = (
    PROJECT_ROOT
    / "data/processed/protein_environment/pkad3/training_candidates.csv"
)
DEFAULT_PKAD3_STRUCTURES = (
    PROJECT_ROOT
    / "data/raw/external_datasets/protein_environment/kamls_pkad3/structures"
)
DEFAULT_DIRECT_LIGAND_GATE = (
    PROJECT_ROOT
    / "data/processed/ml_models_experimental_only/protein_environment/"
    "direct_bound_ligand/calibration_gate.json"
)
DEFAULT_DIRECT_LIGAND_MODEL = DEFAULT_DIRECT_LIGAND_GATE.parent / (
    "direct_bound_ligand_model.pkl"
)
DIRECT_LIGAND_SCHEMA_VERSION = "direct_bound_ligand_calibration_1.0.0"

# Conventional aqueous model-compound/reference side-chain values.  Their role
# is only to define a residual; predictions retain the pKate ligand aqueous pKa.
REFERENCE_PKA = {
    "ASP": 3.90,
    "GLU": 4.30,
    "HIS": 6.50,
    "LYS": 10.50,
    "CYS": 8.30,
    "TYR": 10.10,
}
RESIDUE_CHEMISTRY = {
    "ASP": "carboxyl",
    "GLU": "carboxyl",
    "HIS": "imidazole",
    "LYS": "amine",
    "CYS": "thiol",
    "TYR": "phenol",
}
ENVIRONMENT_CALIBRATION_SCHEMA_VERSION = "1.0.0"
MAX_MAIN_OOF_MAE_DEGRADATION = 0.01
MAX_PKAD3_WEIGHT_FRACTION = 0.25


def _direct_ligand_calibration(
    gate_path: str | Path | None,
    model_path: str | Path | None,
) -> tuple[Dict[str, object], object | None]:
    """Load an optional direct-ligand model only after its evidence gate passed."""
    unavailable: Dict[str, object] = {
        "schema_version": DIRECT_LIGAND_SCHEMA_VERSION,
        "status": "calibration_gate_artifact_unavailable",
        "registry_rows": 0,
        "strict_structure_mapped_candidate_rows": 0,
        "deployed": False,
        "deployed_training_rows": 0,
    }
    if gate_path is None or not Path(gate_path).is_file():
        return unavailable, None
    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    if str(gate.get("schema_version")) != DIRECT_LIGAND_SCHEMA_VERSION:
        raise ValueError("Direct bound-ligand calibration gate schema is incompatible")
    deployed = bool(gate.get("deployed", False))
    deployed_rows = int(gate.get("deployed_training_rows", 0))
    if deployed != (deployed_rows > 0):
        raise ValueError("Direct bound-ligand deployment status and row count disagree")
    if not deployed:
        return gate, None
    if model_path is None or not Path(model_path).is_file():
        raise FileNotFoundError(
            "Direct bound-ligand gate passed but its model artifact is missing"
        )
    with Path(model_path).open("rb") as handle:
        artifact = pickle.load(handle)
    if str(artifact.get("schema_version")) != DIRECT_LIGAND_SCHEMA_VERSION:
        raise ValueError("Direct bound-ligand model schema is incompatible")
    if list(artifact.get("feature_columns", [])) != list(FEATURE_COLUMNS):
        raise ValueError("Direct bound-ligand model feature columns are incompatible")
    if int(artifact.get("training_rows", -1)) != deployed_rows:
        raise ValueError("Direct bound-ligand model and gate training rows disagree")
    return gate, artifact


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def homology_groups(rows: Iterable[Mapping[str, object]]) -> Dict[str, str]:
    """Return leakage-safe connected groups for each ``PDB.chain`` context.

    Besides PKAD-R's explicit >30%-identity links, every serialized
    alternative is joined to its preceding main context.  Mutant and alternate
    state structures are therefore unable to cross validation folds even when
    the source omits a redundant sequence-identity link.
    """
    rows = list(rows)
    union = _UnionFind()
    main_key: Optional[str] = None
    for row in rows:
        key = f"{str(row['PDB']).upper()}.{row['Chain']}"
        union.find(key)
        if str(row.get("pKa_Classification", "")) == "Main":
            main_key = key
        elif main_key is not None:
            union.union(main_key, key)
        linked = str(row.get("Sequence_Identity_>_30%", "")).strip()
        for token in linked.replace(";", ",").split(","):
            token = token.strip().upper()
            if token:
                union.union(key, token)
    members: Dict[str, list[str]] = defaultdict(list)
    for key in union.parent:
        members[union.find(key)].append(key)
    canonical = {
        member: min(component)
        for component in members.values()
        for member in component
    }
    return canonical


def _strict_rows(rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    selected = []
    for row in rows:
        residue = str(row.get("ResName", "")).upper()
        try:
            pka = float(row.get("Expt_pKa", ""))
        except (TypeError, ValueError):
            continue
        if (
            row.get("pKa_Classification") == "Main"
            and not str(row.get("Warning", "")).strip()
            and residue in REFERENCE_PKA
            and np.isfinite(pka)
        ):
            selected.append(row)
    return selected


def build_training_table(
    source_path: str | Path,
    structures_dir: str | Path,
) -> tuple[pd.DataFrame, list[Dict[str, object]]]:
    """Extract strict PKAD-R labels and deterministic local descriptors."""
    source = Path(source_path)
    structures = Path(structures_dir)
    rows = json.loads(source.read_text(encoding="utf-8"))
    strict = _strict_rows(rows)
    groups = homology_groups(rows)
    structure_cache: Dict[str, object] = {}
    records, exclusions = [], []
    for row in strict:
        pdb = str(row["PDB"]).upper()
        structure = structures / f"{pdb}.cif.gz"
        try:
            if pdb not in structure_cache:
                structure_cache[pdb] = read_structure_atoms(structure)
            atoms = structure_cache[pdb]
            center, residue_key = residue_site_center(
                atoms,
                chain=str(row["Chain"]),
                residue_id=str(row["ResID_in_PDB"]),
                residue_name=str(row["ResName"]),
            )
            chemistry = RESIDUE_CHEMISTRY[str(row["ResName"]).upper()]
            features = environment_descriptors(
                atoms,
                center,
                chemistry=chemistry,
                excluded_residue_keys=[residue_key],
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            residue = str(row.get("ResName", "")).upper()
            experimental = float(row["Expt_pKa"])
            key = f"{pdb}.{row.get('Chain', '')}"
            exclusions.append({
                "pkadr_index": str(row.get("Index", "")),
                "pdb": pdb,
                "chain": str(row.get("Chain", "")),
                "residue_id": str(row.get("ResID_in_PDB", "")),
                "residue_name": str(row.get("ResName", "")),
                "chemistry": RESIDUE_CHEMISTRY.get(residue),
                "homology_group": groups.get(key, key),
                "experimental_pka": experimental,
                "reference_pka": REFERENCE_PKA.get(residue),
                "delta_pka": experimental - REFERENCE_PKA[residue],
                "reason": str(exc),
            })
            continue
        residue = str(row["ResName"]).upper()
        experimental = float(row["Expt_pKa"])
        key = f"{pdb}.{row['Chain']}"
        records.append({
            "pkadr_index": str(row.get("Index", "")),
            "pdb": pdb,
            "chain": str(row["Chain"]),
            "residue_id": str(row["ResID_in_PDB"]),
            "residue_name": residue,
            "chemistry": chemistry,
            "homology_group": groups.get(key, key),
            "experimental_pka": experimental,
            "reference_pka": REFERENCE_PKA[residue],
            "delta_pka": experimental - REFERENCE_PKA[residue],
            **features,
        })
    table = pd.DataFrame(records)
    if table.empty:
        raise RuntimeError("No structure-mapped PKAD-R training rows were extracted")
    return table, exclusions


def _normalized_residue_id(value: object) -> str:
    """Normalize spreadsheet-like residue identifiers without losing inserts."""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def _structure_path(
    pdb: str,
    primary_dir: Path,
    fallback_dir: Optional[Path] = None,
) -> Path:
    """Resolve one RCSB coordinate file without changing its provenance."""
    primary = primary_dir / f"{pdb}.cif.gz"
    if primary.is_file() or fallback_dir is None:
        return primary
    fallback = fallback_dir / f"{pdb}.cif.gz"
    return fallback if fallback.is_file() else primary


def build_pkad3_supplement_table(
    candidates_path: str | Path,
    structures_dir: str | Path,
    pkadr_source_path: str | Path,
    *,
    fallback_structures_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, list[Dict[str, object]]]:
    """Recompute descriptors for de-overlapped experimental PKAD-3 rows.

    The KaML/PKAD-3 feature and model columns are never read here.  Exact
    PKAD-R structure-site overlap is checked again at the fitting boundary so
    a stale or manually edited preparation table fails closed.
    """
    candidate_file = Path(candidates_path)
    if not candidate_file.is_file():
        return pd.DataFrame(), [{
            "reason": "pkad3_candidate_table_not_found",
            "path": str(candidate_file),
        }]
    candidates = pd.read_csv(candidate_file, dtype=str, keep_default_na=False)
    required = {
        "supplement_id", "pdb", "chain", "residue_name", "residue_id",
        "experimental_pka", "measurement_group_key", "protein_group_key",
        "sample_weight_raw", "label_origin", "computational_teacher",
        "computed_pkad3_features_used", "computed_model_pka_used",
        "eligible_for_training",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"PKAD-3 candidate table missing columns: {missing}")
    forbidden = candidates[
        candidates["label_origin"].str.lower().ne("experimental")
        | candidates["computational_teacher"].str.strip().ne("")
        | candidates["computed_pkad3_features_used"].str.lower().ne("false")
        | candidates["computed_model_pka_used"].str.lower().ne("false")
        | candidates["eligible_for_training"].str.lower().ne("true")
    ]
    if not forbidden.empty:
        raise ValueError(
            "PKAD-3 supplement contains ineligible or computationally derived rows"
        )

    pkadr_rows = json.loads(Path(pkadr_source_path).read_text(encoding="utf-8"))
    canonical_site_keys = {
        "|".join([
            str(row.get("PDB", "")).upper(), str(row.get("Chain", "")),
            str(row.get("ResName", "")).upper(),
            _normalized_residue_id(row.get("ResID_in_PDB", "")),
        ])
        for row in pkadr_rows
    }
    primary = Path(structures_dir)
    fallback = (
        Path(fallback_structures_dir)
        if fallback_structures_dir is not None else None
    )
    structure_cache: Dict[str, object] = {}
    records: list[Dict[str, object]] = []
    exclusions: list[Dict[str, object]] = []
    for _, row in candidates.iterrows():
        pdb = str(row["pdb"]).upper()
        chain = str(row["chain"])
        residue = str(row["residue_name"]).upper()
        residue_id = _normalized_residue_id(row["residue_id"])
        site_key = "|".join([pdb, chain, residue, residue_id])
        base = {
            "supplement_id": str(row["supplement_id"]),
            "pdb": pdb,
            "chain": chain,
            "residue_name": residue,
            "residue_id": residue_id,
            "measurement_group_key": str(row["measurement_group_key"]),
            "protein_group_key": str(row["protein_group_key"]),
        }
        if site_key in canonical_site_keys:
            exclusions.append({
                **base, "reason": "exact_structure_site_overlap_with_pkadr",
            })
            continue
        if residue not in REFERENCE_PKA:
            exclusions.append({**base, "reason": "unsupported_residue_class"})
            continue
        try:
            experimental = float(row["experimental_pka"])
            raw_weight = float(row["sample_weight_raw"])
        except ValueError:
            exclusions.append({**base, "reason": "invalid_numeric_field"})
            continue
        if not np.isfinite(experimental) or not np.isfinite(raw_weight) or raw_weight <= 0:
            exclusions.append({**base, "reason": "invalid_numeric_field"})
            continue
        structure = _structure_path(pdb, primary, fallback)
        try:
            if pdb not in structure_cache:
                structure_cache[pdb] = read_structure_atoms(structure)
            atoms = structure_cache[pdb]
            center, residue_key = residue_site_center(
                atoms,
                chain=chain,
                residue_id=residue_id,
                residue_name=residue,
            )
            chemistry = RESIDUE_CHEMISTRY[residue]
            features = environment_descriptors(
                atoms,
                center,
                chemistry=chemistry,
                excluded_residue_keys=[residue_key],
            )
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            exclusions.append({**base, "reason": str(exc)})
            continue
        records.append({
            **base,
            "site_key": site_key,
            "chemistry": chemistry,
            "experimental_pka": experimental,
            "reference_pka": REFERENCE_PKA[residue],
            "delta_pka": experimental - REFERENCE_PKA[residue],
            "sample_weight_raw": raw_weight,
            "source_dataset": "PKAD-3/KaMLs",
            **features,
        })
    table = pd.DataFrame(records)
    if not table.empty:
        raw_total = float(table["sample_weight_raw"].sum())
        table["sample_weight"] = table["sample_weight_raw"]
        table.attrs["raw_weight_total"] = raw_total
    return table, exclusions


def extend_homology_groups(
    canonical: pd.DataFrame,
    supplement: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join PKAD-R components and PKAD-3 UniProt contexts transitively."""
    canonical = canonical.copy()
    supplement = supplement.copy()
    if supplement.empty:
        return canonical, supplement
    union = _UnionFind()
    old_group_node: Dict[str, str] = {}
    context_nodes: set[str] = set()
    for _, row in canonical.iterrows():
        context = f"context:{row['pdb']}.{row['chain']}"
        context_nodes.add(context)
        group = str(row["homology_group"])
        group_node = old_group_node.setdefault(group, f"pkadr_group:{group}")
        union.union(context, group_node)
    for _, row in supplement.iterrows():
        context = f"context:{row['pdb']}.{row['chain']}"
        protein = f"uniprot:{row['protein_group_key']}"
        context_nodes.add(context)
        union.union(context, protein)
    component_contexts: Dict[str, list[str]] = defaultdict(list)
    for context in context_nodes:
        component_contexts[union.find(context)].append(context.removeprefix("context:"))
    labels = {
        root: min(contexts) for root, contexts in component_contexts.items()
    }

    def label(pdb: object, chain: object) -> str:
        node = f"context:{str(pdb).upper()}.{chain}"
        return labels[union.find(node)]

    canonical["homology_group"] = [
        label(row.pdb, row.chain) for row in canonical.itertuples(index=False)
    ]
    supplement["homology_group"] = [
        label(row.pdb, row.chain) for row in supplement.itertuples(index=False)
    ]
    return canonical, supplement


def _exact_pka(row: Mapping[str, object]) -> Optional[float]:
    """Return a finite exact pKa, rejecting bounds and approximate strings."""
    text = str(row.get("Expt_pKa", "")).strip()
    if not text or any(marker in text for marker in ("<", ">", "~", "±")):
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _alternative_groups(
    rows: Sequence[Mapping[str, object]],
) -> Iterable[tuple[Mapping[str, object], Mapping[str, object]]]:
    """Yield each alternative together with the preceding PKAD-R main row.

    PKAD-R serializes alternatives immediately after the main record they
    qualify.  This also handles documented residue-number offsets such as the
    ResA Cys73/Cys74 and Cys76/Cys77 pairs, which cannot be joined by a naive
    PDB residue key.
    """
    main: Optional[Mapping[str, object]] = None
    for row in rows:
        if str(row.get("pKa_Classification", "")) == "Main":
            main = row
        elif main is not None:
            yield main, row


def replicate_measurement_summary(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Summarize repeat measurements without treating them as new samples."""
    differences = []
    for main, alternative in _alternative_groups(rows):
        if str(alternative.get("pKa_Classification", "")) != "Alt. pKa":
            continue
        main_pka, alternative_pka = _exact_pka(main), _exact_pka(alternative)
        if (
            main_pka is None
            or alternative_pka is None
            or str(main.get("Warning", "")).strip()
            or str(alternative.get("Warning", "")).strip()
        ):
            continue
        differences.append(float(alternative_pka - main_pka))
    absolute = np.abs(np.asarray(differences, dtype=float))
    return {
        "rows": int(len(differences)),
        "median_absolute_difference": (
            float(np.median(absolute)) if len(absolute) else None
        ),
        "p90_absolute_difference": (
            float(np.quantile(absolute, 0.90)) if len(absolute) else None
        ),
        "maximum_absolute_difference": (
            float(np.max(absolute)) if len(absolute) else None
        ),
        "use": "measurement_noise_summary_not_independent_training_rows",
    }


def build_perturbation_table(
    source_path: str | Path,
    structures_dir: str | Path,
) -> tuple[pd.DataFrame, list[Dict[str, object]]]:
    """Build paired mutant/state rows with distinct experimental structures."""
    source = Path(source_path)
    structures = Path(structures_dir)
    rows = json.loads(source.read_text(encoding="utf-8"))
    groups = homology_groups(rows)
    structure_cache: Dict[str, object] = {}
    records: list[Dict[str, object]] = []
    exclusions: list[Dict[str, object]] = []
    allowed = {"Alt. pKa (mutant)", "Alt. pKa (state)"}

    for main, alternative in _alternative_groups(rows):
        classification = str(alternative.get("pKa_Classification", ""))
        if classification not in allowed:
            continue
        reason = None
        main_pka, alternative_pka = _exact_pka(main), _exact_pka(alternative)
        residue = str(main.get("ResName", "")).upper()
        if main_pka is None or alternative_pka is None:
            reason = "non_exact_pka"
        elif str(main.get("Warning", "")).strip() or str(
            alternative.get("Warning", "")
        ).strip():
            reason = "warning_flagged_measurement"
        elif residue not in RESIDUE_CHEMISTRY or str(
            alternative.get("ResName", "")
        ).upper() != residue:
            reason = "unsupported_or_changed_target_residue"
        elif str(main.get("PDB", "")).upper() == str(
            alternative.get("PDB", "")
        ).upper():
            reason = "no_distinct_state_structure"

        base = {
            "main_pkadr_index": str(main.get("Index", "")),
            "alternative_pkadr_index": str(alternative.get("Index", "")),
            "classification": classification,
            "main_pdb": str(main.get("PDB", "")).upper(),
            "main_chain": str(main.get("Chain", "")),
            "alternative_pdb": str(alternative.get("PDB", "")).upper(),
            "alternative_chain": str(alternative.get("Chain", "")),
            "residue_name": residue,
            "mutation": str(alternative.get("Mut_Pos", "")),
            "notes": str(alternative.get("Notes", "")),
        }
        if reason is not None:
            exclusions.append({**base, "reason": reason})
            continue

        chemistry = RESIDUE_CHEMISTRY[residue]
        feature_sets: Dict[str, Dict[str, float]] = {}
        try:
            for prefix, row in (("main", main), ("alternative", alternative)):
                pdb = str(row["PDB"]).upper()
                path = structures / f"{pdb}.cif.gz"
                if pdb not in structure_cache:
                    structure_cache[pdb] = read_structure_atoms(path)
                atoms = structure_cache[pdb]
                center, residue_key = residue_site_center(
                    atoms,
                    chain=str(row["Chain"]),
                    residue_id=str(row["ResID_in_PDB"]),
                    residue_name=str(row["ResName"]),
                )
                feature_sets[prefix] = environment_descriptors(
                    atoms,
                    center,
                    chemistry=chemistry,
                    excluded_residue_keys=[residue_key],
                )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            exclusions.append({**base, "reason": str(exc)})
            continue

        key = f"{str(main['PDB']).upper()}.{main['Chain']}"
        record: Dict[str, object] = {
            **base,
            "chemistry": chemistry,
            "homology_group": groups.get(key, key),
            "main_experimental_pka": float(main_pka),
            "alternative_experimental_pka": float(alternative_pka),
            "main_delta_pka": float(main_pka - REFERENCE_PKA[residue]),
            "alternative_delta_pka": float(
                alternative_pka - REFERENCE_PKA[residue]
            ),
            "experimental_perturbation_delta_pka": float(
                alternative_pka - main_pka
            ),
        }
        for feature in FEATURE_COLUMNS:
            record[f"main__{feature}"] = feature_sets["main"][feature]
            record[f"alternative__{feature}"] = feature_sets["alternative"][feature]
        records.append(record)

    table = pd.DataFrame(records)
    if not table.empty:
        alternatives_per_main = table.groupby("main_pkadr_index")[
            "alternative_pkadr_index"
        ].transform("count")
        # Each main site contributes at most half the weight of a canonical row,
        # regardless of how many mutant/state alternatives were reported.
        table["augmentation_weight"] = 0.5 / alternatives_per_main.astype(float)
    return table, exclusions


def _model() -> RandomForestRegressor:
    """Return the fixed, regularized nonlinear residual model."""
    return RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=8,
        max_features=0.75,
        random_state=41,
        n_jobs=-1,
    )


def grouped_oof_predictions(
    table: pd.DataFrame,
    *,
    folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return environmental and chemistry-only homology-held-out predictions."""
    x = table.loc[:, FEATURE_COLUMNS]
    y = table["delta_pka"].to_numpy(dtype=float)
    groups = table["homology_group"].astype(str).to_numpy()
    n_splits = min(int(folds), len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("At least two independent homology groups are required")
    split = list(GroupKFold(n_splits=n_splits).split(x, y, groups))
    predictions = np.full(len(table), np.nan, dtype=float)
    chemistry_baseline = np.full(len(table), np.nan, dtype=float)
    for train_index, test_index in split:
        estimator = _model()
        estimator.fit(x.iloc[train_index], y[train_index])
        predictions[test_index] = estimator.predict(x.iloc[test_index])
        training = table.iloc[train_index]
        chemistry_medians = training.groupby("chemistry")["delta_pka"].median()
        global_median = float(training["delta_pka"].median())
        chemistry_baseline[test_index] = (
            table.iloc[test_index]["chemistry"]
            .map(chemistry_medians)
            .fillna(global_median)
            .to_numpy(dtype=float)
        )
    return predictions, chemistry_baseline


def _perturbation_features(
    table: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Return one endpoint's descriptors with canonical feature names."""
    return table.loc[:, [f"{prefix}__{name}" for name in FEATURE_COLUMNS]].rename(
        columns={f"{prefix}__{name}": name for name in FEATURE_COLUMNS}
    )


def grouped_oof_with_perturbations(
    table: pd.DataFrame,
    perturbations: pd.DataFrame,
    *,
    augment_training: bool,
    folds: int = 5,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Cross-fit canonical sites and held-out paired environmental changes."""
    x = table.loc[:, FEATURE_COLUMNS]
    y = table["delta_pka"].to_numpy(dtype=float)
    groups = table["homology_group"].astype(str).to_numpy()
    n_splits = min(int(folds), len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("At least two independent homology groups are required")
    predictions = np.full(len(table), np.nan, dtype=float)
    pair_records: list[Dict[str, object]] = []
    for fold_index, (train_index, test_index) in enumerate(
        GroupKFold(n_splits=n_splits).split(x, y, groups), start=1
    ):
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        train_x = x.iloc[train_index].reset_index(drop=True)
        train_y = y[train_index]
        train_weight = np.ones(len(train_index), dtype=float)
        weighted_training = False
        if augment_training and not perturbations.empty:
            additions = perturbations[
                perturbations["homology_group"].astype(str).isin(train_groups)
            ]
            if not additions.empty:
                train_x = pd.concat([
                    train_x,
                    _perturbation_features(additions, "alternative"),
                ], ignore_index=True)
                train_y = np.concatenate([
                    train_y,
                    additions["alternative_delta_pka"].to_numpy(dtype=float),
                ])
                train_weight = np.concatenate([
                    train_weight,
                    additions["augmentation_weight"].to_numpy(dtype=float),
                ])
                weighted_training = True
        estimator = _model()
        if weighted_training:
            estimator.fit(train_x, train_y, sample_weight=train_weight)
        else:
            # Passing an all-ones sample_weight is not a no-op for every
            # sklearn estimator/version.  Preserve the exact baseline fit
            # whenever this fold has no weighted augmentation rows.
            estimator.fit(train_x, train_y)
        predictions[test_index] = estimator.predict(x.iloc[test_index])

        if perturbations.empty:
            continue
        held_out = perturbations[
            perturbations["homology_group"].astype(str).isin(test_groups)
        ]
        if held_out.empty:
            continue
        main_prediction = estimator.predict(
            _perturbation_features(held_out, "main")
        )
        alternative_prediction = estimator.predict(
            _perturbation_features(held_out, "alternative")
        )
        for position, (_, pair) in enumerate(held_out.iterrows()):
            predicted_change = float(
                alternative_prediction[position] - main_prediction[position]
            )
            observed_change = float(pair["experimental_perturbation_delta_pka"])
            pair_records.append({
                "fold": int(fold_index),
                "main_pkadr_index": str(pair["main_pkadr_index"]),
                "alternative_pkadr_index": str(pair["alternative_pkadr_index"]),
                "classification": str(pair["classification"]),
                "chemistry": str(pair["chemistry"]),
                "homology_group": str(pair["homology_group"]),
                "experimental_perturbation_delta_pka": observed_change,
                "predicted_perturbation_delta_pka": predicted_change,
                "absolute_perturbation_error": abs(predicted_change - observed_change),
            })
    return predictions, pd.DataFrame(pair_records)


def grouped_oof_with_pkad3(
    table: pd.DataFrame,
    supplement: pd.DataFrame,
    perturbations: pd.DataFrame,
    *,
    augment_perturbations: bool,
    augment_pkad3: bool,
    folds: int = 5,
) -> np.ndarray:
    """Predict canonical PKAD-R folds with optional external augmentation.

    PKAD-3 rows connected to the held-out canonical component are excluded
    from that fold.  Components absent from PKAD-R may train every fold; they
    can never place the same protein on the held-out side because UniProt and
    PKAD-R homology links have already been collapsed by
    :func:`extend_homology_groups`.
    """
    x = table.loc[:, FEATURE_COLUMNS]
    y = table["delta_pka"].to_numpy(dtype=float)
    groups = table["homology_group"].astype(str).to_numpy()
    n_splits = min(int(folds), len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("At least two independent homology groups are required")
    predictions = np.full(len(table), np.nan, dtype=float)
    for train_index, test_index in GroupKFold(n_splits=n_splits).split(
        x, y, groups
    ):
        train_groups = set(groups[train_index])
        test_groups = set(groups[test_index])
        train_x = x.iloc[train_index].reset_index(drop=True)
        train_y = y[train_index]
        train_weight = np.ones(len(train_index), dtype=float)
        weighted_training = False
        if augment_perturbations and not perturbations.empty:
            additions = perturbations[
                perturbations["homology_group"].astype(str).isin(train_groups)
            ]
            if not additions.empty:
                train_x = pd.concat([
                    train_x, _perturbation_features(additions, "alternative")
                ], ignore_index=True)
                train_y = np.concatenate([
                    train_y,
                    additions["alternative_delta_pka"].to_numpy(dtype=float),
                ])
                train_weight = np.concatenate([
                    train_weight,
                    additions["augmentation_weight"].to_numpy(dtype=float),
                ])
                weighted_training = True
        if augment_pkad3 and not supplement.empty:
            additions = supplement[
                ~supplement["homology_group"].astype(str).isin(test_groups)
            ]
            if not additions.empty:
                train_x = pd.concat([
                    train_x, additions.loc[:, FEATURE_COLUMNS]
                ], ignore_index=True)
                train_y = np.concatenate([
                    train_y, additions["delta_pka"].to_numpy(dtype=float)
                ])
                train_weight = np.concatenate([
                    train_weight, additions["sample_weight"].to_numpy(dtype=float)
                ])
                weighted_training = True
        estimator = _model()
        if weighted_training:
            estimator.fit(train_x, train_y, sample_weight=train_weight)
        else:
            estimator.fit(train_x, train_y)
        predictions[test_index] = estimator.predict(x.iloc[test_index])
    return predictions


def _group_balanced_mae(
    table: pd.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
) -> float:
    errors = pd.DataFrame({
        "homology_group": table["homology_group"].astype(str).to_numpy(),
        "absolute_error": np.abs(target - prediction),
    })
    return float(errors.groupby("homology_group")["absolute_error"].mean().mean())


def _pkad3_gate_metrics(
    table: pd.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
) -> Dict[str, float]:
    absolute = np.abs(target - prediction)
    return {
        "canonical_pkadr_oof_mae": float(np.mean(absolute)),
        "canonical_pkadr_group_balanced_oof_mae": _group_balanced_mae(
            table, target, prediction
        ),
        "canonical_pkadr_oof_rmse": float(
            root_mean_squared_error(target, prediction)
        ),
        "canonical_pkadr_oof_p90_absolute_error": float(
            np.quantile(absolute, 0.90)
        ),
    }


def _perturbation_metrics(
    canonical_target: np.ndarray,
    canonical_prediction: np.ndarray,
    pair_predictions: pd.DataFrame,
) -> Dict[str, object]:
    pair_error = pair_predictions.get(
        "absolute_perturbation_error", pd.Series(dtype=float)
    ).to_numpy(dtype=float)
    pair_observed = pair_predictions.get(
        "experimental_perturbation_delta_pka", pd.Series(dtype=float)
    ).to_numpy(dtype=float)
    pair_predicted = pair_predictions.get(
        "predicted_perturbation_delta_pka", pd.Series(dtype=float)
    ).to_numpy(dtype=float)
    return {
        "canonical_rows": int(len(canonical_target)),
        "canonical_oof_mae": float(mean_absolute_error(
            canonical_target, canonical_prediction
        )),
        "paired_rows": int(len(pair_error)),
        "paired_delta_oof_mae": float(np.mean(pair_error)) if len(pair_error) else None,
        "paired_delta_oof_rmse": (
            float(root_mean_squared_error(pair_observed, pair_predicted))
            if len(pair_error) else None
        ),
        "paired_delta_oof_bias": (
            float(np.mean(pair_predicted - pair_observed))
            if len(pair_error) else None
        ),
    }


def _finite_sample_radius(values: np.ndarray, coverage: float) -> float:
    """Return the split-conformal finite-sample absolute-residual radius."""
    residuals = np.sort(np.asarray(values, dtype=float))
    if not len(residuals):
        raise ValueError("Cannot calibrate an empty residual distribution")
    rank = min(len(residuals), int(np.ceil((len(residuals) + 1) * coverage)))
    return float(residuals[rank - 1])


def _group_bootstrap_gain(
    table: pd.DataFrame,
    *,
    repetitions: int = 2000,
    seed: int = 41,
) -> Dict[str, float]:
    """Estimate uncertainty in MAE gain by resampling whole homology groups."""
    grouped = []
    for _, group in table.groupby("homology_group"):
        grouped.append((
            len(group),
            float(np.abs(group["delta_pka"]).sum()),
            float(np.abs(
                group["delta_pka"] - group["oof_predicted_delta_pka"]
            ).sum()),
            float(np.abs(
                group["delta_pka"]
                - group["oof_chemistry_median_delta_pka"]
            ).sum()),
        ))
    if not grouped:
        raise ValueError("Cannot bootstrap an empty calibration table")
    aggregates = np.asarray(grouped, dtype=float)
    random = np.random.default_rng(seed)
    gains = np.empty(int(repetitions), dtype=float)
    chemistry_gains = np.empty(int(repetitions), dtype=float)
    for index in range(int(repetitions)):
        sample = aggregates[random.integers(0, len(aggregates), len(aggregates))]
        gains[index] = float(
            (sample[:, 1].sum() - sample[:, 2].sum()) / sample[:, 0].sum()
        )
        chemistry_gains[index] = float(
            (sample[:, 3].sum() - sample[:, 2].sum()) / sample[:, 0].sum()
        )
    return {
        "mae_gain_bootstrap_p05": float(np.quantile(gains, 0.05)),
        "mae_gain_bootstrap_median": float(np.quantile(gains, 0.50)),
        "mae_gain_bootstrap_p95": float(np.quantile(gains, 0.95)),
        "probability_mae_gain_positive": float(np.mean(gains > 0.0)),
        "mae_gain_over_chemistry_baseline_bootstrap_p05": float(
            np.quantile(chemistry_gains, 0.05)
        ),
        "mae_gain_over_chemistry_baseline_bootstrap_median": float(
            np.quantile(chemistry_gains, 0.50)
        ),
        "mae_gain_over_chemistry_baseline_bootstrap_p95": float(
            np.quantile(chemistry_gains, 0.95)
        ),
        "probability_mae_gain_over_chemistry_baseline_positive": float(
            np.mean(chemistry_gains > 0.0)
        ),
        "bootstrap_repetitions": int(repetitions),
        "bootstrap_unit": "homology_group",
    }


def _calibration_summary(table: pd.DataFrame) -> Dict[str, object]:
    """Summarize grouped-OOF accuracy, uncertainty and correction support."""
    observed = table["delta_pka"].to_numpy(dtype=float)
    predicted = table["oof_predicted_delta_pka"].to_numpy(dtype=float)
    error = predicted - observed
    absolute = np.abs(error)
    model_mae = float(np.mean(absolute))
    zero_mae = float(np.mean(np.abs(observed)))
    chemistry_baseline = table[
        "oof_chemistry_median_delta_pka"
    ].to_numpy(dtype=float)
    chemistry_mae = float(np.mean(np.abs(observed - chemistry_baseline)))
    gain = zero_mae - model_mae
    chemistry_gain = chemistry_mae - model_mae
    bootstrap = _group_bootstrap_gain(table)
    lower_gain = float(bootstrap["mae_gain_bootstrap_p05"])
    lower_chemistry_gain = float(
        bootstrap["mae_gain_over_chemistry_baseline_bootstrap_p05"]
    )
    if lower_gain > 0.0 and lower_chemistry_gain > 0.0:
        evidence_tier = "validated_grouped_oof_improvement"
        correction_scale = 1.0
    elif gain > 0.0 and chemistry_gain > 0.0:
        evidence_tier = "provisional_grouped_oof_improvement"
        correction_scale = 0.5
    else:
        evidence_tier = "no_grouped_oof_improvement_correction_withheld"
        correction_scale = 0.0
    correlation = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if len(observed) >= 3
        and float(np.std(observed)) > 0.0
        and float(np.std(predicted)) > 0.0
        else None
    )
    radii: Dict[str, float] = {}
    for level in (0.50, 0.80, 0.90, 0.95):
        label = int(level * 100)
        radius = _finite_sample_radius(absolute, level)
        radii[f"absolute_error_radius_{label}"] = radius
        radii[f"empirical_absolute_error_coverage_{label}"] = float(
            np.mean(absolute <= radius)
        )
    return {
        "rows": int(len(table)),
        "homology_groups": int(table["homology_group"].nunique()),
        "model_oof_mae": model_mae,
        "zero_shift_baseline_mae": zero_mae,
        "mae_gain_over_zero_shift": gain,
        "chemistry_only_oof_median_baseline_mae": chemistry_mae,
        "mae_gain_over_chemistry_only_baseline": chemistry_gain,
        "model_oof_rmse": float(root_mean_squared_error(observed, predicted)),
        "model_oof_bias_predicted_minus_observed": float(np.mean(error)),
        "oof_pearson_correlation": correlation,
        "correction_evidence_tier": evidence_tier,
        "correction_scale": correction_scale,
        **radii,
        **bootstrap,
    }


def build_calibration_artifact(
    table: pd.DataFrame,
    exclusions: Sequence[Mapping[str, object]],
) -> tuple[Dict[str, object], pd.DataFrame]:
    """Build the complete strict-cohort ledger and inference calibration."""
    chemistry = {
        name: _calibration_summary(group.copy())
        for name, group in table.groupby("chemistry")
    }
    descriptor_failures = [
        dict(row) for row in exclusions
        if "lacks ionization-center atoms" in str(row.get("reason", ""))
    ]
    missing_targets = [
        dict(row) for row in exclusions
        if "Missing target residue" in str(row.get("reason", ""))
    ]
    ready = table.copy()
    ready["calibration_status"] = "descriptor_ready_grouped_oof_prediction"
    ready["calibration_exclusion_reason"] = ""
    unavailable = pd.DataFrame(descriptor_failures)
    if not unavailable.empty:
        unavailable["calibration_status"] = "missing_terminal_site_atoms"
        unavailable["calibration_exclusion_reason"] = unavailable["reason"]
        unavailable["oof_predicted_delta_pka"] = np.nan
        unavailable["oof_absolute_error"] = np.nan
    cohort = pd.concat([ready, unavailable], ignore_index=True, sort=False)
    cohort = cohort.sort_values(
        ["pdb", "chain", "residue_id", "pkadr_index"],
        na_position="last",
    ).reset_index(drop=True)
    calibration: Dict[str, object] = {
        "schema_version": ENVIRONMENT_CALIBRATION_SCHEMA_VERSION,
        "method": (
            "homology_grouped_oof_residual_calibration_with_group_bootstrap_"
            "correction_evidence"
        ),
        "strict_numeric_warning_free_rows": int(len(table) + len(exclusions)),
        "strict_structure_mapped_cohort_rows": int(len(cohort)),
        "descriptor_ready_rows": int(len(table)),
        "descriptor_unavailable_rows": int(len(descriptor_failures)),
        "target_residue_missing_rows_excluded_before_cohort": int(
            len(missing_targets)
        ),
        "overall": _calibration_summary(table),
        "chemistry": chemistry,
        "interval_semantics": (
            "finite-sample absolute grouped-OOF residual radii; protein-residue "
            "calibration only, before ligand-analogue transfer penalty"
        ),
        "correction_policy": (
            "full scale only when homology-group bootstrap p05 MAE gains over both "
            "zero shift and a fold-fitted chemistry-only median are positive; half "
            "scale when both point gains are positive but bootstrap support is "
            "uncertain; otherwise withhold the environment correction"
        ),
        "evaluation_status": (
            "cross-fitted representation calibration on the development cohort; "
            "not a locked external bound-ligand test"
        ),
    }
    return calibration, cohort


def train(
    source_path: str | Path = DEFAULT_SOURCE,
    structures_dir: str | Path = DEFAULT_STRUCTURES,
    output_path: str | Path = DEFAULT_OUTPUT,
    pkad3_candidates_path: str | Path | None = None,
    pkad3_structures_dir: str | Path = DEFAULT_PKAD3_STRUCTURES,
    direct_ligand_gate_path: str | Path | None = DEFAULT_DIRECT_LIGAND_GATE,
    direct_ligand_model_path: str | Path | None = DEFAULT_DIRECT_LIGAND_MODEL,
) -> Dict[str, object]:
    """Train, serialize, and audit the protein-environment model."""
    source = Path(source_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table, exclusions = build_training_table(source, structures_dir)
    perturbations, perturbation_exclusions = build_perturbation_table(
        source, structures_dir
    )
    pkad3_exclusions: list[Dict[str, object]] = []
    if pkad3_candidates_path is None:
        pkad3 = pd.DataFrame()
    else:
        pkad3, pkad3_exclusions = build_pkad3_supplement_table(
            pkad3_candidates_path,
            pkad3_structures_dir,
            source,
            fallback_structures_dir=structures_dir,
        )
    pkad3_raw_weight = float(
        pkad3["sample_weight_raw"].sum() if not pkad3.empty else 0.0
    )
    pkad3_weight_cap = float(MAX_PKAD3_WEIGHT_FRACTION * len(table))
    pkad3_weight_scale = (
        min(1.0, pkad3_weight_cap / pkad3_raw_weight)
        if pkad3_raw_weight > 0.0 else 0.0
    )
    if not pkad3.empty:
        pkad3["sample_weight"] = (
            pkad3["sample_weight_raw"].astype(float) * pkad3_weight_scale
        )
        table, pkad3 = extend_homology_groups(table, pkad3)
        context_groups = {
            f"{row.pdb}.{row.chain}": str(row.homology_group)
            for row in table.itertuples(index=False)
        }
        if not perturbations.empty:
            perturbations = perturbations.copy()
            perturbations["homology_group"] = [
                context_groups.get(
                    f"{row.main_pdb}.{row.main_chain}", row.homology_group
                )
                for row in perturbations.itertuples(index=False)
            ]
    source_rows = json.loads(source.read_text(encoding="utf-8"))
    repeat_summary = replicate_measurement_summary(source_rows)
    baseline_oof, chemistry_baseline = grouped_oof_predictions(table)
    paired_baseline_oof, baseline_pair_predictions = (
        grouped_oof_with_perturbations(
            table, perturbations, augment_training=False
        )
    )
    if not np.allclose(baseline_oof, paired_baseline_oof, atol=1e-12, rtol=0.0):
        raise RuntimeError("Baseline grouped folds changed during perturbation audit")
    augmented_oof, augmented_pair_predictions = grouped_oof_with_perturbations(
        table, perturbations, augment_training=True
    )
    target = table["delta_pka"].to_numpy(dtype=float)
    baseline_perturbation_metrics = _perturbation_metrics(
        target, baseline_oof, baseline_pair_predictions
    )
    augmented_perturbation_metrics = _perturbation_metrics(
        target, augmented_oof, augmented_pair_predictions
    )
    baseline_pair_mae = baseline_perturbation_metrics["paired_delta_oof_mae"]
    augmented_pair_mae = augmented_perturbation_metrics["paired_delta_oof_mae"]
    canonical_degradation = float(
        augmented_perturbation_metrics["canonical_oof_mae"]
        - baseline_perturbation_metrics["canonical_oof_mae"]
    )
    deploy_augmentation = bool(
        len(perturbations)
        and baseline_pair_mae is not None
        and augmented_pair_mae is not None
        and float(augmented_pair_mae) < float(baseline_pair_mae)
        and canonical_degradation <= MAX_MAIN_OOF_MAE_DEGRADATION
    )
    pre_pkad3_oof = augmented_oof if deploy_augmentation else baseline_oof
    if not pkad3.empty:
        gate_baseline_oof = grouped_oof_with_pkad3(
            table,
            pkad3,
            perturbations,
            augment_perturbations=deploy_augmentation,
            augment_pkad3=False,
        )
        if not np.allclose(
            pre_pkad3_oof, gate_baseline_oof, atol=1e-12, rtol=0.0
        ):
            raise RuntimeError("PKAD-3 gate changed the canonical baseline folds")
        pkad3_candidate_oof = grouped_oof_with_pkad3(
            table,
            pkad3,
            perturbations,
            augment_perturbations=deploy_augmentation,
            augment_pkad3=True,
        )
    else:
        gate_baseline_oof = pre_pkad3_oof
        pkad3_candidate_oof = pre_pkad3_oof
    target = table["delta_pka"].to_numpy(dtype=float)
    pkad3_baseline_metrics = _pkad3_gate_metrics(
        table, target, gate_baseline_oof
    )
    pkad3_candidate_metrics = _pkad3_gate_metrics(
        table, target, pkad3_candidate_oof
    )
    deploy_pkad3 = bool(
        len(pkad3)
        and pkad3_candidate_metrics["canonical_pkadr_oof_mae"]
        <= pkad3_baseline_metrics["canonical_pkadr_oof_mae"] + 1.0e-12
        and pkad3_candidate_metrics["canonical_pkadr_group_balanced_oof_mae"]
        <= pkad3_baseline_metrics[
            "canonical_pkadr_group_balanced_oof_mae"
        ] + 1.0e-12
    )
    oof = pkad3_candidate_oof if deploy_pkad3 else pre_pkad3_oof
    absolute_error = np.abs(target - oof)
    table = table.copy()
    table["oof_predicted_delta_pka"] = oof
    table["oof_chemistry_median_delta_pka"] = chemistry_baseline
    table["oof_absolute_error"] = absolute_error
    calibration, cohort = build_calibration_artifact(table, exclusions)
    direct_ligand_gate, direct_ligand_model = _direct_ligand_calibration(
        direct_ligand_gate_path, direct_ligand_model_path
    )
    calibration["direct_bound_ligand_candidate_rows"] = int(
        direct_ligand_gate.get("strict_structure_mapped_candidate_rows", 0)
    )
    calibration["direct_bound_ligand_calibration_rows"] = int(
        direct_ligand_gate.get("deployed_training_rows", 0)
    )
    calibration["direct_bound_ligand_calibration"] = direct_ligand_gate
    chemistry_p90 = {
        chemistry: float(summary["absolute_error_radius_90"])
        for chemistry, summary in calibration["chemistry"].items()
    }
    estimator = _model()
    final_x = table.loc[:, FEATURE_COLUMNS].reset_index(drop=True)
    final_y = target
    final_weight = np.ones(len(table), dtype=float)
    if deploy_augmentation:
        final_x = pd.concat([
            final_x,
            _perturbation_features(perturbations, "alternative"),
        ], ignore_index=True)
        final_y = np.concatenate([
            final_y,
            perturbations["alternative_delta_pka"].to_numpy(dtype=float),
        ])
        final_weight = np.concatenate([
            final_weight,
            perturbations["augmentation_weight"].to_numpy(dtype=float),
        ])
    if deploy_pkad3:
        final_x = pd.concat([
            final_x, pkad3.loc[:, FEATURE_COLUMNS]
        ], ignore_index=True)
        final_y = np.concatenate([
            final_y, pkad3["delta_pka"].to_numpy(dtype=float)
        ])
        final_weight = np.concatenate([
            final_weight, pkad3["sample_weight"].to_numpy(dtype=float)
        ])
    if deploy_augmentation or deploy_pkad3:
        estimator.fit(final_x, final_y, sample_weight=final_weight)
    else:
        estimator.fit(final_x, final_y)
    clip = tuple(float(value) for value in np.quantile(final_y, [0.01, 0.99]))
    model_mae = float(mean_absolute_error(target, oof))
    null_mae = float(mean_absolute_error(target, np.zeros(len(target))))
    metrics: Dict[str, object] = {
        "schema_version": PROTEIN_ENVIRONMENT_SCHEMA_VERSION,
        "feature_schema_version": PROTEIN_ENVIRONMENT_FEATURE_SCHEMA_VERSION,
        "method": PROTEIN_ENVIRONMENT_METHOD,
        "training_rows": int(len(table)),
        "excluded_rows": int(len(exclusions)),
        "perturbation_candidate_rows": int(len(perturbations)),
        "perturbation_excluded_rows": int(len(perturbation_exclusions)),
        "perturbation_rows_deployed": (
            int(len(perturbations)) if deploy_augmentation else 0
        ),
        "pkad3_candidate_rows": int(len(pkad3)),
        "pkad3_excluded_rows": int(len(pkad3_exclusions)),
        "pkad3_rows_deployed": int(len(pkad3)) if deploy_pkad3 else 0,
        "deployment_fit_rows": int(len(final_y)),
        "effective_training_weight": float(np.sum(final_weight)),
        "homology_groups": int(table["homology_group"].nunique()),
        "model_class": "RandomForestRegressor",
        "model_parameters": {
            "n_estimators": 250,
            "min_samples_leaf": 8,
            "max_features": 0.75,
            "random_state": 41,
        },
        "oof_mae": model_mae,
        "oof_rmse": float(root_mean_squared_error(target, oof)),
        "oof_p90_abs_error": float(np.quantile(absolute_error, 0.90)),
        "zero_shift_baseline_mae": null_mae,
        "chemistry_only_oof_median_baseline_mae": float(mean_absolute_error(
            target, chemistry_baseline
        )),
        "oof_improvement_over_zero_shift_mae": null_mae - model_mae,
        "perturbation_augmentation": {
            "selection_policy": (
                "deploy only when homology-grouped paired-delta OOF MAE improves "
                "and canonical-site OOF MAE degrades by at most 0.01"
            ),
            "maximum_allowed_canonical_oof_mae_degradation": (
                MAX_MAIN_OOF_MAE_DEGRADATION
            ),
            "canonical_oof_mae_change_augmented_minus_baseline": (
                canonical_degradation
            ),
            "deployed": deploy_augmentation,
            "baseline": baseline_perturbation_metrics,
            "augmented": augmented_perturbation_metrics,
            "repeat_measurements": repeat_summary,
            "pair_types": (
                perturbations["classification"].value_counts().to_dict()
                if not perturbations.empty else {}
            ),
            "data_boundary": (
                "PKAD-R mutant/state alternatives only; repeat measurements are "
                "noise evidence; PKAD-2 and binding-shift panels are never fitted"
            ),
        },
        "pkad3_augmentation": {
            "selection_policy": (
                "deploy only when adding experimental PKAD-3 rows does not "
                "worsen either pooled or homology-group-balanced OOF MAE on "
                "canonical PKAD-R sites"
            ),
            "deployed": deploy_pkad3,
            "candidate_rows_with_recomputed_descriptors": int(len(pkad3)),
            "candidate_measurement_groups": int(
                pkad3["measurement_group_key"].nunique()
                if not pkad3.empty else 0
            ),
            "candidate_protein_groups": int(
                pkad3["protein_group_key"].nunique()
                if not pkad3.empty else 0
            ),
            "raw_candidate_weight": pkad3_raw_weight,
            "maximum_weight_fraction_of_canonical_rows": (
                MAX_PKAD3_WEIGHT_FRACTION
            ),
            "weight_cap": pkad3_weight_cap,
            "applied_weight_scale": pkad3_weight_scale,
            "effective_candidate_weight": float(
                pkad3["sample_weight"].sum() if not pkad3.empty else 0.0
            ),
            "baseline": pkad3_baseline_metrics,
            "candidate": pkad3_candidate_metrics,
            "data_boundary": (
                "experimental Expt_pKa and identifiers only; all KaML/PKAD-3 "
                "computed descriptors and model_pka values are ignored; exact "
                "PKAD-R sites are removed and project descriptors are recomputed"
            ),
            "reference_overlap_limitation": (
                "published split files contain no row-level references, so "
                "reference-level de-overlap cannot be proven"
            ),
        },
        "chemistry_p90_abs_error": chemistry_p90,
        "calibration": calibration,
        "direct_bound_ligand_calibration": direct_ligand_gate,
        "prediction_clip": list(clip),
        "reference_pka": REFERENCE_PKA,
        "feature_importances": {
            feature: float(importance)
            for feature, importance in zip(
                FEATURE_COLUMNS, estimator.feature_importances_
            )
        },
        "training_source_sha256": _sha256(source),
        "pkad3_candidate_source_sha256": (
            _sha256(Path(pkad3_candidates_path))
            if pkad3_candidates_path is not None
            and Path(pkad3_candidates_path).is_file() else None
        ),
        "validation_split": (
            "GroupKFold over connected PKAD-R >30% identity components, "
            "extended transitively through PKAD-3 UniProt contexts"
        ),
        "transfer_scope": (
            "protein-residue environmental shifts; ligand application is analogue "
            "transfer with additional shrinkage and uncertainty"
        ),
    }
    bundle = {
        "model": estimator,
        "feature_columns": list(FEATURE_COLUMNS),
        "direct_bound_ligand_model": direct_ligand_model,
        **metrics,
    }
    with output.open("wb") as handle:
        pickle.dump(bundle, handle)
    table.to_csv(output.parent / "training_features_and_oof.csv", index=False)
    pkad3.to_csv(output.parent / "pkad3_supplement_features.csv", index=False)
    if not perturbations.empty:
        key_columns = ["main_pkadr_index", "alternative_pkadr_index"]
        baseline_pairs = baseline_pair_predictions[key_columns + [
            "predicted_perturbation_delta_pka", "absolute_perturbation_error"
        ]].rename(columns={
            "predicted_perturbation_delta_pka": (
                "baseline_oof_predicted_perturbation_delta_pka"
            ),
            "absolute_perturbation_error": "baseline_oof_absolute_perturbation_error",
        })
        augmented_pairs = augmented_pair_predictions[key_columns + [
            "predicted_perturbation_delta_pka", "absolute_perturbation_error"
        ]].rename(columns={
            "predicted_perturbation_delta_pka": (
                "augmented_oof_predicted_perturbation_delta_pka"
            ),
            "absolute_perturbation_error": "augmented_oof_absolute_perturbation_error",
        })
        perturbation_output = (
            perturbations.merge(baseline_pairs, on=key_columns, how="left")
            .merge(augmented_pairs, on=key_columns, how="left")
        )
    else:
        perturbation_output = perturbations
    perturbation_output.to_csv(
        output.parent / "perturbation_pairs_and_oof.csv", index=False
    )
    cohort.to_csv(
        output.parent / "strict_648_site_calibration_cohort.csv", index=False
    )
    (output.parent / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output.parent / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output.parent / "excluded_rows.json").write_text(
        json.dumps(exclusions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output.parent / "perturbation_excluded_rows.json").write_text(
        json.dumps(perturbation_exclusions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output.parent / "pkad3_excluded_rows.json").write_text(
        json.dumps(pkad3_exclusions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the homology-grouped protein-environment residual model."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--structures", default=str(DEFAULT_STRUCTURES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--pkad3-candidates", default=str(DEFAULT_PKAD3_CANDIDATES),
        help="Prepared experimental PKAD-3 candidates; pass an empty string to disable.",
    )
    parser.add_argument(
        "--pkad3-structures", default=str(DEFAULT_PKAD3_STRUCTURES)
    )
    parser.add_argument(
        "--direct-ligand-gate", default=str(DEFAULT_DIRECT_LIGAND_GATE),
        help="Prepared direct-ligand evidence gate; pass an empty string to disable.",
    )
    parser.add_argument(
        "--direct-ligand-model", default=str(DEFAULT_DIRECT_LIGAND_MODEL),
        help="Optional model emitted only when the direct-ligand gate passes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(train(
        args.source,
        args.structures,
        args.output,
        args.pkad3_candidates or None,
        args.pkad3_structures,
        args.direct_ligand_gate or None,
        args.direct_ligand_model or None,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
