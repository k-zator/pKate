import argparse
import json
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger # type: ignore
from sklearn.ensemble import HistGradientBoostingRegressor # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # type: ignore

from geometry_features import (
    embed_molecule_3d,
    molecule_geometry_features,
    site_geometry_features,
    zero_molecule_geometry_features,
    zero_site_geometry_features,
)
from ml_features import molecule_descriptors, morgan_bits
from ml_splits import random_molecule_split, scaffold_molecule_split, split_manifest
from functional_group_pka_analysis import training_group_label
from pka_evidence_policy import reference_prior
from substructure_match import ( # type: ignore
    _compile_pattern_specs,
    get_default_pattern_specs,
    maximal_site_rank_key,
    resolve_overlapping_sites,
)
from training_data_resolver import (
    DEFAULT_CURATED_ASSIGNMENTS_PATH,
    DEFAULT_RAW_DIR,
    ensure_curated_training_data,
)


_COMPILED_SITE_SPECS = None
_CONTEXT_GROUP_LABELS = tuple(sorted({
    str(spec["label"]) for spec in get_default_pattern_specs()
}))


def _molecule_key(row: pd.Series) -> str:
    return f"{row['source_file']}::{int(row['record_index'])}::{row['smiles']}"


def load_single_group_rows(
    assignments_path: str,
    measurement_method: Optional[str] = None,
    min_supported_pka: float = -5.0,
    max_supported_pka: float = 20.0,
) -> pd.DataFrame:
    df = pd.read_csv(assignments_path)
    required = {
        "source_file",
        "record_index",
        "smiles",
        "pka_value",
        "final_group_label",
        "assignment_status",
        "pka_type_canonical",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "single_group_strict" in df.columns:
        single = df[df["single_group_strict"]].copy()
    else:
        single = df[df["assignment_status"] == "single_group"].copy()

    single = single.dropna(subset=["smiles", "pka_value", "final_group_label"]).copy()
    if measurement_method:
        if "pka_source_method" not in single.columns:
            raise ValueError("Cannot filter measurement provenance: pka_source_method is missing")
        method = str(measurement_method).strip().lower()
        single = single[
            single["pka_source_method"].astype(str).str.strip().str.lower() == method
        ].copy()

    # Filter out outlier pKa values flagged by distribution sanity checks.
    if "distribution_ok" in single.columns:
        n_before = len(single)
        single = single[single["distribution_ok"].astype(bool)].copy()
        n_removed = n_before - len(single)
        if n_removed > 0:
            print(f"  distribution_ok filter removed {n_removed}/{n_before} outlier rows")

    # Keep the legacy loader within the same explicitly supported scale as the
    # canonical Stage 1 data. This prevents older workflows from silently
    # reintroducing solvent/scale-unresolved extremes such as the pKa-30 row.
    pka_numeric = pd.to_numeric(single["pka_value"], errors="coerce")
    supported = pka_numeric.between(float(min_supported_pka), float(max_supported_pka))
    n_before = len(single)
    single = single[supported].copy()
    n_removed = n_before - len(single)
    if n_removed > 0:
        print(
            f"  supported pKa range [{min_supported_pka}, {max_supported_pka}] "
            f"removed {n_removed}/{n_before} rows"
        )

    single["molecule_key"] = single.apply(_molecule_key, axis=1)
    single = single.drop_duplicates(subset=["molecule_key", "pka_value", "final_group_label"])
    if "training_group_label" not in single.columns:
        single["training_group_label"] = single["final_group_label"].map(training_group_label)
    single.reset_index(drop=True, inplace=True)
    return single


def _compiled_site_specs():
    global _COMPILED_SITE_SPECS
    if _COMPILED_SITE_SPECS is None:
        _COMPILED_SITE_SPECS = _compile_pattern_specs(get_default_pattern_specs())
    return _COMPILED_SITE_SPECS


def _find_sites_with_compiled_patterns(mol: Chem.Mol):
    candidates = []
    for spec in _compiled_site_specs():
        matches = mol.GetSubstructMatches(spec["pattern"])
        for match in matches:
            atom_set = set(match)
            hetero_atom_set = {
                atom_idx
                for atom_idx in match
                if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() not in {1, 6}
            }
            specificity = (len(atom_set) * 10000) + (spec["smarts_atoms"] * 100) + spec["priority"]
            candidates.append(
                {
                    "type": spec["site_type"],
                    "label": spec["label"],
                    "smarts": spec["smarts"],
                    "atoms": tuple(match),
                    "atom_set": atom_set,
                    "hetero_atom_set": hetero_atom_set,
                    "priority": spec["priority"],
                    "smarts_atoms": spec["smarts_atoms"],
                    "specificity": specificity,
                    "family": spec.get("family", spec["label"]),
                }
            )
    return candidates


def _coerce_atom_index(value: object) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _sites_matching_atom_index(resolved_sites, atom_index: Optional[int]):
    if atom_index is None:
        return []

    candidate_indices = [atom_index]
    if atom_index > 0:
        candidate_indices.append(atom_index - 1)

    for candidate_index in candidate_indices:
        matches = [site for site in resolved_sites if candidate_index in site.get("atom_set", set())]
        if matches:
            return sorted(matches, key=maximal_site_rank_key, reverse=True)
    return []


def _best_site_for_label(resolved_sites, label: str):
    if not label:
        return None
    matches = [site for site in resolved_sites if str(site.get("label")) == label]
    if not matches:
        return None
    return sorted(matches, key=maximal_site_rank_key, reverse=True)[0]


def _select_site(resolved_sites, label: str, atom_index: Optional[int]):
    atom_matches = _sites_matching_atom_index(resolved_sites, atom_index)
    if atom_matches:
        label_matches = [site for site in atom_matches if str(site.get("label")) == label]
        if label_matches:
            return label_matches[0]
        return atom_matches[0]

    label_match = _best_site_for_label(resolved_sites, label)
    if label_match is not None:
        return label_match

    if len(resolved_sites) == 1:
        return resolved_sites[0]

    if resolved_sites:
        return sorted(resolved_sites, key=maximal_site_rank_key, reverse=True)[0]

    return None


def _site_local_atom_indices(mol: Chem.Mol, distance_matrix, source_atoms, radius: int):
    if not source_atoms:
        return set()
    return {
        atom_idx
        for atom_idx in range(mol.GetNumAtoms())
        if min(float(distance_matrix[src, atom_idx]) for src in source_atoms) <= float(radius)
    }


def _site_numeric_features(mol: Chem.Mol, distance_matrix, source_atoms, radius: int) -> Dict[str, float]:
    if not source_atoms:
        return {
            "site_match_found": 0.0,
            "site_atom_count": 0.0,
            "site_hetero_atom_count": 0.0,
            "site_aromatic_atom_count": 0.0,
            "site_ring_atom_count": 0.0,
            "site_formal_charge": 0.0,
            "site_local_atom_count": 0.0,
            "site_local_hetero_atom_count": 0.0,
            "site_local_aromatic_atom_count": 0.0,
            "site_local_ring_atom_count": 0.0,
            "site_local_formal_charge": 0.0,
        }

    local_atoms = _site_local_atom_indices(mol, distance_matrix, source_atoms, radius=radius)

    def _atom_counts(atom_indices):
        hetero = 0
        aromatic = 0
        ring = 0
        formal_charge = 0.0
        for atom_idx in atom_indices:
            atom = mol.GetAtomWithIdx(atom_idx)
            hetero += int(atom.GetAtomicNum() not in {1, 6})
            aromatic += int(atom.GetIsAromatic())
            ring += int(atom.IsInRing())
            formal_charge += float(atom.GetFormalCharge())
        return hetero, aromatic, ring, formal_charge

    site_hetero, site_aromatic, site_ring, site_charge = _atom_counts(source_atoms)
    local_hetero, local_aromatic, local_ring, local_charge = _atom_counts(local_atoms)

    return {
        "site_match_found": 1.0,
        "site_atom_count": float(len(source_atoms)),
        "site_hetero_atom_count": float(site_hetero),
        "site_aromatic_atom_count": float(site_aromatic),
        "site_ring_atom_count": float(site_ring),
        "site_formal_charge": float(site_charge),
        "site_local_atom_count": float(len(local_atoms)),
        "site_local_hetero_atom_count": float(local_hetero),
        "site_local_aromatic_atom_count": float(local_aromatic),
        "site_local_ring_atom_count": float(local_ring),
        "site_local_formal_charge": float(local_charge),
    }


def _functional_group_context_features(resolved_sites, distance_matrix, source_atoms, radius: int) -> Dict[str, float]:
    """Explicit detector counts complement fingerprints with interpretable context."""
    result = {
        **{f"fg_global_count__{label}": 0.0 for label in _CONTEXT_GROUP_LABELS},
        **{f"fg_local_count__{label}": 0.0 for label in _CONTEXT_GROUP_LABELS},
    }
    for site in resolved_sites:
        label = str(site.get("label", ""))
        global_key = f"fg_global_count__{label}"
        if global_key not in result:
            continue
        result[global_key] += 1.0
        site_atoms = set(site.get("atom_set", set()))
        if source_atoms and site_atoms and min(
            float(distance_matrix[source_atom, context_atom])
            for source_atom in source_atoms
            for context_atom in site_atoms
        ) <= float(radius):
            result[f"fg_local_count__{label}"] += 1.0
    return result


def _local_fragment_smiles(mol: Chem.Mol, distance_matrix, source_atoms, radius: int) -> str:
    if not source_atoms:
        return ""

    local_atoms = sorted(_site_local_atom_indices(mol, distance_matrix, source_atoms, radius=radius))
    if not local_atoms:
        return ""

    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=local_atoms,
        canonical=True,
        isomericSmiles=True,
    )


def _local_fragment_feature_payload(fragment_smiles: str, nbits: int, radius: int) -> Tuple[Dict[str, float], np.ndarray]:
    if not fragment_smiles:
        return _zero_local_descriptors(), np.zeros(nbits, dtype=np.float32)

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    try:
        desc = {
            f"site_{key}": float(value)
            for key, value in molecule_descriptors(fragment_smiles).items()
        }
        fp = morgan_bits(fragment_smiles, nbits=nbits, radius=radius)
    finally:
        RDLogger.EnableLog("rdApp.error")
        RDLogger.EnableLog("rdApp.warning")

    return desc, fp


def _zero_local_descriptors() -> Dict[str, float]:
    base = molecule_descriptors("")
    return {f"site_{key}": float(value) for key, value in base.items()}


def _local_site_features(
    df: pd.DataFrame,
    nbits: int,
    radius: int,
    label_col: str = "final_group_label",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    zero_desc = _zero_local_descriptors()
    zero_mol_geom = zero_molecule_geometry_features(prefix="mol_geom")
    zero_site_geom = zero_site_geometry_features(prefix="site_geom")
    zero_numeric = {
        "site_match_found": 0.0,
        "site_atom_count": 0.0,
        "site_hetero_atom_count": 0.0,
        "site_aromatic_atom_count": 0.0,
        "site_ring_atom_count": 0.0,
        "site_formal_charge": 0.0,
        "site_local_atom_count": 0.0,
        "site_local_hetero_atom_count": 0.0,
        "site_local_aromatic_atom_count": 0.0,
        "site_local_ring_atom_count": 0.0,
        "site_local_formal_charge": 0.0,
        **{f"fg_global_count__{label}": 0.0 for label in _CONTEXT_GROUP_LABELS},
        **{f"fg_local_count__{label}": 0.0 for label in _CONTEXT_GROUP_LABELS},
    }
    zero_fp = np.zeros(nbits, dtype=np.float32)

    atom_index_series = (
        df["atom_index_raw"]
        if "atom_index_raw" in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )

    per_smiles_cache: Dict[str, Dict[str, object]] = {}
    per_site_cache: Dict[Tuple[str, str, Optional[int]], Tuple[Dict[str, float], np.ndarray]] = {}

    def _smiles_payload(smiles: str) -> Dict[str, object]:
        if smiles in per_smiles_cache:
            return per_smiles_cache[smiles]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            payload = {
                "mol": None,
                "mol_3d": None,
                "mol_geom": zero_mol_geom,
                "resolved": [],
                "distance": None,
            }
            per_smiles_cache[smiles] = payload
            return payload

        candidates = _find_sites_with_compiled_patterns(mol)
        resolved, _ = resolve_overlapping_sites(candidates, overlap_threshold=0.75)
        mol_3d = embed_molecule_3d(mol)
        payload = {
            "mol": mol,
            "mol_3d": mol_3d,
            "mol_geom": molecule_geometry_features(mol_3d, prefix="mol_geom"),
            "resolved": resolved,
            "distance": Chem.GetDistanceMatrix(mol),
        }
        per_smiles_cache[smiles] = payload
        return payload

    numeric_rows = []
    fp_rows = []

    for smiles, label, atom_index_raw in zip(df["smiles"], df[label_col], atom_index_series):
        smiles_text = "" if pd.isna(smiles) else str(smiles).strip()
        label_text = "" if pd.isna(label) else str(label).strip()
        atom_index = _coerce_atom_index(atom_index_raw)
        cache_key = (smiles_text, label_text, atom_index)

        cached = per_site_cache.get(cache_key)
        if cached is None:
            payload = _smiles_payload(smiles_text)
            mol = payload["mol"]
            mol_3d = payload["mol_3d"]
            mol_geom = payload["mol_geom"]
            resolved = payload["resolved"]
            distance = payload["distance"]

            if mol is None or distance is None:
                numeric_payload = {**zero_numeric, **zero_mol_geom, **zero_site_geom, **zero_desc}
                fp_payload = zero_fp
            else:
                selected_site = _select_site(resolved, label_text, atom_index)
                source_atoms = set(selected_site.get("atom_set", set())) if selected_site is not None else set()
                local_atoms = _site_local_atom_indices(mol, distance, source_atoms, radius=radius)
                numeric_payload = _site_numeric_features(mol, distance, source_atoms, radius=radius)
                context_features = _functional_group_context_features(
                    resolved, distance, source_atoms, radius=radius
                )
                fragment_smiles = _local_fragment_smiles(mol, distance, source_atoms, radius=radius)
                local_desc, fp_payload = _local_fragment_feature_payload(
                    fragment_smiles,
                    nbits=nbits,
                    radius=radius,
                )
                site_geom = site_geometry_features(mol_3d, source_atoms, local_atoms, prefix="site_geom")
                numeric_payload = {
                    **numeric_payload,
                    **context_features,
                    **mol_geom,
                    **site_geom,
                    **local_desc,
                }

            cached = (numeric_payload, fp_payload)
            per_site_cache[cache_key] = cached

        numeric_rows.append(cached[0])
        fp_rows.append(cached[1])

    numeric_df = pd.DataFrame(numeric_rows).fillna(0.0)
    fp_df = pd.DataFrame(
        np.vstack(fp_rows) if fp_rows else np.zeros((0, nbits), dtype=np.float32),
        columns=[f"site_fp_{i}" for i in range(nbits)],
    )
    return numeric_df, fp_df


def build_intrinsic_features(
    df: pd.DataFrame,
    nbits: int,
    radius: int,
    label_col: str = "final_group_label",
) -> pd.DataFrame:
    smiles_unique = df["smiles"].dropna().unique()
    smiles_to_desc = {smi: molecule_descriptors(smi) for smi in smiles_unique}
    smiles_to_fp = {smi: morgan_bits(smi, nbits=nbits, radius=radius) for smi in smiles_unique}

    desc_df = pd.DataFrame(df["smiles"].map(smiles_to_desc).tolist())
    fp_matrix = np.vstack([smiles_to_fp.get(smi, np.zeros(nbits, dtype=np.float32)) for smi in df["smiles"]])
    fp_df = pd.DataFrame(fp_matrix, columns=[f"fp_{i}" for i in range(nbits)])
    site_df, site_fp_df = _local_site_features(df, nbits=nbits, radius=radius, label_col=label_col)

    raw_labels = df[label_col].astype(str)
    training_labels = raw_labels.map(training_group_label)
    group_df = pd.get_dummies(training_labels, prefix="group")
    type_df = pd.get_dummies(df["pka_type_canonical"].fillna("unknown"), prefix="pka_type")

    families = (
        df["site_family"].astype(str)
        if "site_family" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    priors = [
        reference_prior(label, family)
        for label, family in zip(raw_labels, families)
    ]
    reference_df = pd.DataFrame({
        "stage1_reference_pka": [prior.pka if prior is not None else 0.0 for prior in priors],
        "stage1_reference_uncertainty": [
            prior.uncertainty if prior is not None else 0.0 for prior in priors
        ],
        "stage1_reference_prior_present": [
            float(prior is not None) for prior in priors
        ],
    })

    X = pd.concat([
        desc_df, site_df, reference_df, group_df, type_df, fp_df, site_fp_df
    ], axis=1)
    return X


def _eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _fit_and_eval(
    frame: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[HistGradientBoostingRegressor, Dict[str, float], pd.DataFrame]:
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=400,
        random_state=17,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    pred = model.predict(X.iloc[test_idx])
    metrics = _eval_regression(y.iloc[test_idx].to_numpy(), pred)

    eval_df = frame.iloc[test_idx].copy()
    eval_df["pred_intrinsic_pka"] = pred
    eval_df["abs_error"] = np.abs(eval_df["pred_intrinsic_pka"] - eval_df["pka_value"])
    return model, metrics, eval_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: train intrinsic single-group pKa model.")
    parser.add_argument("--assignments", default=DEFAULT_CURATED_ASSIGNMENTS_PATH)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--out-dir",
        default="data/processed/ml_models_experimental_only/stage1_intrinsic",
    )
    parser.add_argument("--fp-bits", type=int, default=512)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument(
        "--measurement-method",
        default="experimental",
        help="Only train on this pka_source_method; use an empty value to disable filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    assignments_path, _ = ensure_curated_training_data(
        assignments_path=args.assignments,
        raw_dir=getattr(args, "raw_dir", DEFAULT_RAW_DIR),
    )
    measurement_method = str(getattr(args, "measurement_method", "experimental")).strip() or None
    single_df = load_single_group_rows(
        assignments_path,
        measurement_method=measurement_method,
    )
    if single_df.empty:
        raise RuntimeError("No single-group rows available for Stage 1 training.")

    X = build_intrinsic_features(single_df, nbits=args.fp_bits, radius=args.fp_radius)
    y = single_df["pka_value"].astype(float)

    r_train, r_test = random_molecule_split(single_df, test_frac=args.test_frac)
    s_train, s_test = scaffold_molecule_split(single_df, test_frac=args.test_frac)

    random_model, random_metrics, random_eval = _fit_and_eval(single_df, X, y, r_train, r_test)
    scaffold_model, scaffold_metrics, scaffold_eval = _fit_and_eval(single_df, X, y, s_train, s_test)

    bundle = {
        "model": scaffold_model,
        "feature_columns": list(X.columns),
        "fp_bits": int(args.fp_bits),
        "fp_radius": int(args.fp_radius),
        "training_measurement_method": measurement_method,
        "training_rows": int(len(single_df)),
    }
    with open(os.path.join(args.out_dir, "stage1_intrinsic_model.pkl"), "wb") as handle:
        pickle.dump(bundle, handle)

    random_eval.to_csv(os.path.join(args.out_dir, "random_eval_intrinsic.csv"), index=False)
    scaffold_eval.to_csv(os.path.join(args.out_dir, "scaffold_eval_intrinsic.csv"), index=False)

    split_manifest(single_df, r_train, r_test).to_csv(
        os.path.join(args.out_dir, "random_split_manifest.csv"),
        index=False,
    )
    split_manifest(single_df, s_train, s_test).to_csv(
        os.path.join(args.out_dir, "scaffold_split_manifest.csv"),
        index=False,
    )

    report = {
        "rows": int(len(single_df)),
        "molecules": int(single_df["molecule_key"].nunique()),
        "measurement_method": measurement_method,
        "random": random_metrics,
        "scaffold": scaffold_metrics,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Saved Stage 1 artifacts to", args.out_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
