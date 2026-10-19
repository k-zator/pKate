from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from stage1_canonical_data import scaffold_group_key


def _murcko_scaffold(smiles: str) -> str:
    """Use the canonical scaffold grouping policy shared with Stage 1.

    Ring-containing analogues share a Murcko scaffold.  Acyclic compounds are
    grouped by canonical structure rather than all collapsing into the empty
    Murcko scaffold, and malformed strings receive stable hashed groups.
    """
    return scaffold_group_key(str(smiles))


def _validate_split_input(
    df: pd.DataFrame,
    molecule_key_col: str,
    test_frac: float,
) -> np.ndarray:
    """Return unique molecule keys after validating a usable split domain."""
    if molecule_key_col not in df.columns:
        raise ValueError(f"Missing molecule key column: {molecule_key_col}")
    if not 0.0 < float(test_frac) < 1.0:
        raise ValueError("test_frac must be strictly between zero and one")
    if df[molecule_key_col].isna().any():
        raise ValueError("Molecule keys must be non-missing before splitting")
    keys = df[molecule_key_col].unique()
    if len(keys) < 2:
        raise ValueError("At least two distinct molecules are required for a split")
    return np.asarray(keys)


def random_molecule_split(
    df: pd.DataFrame,
    molecule_key_col: str = "molecule_key",
    test_frac: float = 0.2,
    seed: int = 17,
) -> Tuple[np.ndarray, np.ndarray]:
    keys = _validate_split_input(df, molecule_key_col, test_frac)
    rng = np.random.default_rng(seed)
    shuffled = np.array(keys, copy=True)
    rng.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_frac))
    test_keys = set(shuffled[:n_test])
    is_test = df[molecule_key_col].isin(test_keys).to_numpy()
    test_idx = np.where(is_test)[0]
    train_idx = np.where(~is_test)[0]
    if not len(train_idx) or not len(test_idx):
        raise ValueError("Random molecule split produced an empty partition")
    return train_idx, test_idx


def scaffold_molecule_split(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    molecule_key_col: str = "molecule_key",
    test_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    _validate_split_input(df, molecule_key_col, test_frac)
    if smiles_col not in df.columns:
        raise ValueError(f"Missing SMILES column: {smiles_col}")
    if df[smiles_col].isna().any():
        raise ValueError("SMILES must be non-missing before scaffold splitting")
    smiles_per_key = df.groupby(molecule_key_col)[smiles_col].nunique(dropna=False)
    inconsistent = smiles_per_key[smiles_per_key != 1]
    if not inconsistent.empty:
        raise ValueError(
            "Each molecule key must map to exactly one SMILES; inconsistent keys: "
            + ", ".join(map(str, inconsistent.index[:5]))
        )
    per_molecule = df[[molecule_key_col, smiles_col]].drop_duplicates(
        subset=[molecule_key_col]
    ).copy()
    per_molecule["scaffold"] = per_molecule[smiles_col].map(_murcko_scaffold)

    scaffold_groups = (
        per_molecule.groupby("scaffold")[molecule_key_col]
        .apply(list)
        # Sort ASCENDING by size: rarest (smallest) scaffolds go to test
        # so we evaluate true out-of-distribution generalization.
        .sort_values(key=lambda x: x.map(len), ascending=True)
    )

    total = len(per_molecule)
    target_test = max(1, int(total * test_frac))
    selected_test_keys = []
    running = 0
    for keys in scaffold_groups:
        if running >= target_test:
            break
        selected_test_keys.extend(keys)
        running += len(keys)

    test_key_set = set(selected_test_keys)
    is_test = df[molecule_key_col].isin(test_key_set).to_numpy()
    test_idx = np.where(is_test)[0]
    train_idx = np.where(~is_test)[0]
    if not len(train_idx) or not len(test_idx):
        raise ValueError(
            "Scaffold split produced an empty partition; at least two independent "
            "scaffold groups are required"
        )
    return train_idx, test_idx


def split_manifest(
    df: pd.DataFrame,
    train_idx: Iterable[int],
    test_idx: Iterable[int],
) -> pd.DataFrame:
    required = {"molecule_key", "smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split manifest missing columns: {sorted(missing)}")
    manifest = df[["molecule_key", "smiles"]].reset_index(drop=True).copy()
    manifest["split"] = "unused"
    train_positions = np.asarray(list(train_idx), dtype=int)
    test_positions = np.asarray(list(test_idx), dtype=int)
    if np.intersect1d(train_positions, test_positions).size:
        raise ValueError("Train and test row indices overlap")
    all_positions = np.concatenate([train_positions, test_positions])
    if len(all_positions) and (
        int(all_positions.min()) < 0 or int(all_positions.max()) >= len(manifest)
    ):
        raise IndexError("Split row index is outside the dataset")
    split_col = manifest.columns.get_loc("split")
    manifest.iloc[train_positions, split_col] = "train"
    manifest.iloc[test_positions, split_col] = "test"
    assigned = manifest[manifest["split"] != "unused"]
    leaked = assigned.groupby("molecule_key")["split"].nunique()
    if (leaked > 1).any():
        bad = leaked[leaked > 1].index
        raise ValueError(
            "A molecule appears in both train and test: "
            + ", ".join(map(str, bad[:5]))
        )
    return manifest.drop_duplicates(subset=["molecule_key", "smiles", "split"])
