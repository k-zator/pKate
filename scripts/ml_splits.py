from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem # type: ignore
from rdkit.Chem.Scaffolds import MurckoScaffold # type: ignore


def _murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def random_molecule_split(
    df: pd.DataFrame,
    molecule_key_col: str = "molecule_key",
    test_frac: float = 0.2,
    seed: int = 17,
) -> Tuple[np.ndarray, np.ndarray]:
    keys = df[molecule_key_col].dropna().unique()
    rng = np.random.default_rng(seed)
    shuffled = np.array(keys, copy=True)
    rng.shuffle(shuffled)

    n_test = max(1, int(len(shuffled) * test_frac))
    test_keys = set(shuffled[:n_test])
    is_test = df[molecule_key_col].isin(test_keys).to_numpy()
    test_idx = np.where(is_test)[0]
    train_idx = np.where(~is_test)[0]
    return train_idx, test_idx


def scaffold_molecule_split(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    molecule_key_col: str = "molecule_key",
    test_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    per_molecule = df[[molecule_key_col, smiles_col]].drop_duplicates(subset=[molecule_key_col]).copy()
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
    return train_idx, test_idx


def split_manifest(
    df: pd.DataFrame,
    train_idx: Iterable[int],
    test_idx: Iterable[int],
) -> pd.DataFrame:
    manifest = df[["molecule_key", "smiles"]].copy()
    manifest["split"] = "unused"
    manifest.loc[list(train_idx), "split"] = "train"
    manifest.loc[list(test_idx), "split"] = "test"
    return manifest.drop_duplicates(subset=["molecule_key", "smiles", "split"]) 
