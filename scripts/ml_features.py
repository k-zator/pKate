from typing import Dict, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem # type: ignore
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors # type: ignore
from rdkit.Chem import rdFingerprintGenerator # type: ignore


def _safe_float(value: float) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return float(value)


def molecule_descriptors(smiles: str) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "desc_mol_wt": 0.0,
            "desc_logp": 0.0,
            "desc_tpsa": 0.0,
            "desc_hbd": 0.0,
            "desc_hba": 0.0,
            "desc_rot_bonds": 0.0,
            "desc_ring_count": 0.0,
            "desc_formal_charge": 0.0,
            "desc_heavy_atoms": 0.0,
        }

    return {
        "desc_mol_wt": _safe_float(Descriptors.MolWt(mol)),
        "desc_logp": _safe_float(Crippen.MolLogP(mol)),
        "desc_tpsa": _safe_float(rdMolDescriptors.CalcTPSA(mol)),
        "desc_hbd": _safe_float(Lipinski.NumHDonors(mol)),
        "desc_hba": _safe_float(Lipinski.NumHAcceptors(mol)),
        "desc_rot_bonds": _safe_float(Lipinski.NumRotatableBonds(mol)),
        "desc_ring_count": _safe_float(rdMolDescriptors.CalcNumRings(mol)),
        "desc_formal_charge": _safe_float(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
        "desc_heavy_atoms": _safe_float(mol.GetNumHeavyAtoms()),
    }


def morgan_bits(smiles: str, nbits: int = 512, radius: int = 2) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((nbits,), dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


def build_feature_matrix(
    candidate_df: pd.DataFrame,
    nbits: int = 512,
    radius: int = 2,
    include_observed_pka: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    required = {
        "smiles",
        "candidate_label",
        "resolved_group_count",
        "candidate_uniqueness",
        "candidate_prior_mean_pka",
        "candidate_prior_median_pka",
        "candidate_prior_iqr_pka",
        "candidate_prior_count",
        "is_true_site",
    }
    if include_observed_pka:
        required.add("pka_value")
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature build: {sorted(missing)}")

    smiles_to_desc = {
        smi: molecule_descriptors(smi)
        for smi in candidate_df["smiles"].dropna().unique()
    }
    smiles_to_fp = {
        smi: morgan_bits(smi, nbits=nbits, radius=radius)
        for smi in candidate_df["smiles"].dropna().unique()
    }

    descriptor_df = pd.DataFrame(candidate_df["smiles"].map(smiles_to_desc).tolist())

    fp_rows = [smiles_to_fp.get(smi, np.zeros(nbits, dtype=np.float32)) for smi in candidate_df["smiles"]]
    fp_matrix = np.vstack(fp_rows)
    fp_cols = [f"fp_{i}" for i in range(nbits)]
    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols)

    candidate_dummies = pd.get_dummies(candidate_df["candidate_label"], prefix="cand")
    type_dummies = pd.get_dummies(candidate_df.get("pka_type_canonical", "unknown"), prefix="pka_type")

    numeric_cols = [
        "resolved_group_count",
        "candidate_uniqueness",
        "candidate_prior_mean_pka",
        "candidate_prior_median_pka",
        "candidate_prior_iqr_pka",
        "candidate_prior_count",
    ]
    for predicted_col in ["intrinsic_pred_pka", "pred_delta_pka", "pred_effective_pka"]:
        if predicted_col in candidate_df.columns:
            numeric_cols.append(predicted_col)
    if include_observed_pka:
        numeric_cols.append("pka_value")

    numeric = pd.DataFrame(
        {
            col: pd.to_numeric(candidate_df[col], errors="coerce").fillna(0.0)
            for col in numeric_cols
        }
    ).reset_index(drop=True)

    X = pd.concat([numeric, descriptor_df, candidate_dummies, type_dummies, fp_df], axis=1)
    y = candidate_df["is_true_site"].astype(int)
    return X, y
