"""
ML Feature Engineering Tests
==============================
Tests for molecule_descriptors, morgan_bits, and build_feature_matrix.
"""

import numpy as np
import pandas as pd
import pytest

from ml_features import molecule_descriptors, morgan_bits, build_feature_matrix


# ===================================================================
# 1. molecule_descriptors
# ===================================================================

DESCRIPTOR_KEYS = {
    "desc_mol_wt",
    "desc_logp",
    "desc_tpsa",
    "desc_hbd",
    "desc_hba",
    "desc_rot_bonds",
    "desc_ring_count",
    "desc_formal_charge",
    "desc_heavy_atoms",
}


def test_molecule_descriptors_keys():
    desc = molecule_descriptors("CC(=O)O")
    assert set(desc.keys()) == DESCRIPTOR_KEYS


def test_molecule_descriptors_types():
    desc = molecule_descriptors("CC(=O)O")
    for key, val in desc.items():
        assert isinstance(val, float), f"{key} should be float, got {type(val)}"


def test_molecule_descriptors_acetic_acid_values():
    desc = molecule_descriptors("CC(=O)O")
    assert desc["desc_mol_wt"] > 50, "Acetic acid mol wt should be ~60"
    assert desc["desc_heavy_atoms"] == 4, "Acetic acid has 4 heavy atoms"
    assert desc["desc_ring_count"] == 0, "Acetic acid has no rings"
    assert desc["desc_formal_charge"] == 0, "Neutral acetic acid has charge 0"
    assert desc["desc_hbd"] >= 1, "Acetic acid has at least 1 H-bond donor (OH)"
    assert desc["desc_hba"] >= 1, "Acetic acid has at least 1 H-bond acceptor"


def test_molecule_descriptors_charged():
    desc = molecule_descriptors("CC(=O)[O-]")
    assert desc["desc_formal_charge"] == -1, "Acetate has charge -1"


def test_molecule_descriptors_invalid_smiles():
    desc = molecule_descriptors("INVALID_SMILES_XYZ")
    assert set(desc.keys()) == DESCRIPTOR_KEYS
    assert all(v == 0.0 for v in desc.values()), "Invalid SMILES should return all-zero descriptors"


def test_molecule_descriptors_aromatic():
    desc = molecule_descriptors("c1ccccc1")
    assert desc["desc_ring_count"] == 1, "Benzene has 1 ring"
    assert desc["desc_heavy_atoms"] == 6


# ===================================================================
# 2. morgan_bits
# ===================================================================

def test_morgan_bits_shape_default():
    fp = morgan_bits("CC(=O)O")
    assert fp.shape == (512,), f"Expected (512,), got {fp.shape}"


def test_morgan_bits_shape_custom():
    fp = morgan_bits("CC(=O)O", nbits=1024, radius=3)
    assert fp.shape == (1024,)


def test_morgan_bits_binary():
    fp = morgan_bits("CC(=O)O")
    assert set(np.unique(fp)).issubset({0.0, 1.0}), "Fingerprint must be binary"


def test_morgan_bits_has_on_bits():
    fp = morgan_bits("CC(=O)O")
    assert fp.sum() > 0, "Fingerprint must have at least some bits set"


def test_morgan_bits_different_radius():
    # Use a larger molecule so the radii make a difference
    smi = "c1ccc(CC(=O)O)cc1"  # phenylacetic acid
    fp1 = morgan_bits(smi, radius=1)
    fp2 = morgan_bits(smi, radius=3)
    # Different radii should produce different fingerprints
    assert not np.array_equal(fp1, fp2), "Radius 1 vs 3 should differ for phenylacetic acid"


def test_morgan_bits_invalid_smiles():
    fp = morgan_bits("INVALID")
    assert fp.shape == (512,)
    assert fp.sum() == 0, "Invalid SMILES should return all-zero fingerprint"


def test_morgan_bits_different_molecules():
    fp_a = morgan_bits("CC(=O)O")  # acetic acid
    fp_b = morgan_bits("c1ccccc1")  # benzene
    assert not np.array_equal(fp_a, fp_b), "Different molecules should produce different fingerprints"


# ===================================================================
# 3. build_feature_matrix
# ===================================================================

def _make_candidate_df(n_rows=5):
    """Create a minimal candidate DataFrame with required columns."""
    return pd.DataFrame({
        "smiles": ["CC(=O)O", "CN", "Oc1ccccc1", "c1ccncc1", "CS"] * (n_rows // 5 + 1),
        "candidate_label": ["carboxylic_acid", "primary_amine", "phenol", "pyridine", "thiol"] * (n_rows // 5 + 1),
        "pka_type_canonical": ["acidic", "basic", "acidic", "basic", "acidic"] * (n_rows // 5 + 1),
        "resolved_group_count": [1, 1, 1, 1, 1] * (n_rows // 5 + 1),
        "candidate_uniqueness": [0.5, 0.5, 0.5, 0.5, 0.5] * (n_rows // 5 + 1),
        "candidate_prior_mean_pka": [4.0, 10.0, 10.0, 5.0, 10.0] * (n_rows // 5 + 1),
        "candidate_prior_median_pka": [4.0, 10.0, 10.0, 5.0, 10.0] * (n_rows // 5 + 1),
        "candidate_prior_iqr_pka": [1.0, 2.0, 1.0, 1.5, 2.0] * (n_rows // 5 + 1),
        "candidate_prior_count": [50, 30, 20, 40, 10] * (n_rows // 5 + 1),
        "is_true_site": [1, 0, 1, 0, 1] * (n_rows // 5 + 1),
    }).head(n_rows).reset_index(drop=True)


def test_build_feature_matrix_output_shape():
    df = _make_candidate_df(10)
    X, y = build_feature_matrix(df)
    assert X.shape[0] == 10, f"Expected 10 rows, got {X.shape[0]}"
    assert X.shape[1] > 520, "Should have descriptors + FP + dummies (> 520 total columns)"
    assert y.shape == (10,)


def test_build_feature_matrix_no_nans():
    df = _make_candidate_df(5)
    X, y = build_feature_matrix(df)
    assert not X.isnull().any().any(), "Feature matrix should have no NaN values"


def test_build_feature_matrix_y_is_binary():
    df = _make_candidate_df(5)
    X, y = build_feature_matrix(df)
    assert set(y.unique()).issubset({0, 1}), "Target must be binary (0 or 1)"


def test_build_feature_matrix_has_descriptor_cols():
    df = _make_candidate_df(5)
    X, _ = build_feature_matrix(df)
    for key in DESCRIPTOR_KEYS:
        assert key in X.columns, f"Feature matrix missing descriptor column: {key}"


def test_build_feature_matrix_has_fp_cols():
    df = _make_candidate_df(5)
    X, _ = build_feature_matrix(df)
    assert "fp_0" in X.columns
    assert "fp_511" in X.columns


def test_build_feature_matrix_has_candidate_dummies():
    df = _make_candidate_df(5)
    X, _ = build_feature_matrix(df)
    cand_cols = [c for c in X.columns if c.startswith("cand_")]
    assert len(cand_cols) > 0, "Should have candidate label dummies"


def test_build_feature_matrix_with_observed_pka():
    df = _make_candidate_df(5)
    df["pka_value"] = [4.0, 10.0, 10.0, 5.0, 10.0]
    X, _ = build_feature_matrix(df, include_observed_pka=True)
    assert "pka_value" in X.columns


def test_build_feature_matrix_with_predicted_pka_features():
    df = _make_candidate_df(5)
    df["intrinsic_pred_pka"] = [4.2, 9.8, 9.7, 5.1, 9.9]
    df["pred_delta_pka"] = [0.0, 0.4, -0.2, 0.1, -0.1]
    df["pred_effective_pka"] = df["intrinsic_pred_pka"] + df["pred_delta_pka"]
    X, _ = build_feature_matrix(df)
    assert "intrinsic_pred_pka" in X.columns
    assert "pred_delta_pka" in X.columns
    assert "pred_effective_pka" in X.columns


def test_build_feature_matrix_missing_columns():
    df = pd.DataFrame({"smiles": ["CC(=O)O"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        build_feature_matrix(df)


def test_build_feature_matrix_custom_fp_params():
    df = _make_candidate_df(3)
    X, _ = build_feature_matrix(df, nbits=256, radius=1)
    fp_cols = [c for c in X.columns if c.startswith("fp_")]
    assert len(fp_cols) == 256
