"""
Ionization Context Feature Tests
==================================
Tests for the ionization context feature engineering pipeline that
produces the 24 ``ion_ctx_*`` columns used by Stage 2 and form heads.
"""

import numpy as np
import pandas as pd
import pytest

from ionization_context_features import (
    build_ionization_context_frame,
    build_carboxyl_form_feature_frame,
    AMMONIUM_LIKE_LABELS,
    AMINE_LIKE_LABELS,
)

# ===================================================================
# Helpers
# ===================================================================

ION_CTX_COLUMNS = [
    "ion_ctx_total_groups",
    "ion_ctx_acidic_groups",
    "ion_ctx_basic_groups",
    "ion_ctx_carboxyl_groups",
    "ion_ctx_amide_like_groups",
    "ion_ctx_amine_like_groups",
    "ion_ctx_ammonium_like_groups",
    "ion_ctx_same_family_groups",
    "ion_ctx_other_family_groups",
    "ion_ctx_candidate_is_acidic_family",
    "ion_ctx_candidate_is_basic_family",
    "ion_ctx_candidate_is_carboxyl",
    "ion_ctx_has_carboxyl_and_basic",
    "ion_ctx_charge_proxy",
    "ion_ctx_local_site_atom_count",
    "ion_ctx_local_site_charge_sum",
    "ion_ctx_local_charge_sum_r2",
    "ion_ctx_local_min_dist_pos_n",
    "ion_ctx_local_pos_n_within_3",
    "ion_ctx_local_pos_n_within_5",
    "ion_ctx_local_min_dist_basic",
    "ion_ctx_local_basic_within_4",
    "ion_ctx_local_min_dist_amide",
    "ion_ctx_local_amide_within_4",
]


def _make_df(smiles, resolved_groups, candidate_label):
    """Create a 1-row DataFrame for ionization context testing."""
    return pd.DataFrame({
        "smiles": [smiles],
        "resolved_groups": [resolved_groups],
        "candidate_label": [candidate_label],
    })


def _ctx(smiles, resolved_groups, candidate_label):
    """Compute ionization context for a single molecule."""
    df = _make_df(smiles, resolved_groups, candidate_label)
    result = build_ionization_context_frame(df)
    return result.iloc[0]


# ===================================================================
# 1. Output shape and columns
# ===================================================================

def test_context_frame_has_all_columns():
    df = _make_df("CC(=O)O", "carboxylic_acid", "carboxylic_acid")
    ctx = build_ionization_context_frame(df)
    assert ctx.shape[0] == 1
    for col in ION_CTX_COLUMNS:
        assert col in ctx.columns, f"Missing column: {col}"


def test_context_frame_no_nans():
    df = _make_df("CC(=O)O", "carboxylic_acid", "carboxylic_acid")
    ctx = build_ionization_context_frame(df)
    assert not ctx.isnull().any().any(), "Context frame should have no NaN values"


# ===================================================================
# 2. Single-group molecules
# ===================================================================

def test_single_group_acetic_acid():
    row = _ctx("CC(=O)O", "carboxylic_acid", "carboxylic_acid")
    assert row["ion_ctx_total_groups"] == 1
    assert row["ion_ctx_acidic_groups"] == 1
    assert row["ion_ctx_basic_groups"] == 0
    assert row["ion_ctx_carboxyl_groups"] == 1
    assert row["ion_ctx_same_family_groups"] == 1
    assert row["ion_ctx_other_family_groups"] == 0
    assert row["ion_ctx_candidate_is_acidic_family"] == 1.0
    assert row["ion_ctx_candidate_is_basic_family"] == 0.0
    assert row["ion_ctx_candidate_is_carboxyl"] == 1.0
    assert row["ion_ctx_has_carboxyl_and_basic"] == 0.0


def test_single_group_amine():
    row = _ctx("CN", "primary_amine", "primary_amine")
    assert row["ion_ctx_total_groups"] == 1
    assert row["ion_ctx_basic_groups"] == 1  # primary_amine's family 'amine' is in BASIC_FAMILIES
    assert row["ion_ctx_amine_like_groups"] == 1
    assert row["ion_ctx_candidate_is_basic_family"] == 1.0


# ===================================================================
# 3. Multi-group: glycine (carboxyl + ammonium)
# ===================================================================

def test_glycine_context():
    """Glycine ([NH3+]CC(=O)[O-]) has carboxylate + ammonium."""
    row = _ctx(
        "[NH3+]CC(=O)[O-]",
        "carboxylate|primary_ammonium",
        "carboxylate",
    )
    assert row["ion_ctx_total_groups"] >= 2
    assert row["ion_ctx_carboxyl_groups"] >= 1
    assert row["ion_ctx_ammonium_like_groups"] >= 1
    assert row["ion_ctx_has_carboxyl_and_basic"] == 1.0
    # charge_proxy = ammonium_like - carboxyl, should be ~0 for glycine
    assert abs(row["ion_ctx_charge_proxy"]) <= 1


def test_glycine_positive_n_distance():
    """In glycine the [NH3+] should be within a few bonds of the carboxylate site."""
    row = _ctx(
        "[NH3+]CC(=O)[O-]",
        "carboxylate|primary_ammonium",
        "carboxylate",
    )
    # min_dist_pos_n should be finite and less than 99 (the sentinel)
    assert row["ion_ctx_local_min_dist_pos_n"] < 99.0
    assert row["ion_ctx_local_pos_n_within_5"] >= 1


# ===================================================================
# 4. Pure carboxyl — no basic context
# ===================================================================

def test_pure_carboxyl_no_basic():
    row = _ctx("CC(=O)O", "carboxylic_acid", "carboxylic_acid")
    assert row["ion_ctx_basic_groups"] == 0
    assert row["ion_ctx_has_carboxyl_and_basic"] == 0.0
    # No positive N present
    assert row["ion_ctx_local_min_dist_pos_n"] == 99.0


# ===================================================================
# 5. Local distance features
# ===================================================================

def test_local_distance_ammonium_present():
    """Molecule with [NH3+], the pos_n distance should be finite."""
    row = _ctx(
        "C[NH3+]",
        "primary_ammonium",
        "primary_ammonium",
    )
    assert row["ion_ctx_local_min_dist_pos_n"] < 99.0


def test_local_distance_no_positive_n():
    """Molecule without any positive N — sentinel value 99."""
    row = _ctx("CC(=O)O", "carboxylic_acid", "carboxylic_acid")
    assert row["ion_ctx_local_min_dist_pos_n"] == 99.0


# ===================================================================
# 6. build_carboxyl_form_feature_frame
# ===================================================================

def test_carboxyl_form_feature_frame_shape():
    df = pd.DataFrame({
        "smiles": ["CC(=O)O", "[NH3+]CC(=O)[O-]"],
        "resolved_groups": ["carboxylic_acid", "carboxylate|primary_ammonium"],
        "candidate_label": ["carboxylic_acid", "carboxylate"],
        "resolved_group_count": [1, 2],
        "pka_type_canonical": ["acidic", "acidic"],
        "molecule_formal_charge": [0, 0],
        "neutral_input_risk": [0, 0],
    })
    result = build_carboxyl_form_feature_frame(df)
    assert result.shape[0] == 2
    # Should have numeric cols + ion_ctx cols
    assert "resolved_group_count" in result.columns
    assert "molecule_formal_charge" in result.columns
    assert "neutral_input_risk" in result.columns
    for col in ION_CTX_COLUMNS:
        assert col in result.columns, f"Missing: {col}"


def test_carboxyl_form_feature_frame_with_pka_type():
    df = pd.DataFrame({
        "smiles": ["CC(=O)O"],
        "resolved_groups": ["carboxylic_acid"],
        "candidate_label": ["carboxylic_acid"],
        "resolved_group_count": [1],
        "pka_type_canonical": ["acidic"],
        "molecule_formal_charge": [0],
        "neutral_input_risk": [0],
    })
    result = build_carboxyl_form_feature_frame(df, include_pka_type=True)
    pka_type_cols = [c for c in result.columns if c.startswith("pka_type_")]
    assert len(pka_type_cols) > 0, "Should have pka_type dummies when include_pka_type=True"


def test_carboxyl_form_feature_frame_no_nans():
    df = pd.DataFrame({
        "smiles": ["CC(=O)O"],
        "resolved_groups": ["carboxylic_acid"],
        "candidate_label": ["carboxylic_acid"],
        "resolved_group_count": [1],
        "pka_type_canonical": ["acidic"],
        "molecule_formal_charge": [0],
        "neutral_input_risk": [0],
    })
    result = build_carboxyl_form_feature_frame(df)
    assert not result.isnull().any().any()


# ===================================================================
# 7. Label sets consistency
# ===================================================================

def test_ammonium_like_labels_not_empty():
    assert len(AMMONIUM_LIKE_LABELS) >= 10


def test_amine_like_labels_not_empty():
    assert len(AMINE_LIKE_LABELS) >= 10


def test_ammonium_and_amine_disjoint():
    overlap = AMMONIUM_LIKE_LABELS & AMINE_LIKE_LABELS
    assert not overlap, f"Labels appear in both AMMONIUM and AMINE sets: {overlap}"


# ===================================================================
# 8. Multi-row batch computation
# ===================================================================

def test_batch_computation_row_independence():
    """Context features for row i should not depend on row j."""
    df_single = _make_df("CC(=O)O", "carboxylic_acid", "carboxylic_acid")
    ctx_single = build_ionization_context_frame(df_single).iloc[0]

    df_batch = pd.DataFrame({
        "smiles": ["CC(=O)O", "c1ccncc1", "CN"],
        "resolved_groups": ["carboxylic_acid", "pyridine", "primary_amine"],
        "candidate_label": ["carboxylic_acid", "pyridine", "primary_amine"],
    })
    ctx_batch = build_ionization_context_frame(df_batch).iloc[0]

    for col in ION_CTX_COLUMNS:
        assert ctx_single[col] == ctx_batch[col], (
            f"Column {col} changed in batch mode: {ctx_single[col]} vs {ctx_batch[col]}"
        )
