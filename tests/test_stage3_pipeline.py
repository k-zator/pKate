"""
Stage 3 Pipeline — Deep Coverage Tests
========================================
Tests that exercise the UNCOVERED Stage 3 code paths:
  - _build_stage1_like_features / _build_stage2_features (full feature
    construction with descriptors, fingerprints, dummies, multihot, ion_ctx)
  - _resolved_group_multihot encoding
  - _infer_site_probability / _infer_carboxyl_form_probability with
    mock and real model bundles
  - _apply_carboxyl_form_head (end-to-end with a real LogisticRegression)
  - _apply_pair_form_heads — all three branches: carboxyl conservative,
    pyridine_like conservative, generic HH-gated
  - _molecule_level_summary
  - _load_bundle
"""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from stage3_protonation_state_inference import (
    _build_stage1_like_features,
    _build_stage2_features,
    _resolved_group_multihot,
    _infer_site_probability,
    _infer_carboxyl_form_probability,
    _apply_carboxyl_form_head,
    _apply_pair_form_heads,
    _molecule_level_summary,
    _load_bundle,
    _protonated_fraction,
)


# ===================================================================
# Helpers
# ===================================================================

def _mini_candidate_df(n: int = 5) -> pd.DataFrame:
    """Build a small candidate-style DataFrame with real SMILES."""
    smiles_list = ["CC(=O)O", "c1ccncc1", "CCCC(=O)O", "CCN", "c1ccc(O)cc1"][:n]
    labels = ["carboxylic_acid", "pyridine", "carboxylic_acid", "primary_amine", "phenol"][:n]
    return pd.DataFrame({
        "smiles": smiles_list,
        "candidate_label": labels,
        "pka_type_canonical": ["acidic", "basic", "acidic", "basic", "acidic"][:n],
        "resolved_groups": [
            "carboxylic_acid",
            "pyridine",
            "carboxylic_acid|primary_amine",
            "primary_amine",
            "phenol",
        ][:n],
        "resolved_group_count": [1, 1, 2, 1, 1][:n],
        "intrinsic_pred_pka": [4.8, 5.2, 4.5, 10.0, 10.3][:n],
        "molecule_formal_charge": [0, 0, 0, 0, 0][:n],
        "neutral_input_risk": [False, False, False, False, False][:n],
        "group_mode": ["pair_type", "pair_type", "pair_type", "pair_type", "pair_type"][:n],
        "candidate_own_form": ["acid_form", "base_form", "acid_form", "base_form", "acid_form"][:n],
        "pair_member_form": ["acid_form", "base_form", "acid_form", "base_form", "acid_form"][:n],
    })


# ===================================================================
# 1. _resolved_group_multihot
# ===================================================================

class TestResolvedGroupMultihot:

    def test_single_group(self):
        s = pd.Series(["carboxylic_acid"])
        result = _resolved_group_multihot(s)
        assert result.shape == (1, 1)
        assert "ctx_has_carboxylic_acid" in result.columns
        assert result.iloc[0, 0] == 1.0

    def test_two_groups_pipe_separated(self):
        s = pd.Series(["carboxylic_acid|primary_amine"])
        result = _resolved_group_multihot(s)
        assert "ctx_has_carboxylic_acid" in result.columns
        assert "ctx_has_primary_amine" in result.columns
        assert result["ctx_has_carboxylic_acid"].iloc[0] == 1.0
        assert result["ctx_has_primary_amine"].iloc[0] == 1.0

    def test_absent_group_is_zero(self):
        s = pd.Series(["carboxylic_acid", "primary_amine"])
        result = _resolved_group_multihot(s)
        # Row 0 should have carboxylic_acid=1, primary_amine=0
        assert result["ctx_has_carboxylic_acid"].iloc[0] == 1.0
        assert result["ctx_has_primary_amine"].iloc[0] == 0.0
        # Row 1 should have carboxylic_acid=0, primary_amine=1
        assert result["ctx_has_carboxylic_acid"].iloc[1] == 0.0
        assert result["ctx_has_primary_amine"].iloc[1] == 1.0

    def test_empty_series(self):
        s = pd.Series(["", ""], dtype=str)
        result = _resolved_group_multihot(s)
        assert result.shape[0] == 2
        # No columns or all-zero (empty tokens filtered out)

    def test_consistent_column_order(self):
        """Columns should be sorted across calls."""
        s = pd.Series(["b|a", "a|c"])
        result = _resolved_group_multihot(s)
        assert list(result.columns) == sorted(result.columns)


# ===================================================================
# 2. _build_stage1_like_features
# ===================================================================

class TestBuildStage1LikeFeatures:

    def test_output_shape_matches_feature_columns(self):
        df = _mini_candidate_df()
        # Simulate a feature_columns list that build_intrinsic_features would produce
        n_fp = 64
        feature_cols = [f"fp_{i}" for i in range(n_fp)] + ["group_carboxylic_acid", "pka_type_acidic"]
        result = _build_stage1_like_features(df, feature_columns=feature_cols, fp_bits=n_fp, fp_radius=2)
        assert result.shape == (len(df), len(feature_cols))
        assert list(result.columns) == feature_cols

    def test_no_nans_in_fingerprints(self):
        df = _mini_candidate_df()
        n_fp = 64
        feature_cols = [f"fp_{i}" for i in range(n_fp)]
        result = _build_stage1_like_features(df, feature_columns=feature_cols, fp_bits=n_fp, fp_radius=2)
        fp_cols = [c for c in result.columns if c.startswith("fp_")]
        assert not result[fp_cols].isna().any().any(), "Fingerprint columns should never contain NaN"

    def test_missing_columns_filled_with_zero(self):
        """Feature columns not present in the data should be filled with 0."""
        df = _mini_candidate_df(n=1)
        feature_cols = ["fp_0", "group_NONEXISTENT_LABEL", "pka_type_magical"]
        result = _build_stage1_like_features(df, feature_columns=feature_cols, fp_bits=64, fp_radius=2)
        assert result["group_NONEXISTENT_LABEL"].iloc[0] == 0.0
        assert result["pka_type_magical"].iloc[0] == 0.0

    def test_group_dummies_contain_real_labels(self):
        """At least some group dummy columns should be active."""
        df = _mini_candidate_df()
        # Use a large feature_columns list that includes the expected dummy
        n_fp = 64
        feature_cols = [f"fp_{i}" for i in range(n_fp)] + ["group_carboxylic_acid"]
        result = _build_stage1_like_features(df, feature_columns=feature_cols, fp_bits=n_fp, fp_radius=2)
        # Row 0 is carboxylic_acid → group_carboxylic_acid should be 1
        assert result["group_carboxylic_acid"].iloc[0] == 1.0


# ===================================================================
# 3. _build_stage2_features
# ===================================================================

class TestBuildStage2Features:

    def test_includes_intrinsic_pred_pka(self):
        df = _mini_candidate_df()
        n_fp = 64
        feature_cols = ["intrinsic_pred_pka", "resolved_group_count"] + [f"fp_{i}" for i in range(n_fp)]
        result = _build_stage2_features(df, feature_columns=feature_cols, fp_bits=n_fp, fp_radius=2)
        assert result.shape[0] == len(df)
        assert result["intrinsic_pred_pka"].iloc[0] == pytest.approx(4.8, abs=0.01)

    def test_resolved_group_count_numeric(self):
        df = _mini_candidate_df()
        n_fp = 64
        feature_cols = ["resolved_group_count"] + [f"fp_{i}" for i in range(n_fp)]
        result = _build_stage2_features(df, feature_columns=feature_cols, fp_bits=n_fp, fp_radius=2)
        assert np.issubdtype(result["resolved_group_count"].dtype, np.number)

    def test_molecule_formal_charge_included(self):
        df = _mini_candidate_df()
        n_fp = 64
        feature_cols = ["molecule_formal_charge"] + [f"fp_{i}" for i in range(n_fp)]
        result = _build_stage2_features(df, feature_columns=feature_cols, fp_bits=n_fp, fp_radius=2)
        assert "molecule_formal_charge" in result.columns

    def test_no_nans_in_output(self):
        df = _mini_candidate_df()
        n_fp = 64
        feature_cols = ["intrinsic_pred_pka", "resolved_group_count", "molecule_formal_charge",
                        "neutral_input_risk"] + [f"fp_{i}" for i in range(n_fp)]
        result = _build_stage2_features(df, feature_columns=feature_cols, fp_bits=n_fp, fp_radius=2)
        assert not result.isna().any().any(), f"NaN found in columns: {result.columns[result.isna().any()].tolist()}"


# ===================================================================
# 4. _infer_site_probability
# ===================================================================

class TestInferSiteProbability:

    def test_none_bundle_returns_ones(self):
        df = _mini_candidate_df()
        result = _infer_site_probability(df, None)
        assert len(result) == len(df)
        np.testing.assert_array_equal(result.values, np.ones(len(df)))

    def test_bundle_without_model_returns_ones(self):
        result = _infer_site_probability(_mini_candidate_df(), {"model": None, "feature_columns": None})
        np.testing.assert_array_equal(result.values, np.ones(len(result)))

    @pytest.mark.slow
    def test_with_real_bundle(self, site_model_bundle, candidate_dataset):
        """When a real site-state model is available, probabilities should be in [0,1]."""
        sample = candidate_dataset.head(50).copy()
        if sample.empty:
            pytest.skip("Empty candidate dataset")
        proba = _infer_site_probability(sample, site_model_bundle)
        assert len(proba) == len(sample)
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0


# ===================================================================
# 5. _infer_carboxyl_form_probability
# ===================================================================

class TestInferCarboxylFormProbability:

    def test_none_bundle_returns_nan(self):
        df = _mini_candidate_df()
        result = _infer_carboxyl_form_probability(df, None)
        assert len(result) == len(df)
        assert result.isna().all()

    def test_bundle_without_model_returns_nan(self):
        result = _infer_carboxyl_form_probability(
            _mini_candidate_df(), {"model": None, "feature_columns": None}
        )
        assert result.isna().all()

    @pytest.mark.slow
    def test_with_real_bundle(self, carboxyl_form_bundle, candidate_dataset):
        """Real carboxyl form bundle produces valid probabilities."""
        sample = candidate_dataset.head(50).copy()
        if sample.empty:
            pytest.skip("Empty candidate dataset")
        proba = _infer_carboxyl_form_probability(sample, carboxyl_form_bundle)
        assert len(proba) == len(sample)
        valid = proba.dropna()
        if len(valid) > 0:
            assert valid.min() >= 0.0
            assert valid.max() <= 1.0


# ===================================================================
# 6. _apply_carboxyl_form_head (end-to-end with mock LR)
# ===================================================================

class TestApplyCarboxylFormHeadEndToEnd:

    def _make_df_with_carboxyl(self, prob_acid: float):
        """Build a DataFrame that triggers the carboxyl form head path."""
        return pd.DataFrame({
            "candidate_family": ["carboxyl"],
            "candidate_label": ["carboxylic_acid"],
            "candidate_member_form": ["acid_form"],
            "pred_member_form": ["base_form"],  # baseline says base
            "smiles": ["CC(=O)O"],
            "resolved_groups": ["carboxylic_acid"],
            "resolved_group_count": [1],
            "pka_type_canonical": ["acidic"],
            # Override _infer_carboxyl_form_probability by pre-injecting
        })

    def test_high_acid_prob_flips_to_acid(self):
        """prob_acid=0.85 > 0.5+margin → override to acid_form."""
        from unittest.mock import patch
        df = self._make_df_with_carboxyl(0.85)

        with patch(
            "stage3_protonation_state_inference._infer_carboxyl_form_probability",
            return_value=pd.Series([0.85]),
        ):
            bundle = {"model": "dummy", "feature_columns": ["x"]}
            result = _apply_carboxyl_form_head(df, bundle, confidence_margin=0.10)
            assert result["pred_member_form"].iloc[0] == "acid_form"
            assert result["pred_member_form_rule_source"].iloc[0] == "carboxyl_form_head"

    def test_low_acid_prob_flips_to_base(self):
        """prob_acid=0.15 < 0.5-margin → override to base_form."""
        from unittest.mock import patch
        df = self._make_df_with_carboxyl(0.15)

        with patch(
            "stage3_protonation_state_inference._infer_carboxyl_form_probability",
            return_value=pd.Series([0.15]),
        ):
            bundle = {"model": "dummy", "feature_columns": ["x"]}
            result = _apply_carboxyl_form_head(df, bundle, confidence_margin=0.10)
            assert result["pred_member_form"].iloc[0] == "base_form"
            assert result["pred_member_form_rule_source"].iloc[0] == "carboxyl_form_head"

    def test_prob_near_half_no_override(self):
        """prob_acid=0.52 within margin → no override."""
        from unittest.mock import patch
        df = self._make_df_with_carboxyl(0.52)

        with patch(
            "stage3_protonation_state_inference._infer_carboxyl_form_probability",
            return_value=pd.Series([0.52]),
        ):
            bundle = {"model": "dummy", "feature_columns": ["x"]}
            result = _apply_carboxyl_form_head(df, bundle, confidence_margin=0.10)
            # pred_member_form should remain as baseline (base_form) because
            # |0.52 - 0.5| = 0.02 < 0.10 margin
            assert result["pred_member_form"].iloc[0] == "base_form"

    def test_non_carboxyl_not_affected(self):
        """Non-carboxyl rows should pass through unchanged."""
        from unittest.mock import patch
        df = pd.DataFrame({
            "candidate_family": ["amine"],
            "candidate_label": ["primary_amine"],
            "candidate_member_form": ["base_form"],
            "pred_member_form": ["base_form"],
            "smiles": ["CCN"],
            "resolved_groups": ["primary_amine"],
            "resolved_group_count": [1],
            "pka_type_canonical": ["basic"],
        })
        with patch(
            "stage3_protonation_state_inference._infer_carboxyl_form_probability",
            return_value=pd.Series([0.85]),
        ):
            bundle = {"model": "dummy", "feature_columns": ["x"]}
            result = _apply_carboxyl_form_head(df, bundle, confidence_margin=0.10)
            assert result["pred_member_form"].iloc[0] == "base_form"


# ===================================================================
# 7. _apply_pair_form_heads — all three branch types
# ===================================================================

class TestApplyPairFormHeadsBranches:
    """Exercise the carboxyl conservative, pyridine conservative,
    and generic HH-gated branches of _apply_pair_form_heads."""

    @staticmethod
    def _make_bundle(head_name, acid_labels, base_labels, threshold=0.5, n_features=3):
        """Build a minimal pair-form-heads bundle with a real LR model."""
        model = LogisticRegression()
        # Fit on trivial data to make predict_proba work
        X_fake = np.random.default_rng(42).standard_normal((20, n_features))
        y_fake = np.array([0]*10 + [1]*10)
        model.fit(X_fake, y_fake)

        feature_columns = [f"f{i}" for i in range(n_features)]
        head = {
            "model": model,
            "feature_columns": feature_columns,
            "acid_labels": acid_labels,
            "base_labels": base_labels,
            "candidate_col": "candidate_label",
            "resolved_col": "resolved_groups",
            "smiles_col": "smiles",
            "pka_type_col": "pka_type_canonical",
            "include_pka_type": False,
            "decision_threshold": threshold,
            "decision_margin": 0.10,
        }
        return {"heads": {head_name: head}}

    @staticmethod
    def _make_df(candidate_label, pred_pka, ph=7.4):
        return pd.DataFrame({
            "candidate_label": [candidate_label],
            "pred_member_form": ["base_form"],
            "pred_effective_pka": [pred_pka],
            "ph_target": [ph],
            "smiles": ["CC(=O)O"],
            "resolved_groups": [candidate_label],
            "resolved_group_count": [1],
            "pka_type_canonical": ["acidic"],
        })

    def test_carboxyl_conservative_high_acid_prob(self):
        """Carboxyl branch: proba >= strong_acid_threshold AND pKa > pH+0.75."""
        from unittest.mock import patch
        bundle = self._make_bundle("carboxyl", ["carboxylic_acid"], ["carboxylate"])
        df = self._make_df("carboxylic_acid", pred_pka=10.0, ph=7.4)

        with patch(
            "stage3_protonation_state_inference.build_carboxyl_form_feature_frame",
            return_value=pd.DataFrame({"f0": [1.0], "f1": [0.0], "f2": [0.5]}),
        ):
            # Need proba >= strong_acid_threshold (max(0.80, threshold+0.20) = 0.80 for thr=0.5)
            # AND acid_context_gate (pred_pka >= pH+0.75 = 8.15; 10.0 >= 8.15 ✓)
            with patch.object(
                bundle["heads"]["carboxyl"]["model"], "predict_proba",
                return_value=np.array([[0.15, 0.85]]),  # proba=0.85 >= 0.80
            ):
                result = _apply_pair_form_heads(df, bundle)
                assert result["pred_member_form"].iloc[0] == "acid_form"
                assert "carboxyl_conservative" in result["pred_member_form_rule_source"].iloc[0]

    def test_carboxyl_conservative_low_prob_not_overridden(self):
        """Carboxyl branch: proba < strong_acid_threshold → no override."""
        from unittest.mock import patch
        bundle = self._make_bundle("carboxyl", ["carboxylic_acid"], ["carboxylate"])
        df = self._make_df("carboxylic_acid", pred_pka=10.0, ph=7.4)

        with patch(
            "stage3_protonation_state_inference.build_carboxyl_form_feature_frame",
            return_value=pd.DataFrame({"f0": [1.0], "f1": [0.0], "f2": [0.5]}),
        ):
            with patch.object(
                bundle["heads"]["carboxyl"]["model"], "predict_proba",
                return_value=np.array([[0.40, 0.60]]),  # 0.60 < 0.80 threshold
            ):
                result = _apply_pair_form_heads(df, bundle)
                # Should NOT be overridden — stays as baseline
                assert result["pred_member_form"].iloc[0] == "base_form"

    def test_pyridine_conservative_strong_base(self):
        """Pyridine branch: low proba → base_form assignment."""
        from unittest.mock import patch
        bundle = self._make_bundle("pyridine_like", ["pyridinium"], ["pyridine"], threshold=0.5)
        df = self._make_df("pyridine", pred_pka=3.0, ph=7.4)

        with patch(
            "stage3_protonation_state_inference.build_carboxyl_form_feature_frame",
            return_value=pd.DataFrame({"f0": [0.0], "f1": [1.0], "f2": [0.5]}),
        ):
            with patch.object(
                bundle["heads"]["pyridine_like"]["model"], "predict_proba",
                return_value=np.array([[0.80, 0.20]]),  # proba=0.20 <= strong_base=0.35
            ):
                result = _apply_pair_form_heads(df, bundle)
                assert result["pred_member_form"].iloc[0] == "base_form"
                assert "pyridine_like_conservative" in result["pred_member_form_rule_source"].iloc[0]

    def test_generic_hh_gated_acid_blocked_by_low_pka(self):
        """Generic head: ML says acid, but pKa << pH → HH gate blocks."""
        from unittest.mock import patch
        bundle = self._make_bundle("primary_amine", ["primary_ammonium"], ["primary_amine"], threshold=0.5)
        df = self._make_df("primary_ammonium", pred_pka=3.0, ph=7.4)

        with patch(
            "stage3_protonation_state_inference.build_carboxyl_form_feature_frame",
            return_value=pd.DataFrame({"f0": [1.0], "f1": [0.0], "f2": [0.5]}),
        ):
            with patch.object(
                bundle["heads"]["primary_amine"]["model"], "predict_proba",
                return_value=np.array([[0.10, 0.90]]),  # ML says acid with high confidence
            ):
                result = _apply_pair_form_heads(df, bundle)
                # acid_context_gate: 3.0 >= 8.15 → False
                # base_context_gate: 3.0 <= 6.65 → True
                # inconclusive: False
                # ML acid AND (acid_gate OR inconclusive) → False → BLOCKED
                # ML acid AND inconclusive → False → still blocked
                # The acid override should be blocked
                # But base prediction IS allowed: ml_base is false (proba=0.9 >= 0.5),
                # so no base override either → stays baseline
                assert result["pred_member_form"].iloc[0] == "base_form"

    def test_generic_hh_gated_acid_allowed_with_high_pka(self):
        """Generic head: ML says acid AND pKa >> pH → acid allowed."""
        from unittest.mock import patch
        bundle = self._make_bundle("primary_amine", ["primary_ammonium"], ["primary_amine"], threshold=0.5)
        df = self._make_df("primary_ammonium", pred_pka=12.0, ph=7.4)

        with patch(
            "stage3_protonation_state_inference.build_carboxyl_form_feature_frame",
            return_value=pd.DataFrame({"f0": [1.0], "f1": [0.0], "f2": [0.5]}),
        ):
            with patch.object(
                bundle["heads"]["primary_amine"]["model"], "predict_proba",
                return_value=np.array([[0.10, 0.90]]),
            ):
                result = _apply_pair_form_heads(df, bundle)
                # acid_context_gate: 12.0 >= 8.15 → True
                # ML acid (proba=0.9 >= 0.5) AND confident (|0.9-0.5|=0.4 >= 0.10)
                # acid_mask = True → override to acid_form
                assert result["pred_member_form"].iloc[0] == "acid_form"
                assert "hh_gated" in result["pred_member_form_rule_source"].iloc[0]

    def test_generic_hh_gated_inconclusive_allows_ml(self):
        """Generic head: pKa ≈ pH → inconclusive gate → ML allowed."""
        from unittest.mock import patch
        bundle = self._make_bundle("primary_amine", ["primary_ammonium"], ["primary_amine"], threshold=0.5)
        df = self._make_df("primary_ammonium", pred_pka=7.5, ph=7.4)

        with patch(
            "stage3_protonation_state_inference.build_carboxyl_form_feature_frame",
            return_value=pd.DataFrame({"f0": [1.0], "f1": [0.0], "f2": [0.5]}),
        ):
            with patch.object(
                bundle["heads"]["primary_amine"]["model"], "predict_proba",
                return_value=np.array([[0.10, 0.90]]),
            ):
                result = _apply_pair_form_heads(df, bundle)
                # acid_context_gate: 7.5 >= 8.15 → False
                # base_context_gate: 7.5 <= 6.65 → False
                # inconclusive: True → ML acid allowed
                assert result["pred_member_form"].iloc[0] == "acid_form"


# ===================================================================
# 8. _molecule_level_summary
# ===================================================================

class TestMoleculeLevelSummary:

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=[
            "molecule_key", "combined_score", "candidate_label", "true_label",
            "pred_member_form", "true_pair_member_form", "selection_confidence",
        ])
        result = _molecule_level_summary(df)
        assert result["molecules"] == 0
        assert result["site_top1_acc"] == 0.0

    def test_perfect_site_prediction(self):
        df = pd.DataFrame({
            "molecule_key": ["m1", "m1", "m2", "m2"],
            "combined_score": [0.9, 0.1, 0.8, 0.2],
            "candidate_label": ["carboxylic_acid", "phenol", "pyridine", "amine"],
            "true_label": ["carboxylic_acid", "carboxylic_acid", "pyridine", "pyridine"],
            "pred_member_form": ["acid_form", "acid_form", "base_form", "base_form"],
            "true_pair_member_form": ["acid_form", "acid_form", "base_form", "base_form"],
            "selection_confidence": [0.8, 0.4, 0.7, 0.3],
        })
        result = _molecule_level_summary(df)
        assert result["molecules"] == 2
        assert result["site_top1_acc"] == 1.0

    def test_wrong_site_prediction(self):
        df = pd.DataFrame({
            "molecule_key": ["m1", "m1"],
            "combined_score": [0.9, 0.1],
            "candidate_label": ["phenol", "carboxylic_acid"],
            "true_label": ["carboxylic_acid", "carboxylic_acid"],
            "pred_member_form": ["acid_form", "acid_form"],
            "true_pair_member_form": ["acid_form", "acid_form"],
            "selection_confidence": [0.6, 0.3],
        })
        result = _molecule_level_summary(df)
        assert result["site_top1_acc"] == 0.0  # top pick is phenol, truth is carboxylic_acid

    def test_pair_form_accuracy(self):
        """Pair form accuracy should only count rows with acid/base form."""
        df = pd.DataFrame({
            "molecule_key": ["m1", "m2"],
            "combined_score": [0.9, 0.8],
            "candidate_label": ["carboxylic_acid", "primary_amine"],
            "true_label": ["carboxylic_acid", "primary_amine"],
            "pred_member_form": ["acid_form", "base_form"],
            "true_pair_member_form": ["acid_form", "acid_form"],
            "selection_confidence": [0.8, 0.7],
        })
        result = _molecule_level_summary(df)
        # m1 correct, m2 incorrect → 50%
        assert result["pair_form_acc"] == pytest.approx(0.5)


# ===================================================================
# 9. _load_bundle
# ===================================================================

class TestLoadBundle:

    def test_load_dict_with_model(self, tmp_path):
        bundle = {"model": "my_model", "feature_columns": ["a", "b"]}
        path = str(tmp_path / "bundle.pkl")
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh)
        result = _load_bundle(path)
        assert result["model"] == "my_model"

    def test_load_dict_with_heads(self, tmp_path):
        bundle = {"heads": {"carboxyl": {"model": "lr"}}}
        path = str(tmp_path / "heads.pkl")
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh)
        result = _load_bundle(path)
        assert "heads" in result

    def test_load_raw_model_wrapped(self, tmp_path):
        """A bare model (not a dict) should be wrapped in {"model": ...}."""
        model = LogisticRegression()
        path = str(tmp_path / "raw.pkl")
        with open(path, "wb") as fh:
            pickle.dump(model, fh)
        result = _load_bundle(path)
        assert "model" in result
        assert isinstance(result["model"], LogisticRegression)


# ===================================================================
# 10. End-to-end: build features → predict with real models
# ===================================================================

@pytest.mark.slow
class TestEndToEndPipeline:

    def test_stage1_features_with_real_bundle(self, stage1_bundle, candidate_dataset):
        """Build Stage 1 features for a sample and predict — values should be finite."""
        sample = candidate_dataset.head(30).copy()
        if sample.empty:
            pytest.skip("Empty candidate dataset")
        X = _build_stage1_like_features(
            sample,
            feature_columns=stage1_bundle["feature_columns"],
            fp_bits=int(stage1_bundle.get("fp_bits", 512)),
            fp_radius=int(stage1_bundle.get("fp_radius", 2)),
        )
        assert X.shape == (len(sample), len(stage1_bundle["feature_columns"]))
        preds = stage1_bundle["model"].predict(X)
        assert np.all(np.isfinite(preds))

    def test_stage2_features_with_real_bundle(self, stage1_bundle, stage2_bundle, candidate_dataset):
        """Build Stage 2 features for multi-group candidates."""
        sample = candidate_dataset.copy()
        sample["resolved_group_count"] = pd.to_numeric(sample["resolved_group_count"], errors="coerce").fillna(1)
        multi = sample[sample["resolved_group_count"] > 1].head(30).copy()
        if multi.empty:
            pytest.skip("No multi-group rows")
        multi = multi.reset_index(drop=True)

        # First get intrinsic predictions using stage3's _build_stage1_like_features
        # (which uses candidate_label, not final_group_label)
        X1 = _build_stage1_like_features(
            multi,
            feature_columns=stage1_bundle["feature_columns"],
            fp_bits=int(stage1_bundle.get("fp_bits", 512)),
            fp_radius=int(stage1_bundle.get("fp_radius", 2)),
        )
        multi = multi.iloc[:len(X1)].copy()
        multi["intrinsic_pred_pka"] = stage1_bundle["model"].predict(X1)

        X2 = _build_stage2_features(
            multi,
            feature_columns=stage2_bundle["feature_columns"],
            fp_bits=int(stage2_bundle.get("fp_bits", 512)),
            fp_radius=int(stage2_bundle.get("fp_radius", 2)),
        )
        assert X2.shape == (len(multi), len(stage2_bundle["feature_columns"]))
        deltas = stage2_bundle["model"].predict(X2)
        assert np.all(np.isfinite(deltas))

    def test_carboxyl_form_head_real_pipeline(self, carboxyl_form_bundle, candidate_dataset):
        """Apply carboxyl form head to real carboxyl candidates."""
        from stage3_protonation_state_inference import _label_form
        sample = candidate_dataset.head(100).copy()
        if sample.empty:
            pytest.skip("Empty candidate dataset")

        family_form = sample["candidate_label"].map(_label_form)
        sample["candidate_family"] = family_form.map(lambda x: x[0])
        sample["candidate_member_form"] = family_form.map(lambda x: x[1] if x[1] else "non_pair")
        sample["pred_member_form"] = "base_form"

        carboxyl_rows = sample[sample["candidate_family"] == "carboxyl"]
        if carboxyl_rows.empty:
            pytest.skip("No carboxyl candidates in sample")

        result = _apply_carboxyl_form_head(sample, carboxyl_form_bundle)
        assert "carboxyl_form_prob_acid" in result.columns

    def test_pair_form_heads_real_pipeline(self, pair_form_heads_bundle, candidate_dataset):
        """Apply pair form heads to real candidates — all branches."""
        from stage3_protonation_state_inference import _label_form
        sample = candidate_dataset.head(200).copy()
        if sample.empty:
            pytest.skip("Empty candidate dataset")

        family_form = sample["candidate_label"].map(_label_form)
        sample["candidate_family"] = family_form.map(lambda x: x[0])
        sample["candidate_member_form"] = family_form.map(lambda x: x[1] if x[1] else "non_pair")
        sample["pred_member_form"] = "base_form"
        sample["pred_effective_pka"] = 5.0
        sample["ph_target"] = 7.4

        result = _apply_pair_form_heads(sample, pair_form_heads_bundle)
        # The column is added only when matching candidates exist;
        # verify it's present OR that no heads matched (both valid).
        if "pred_member_form_rule_source" in result.columns:
            # At least some rows were processed
            overridden = result["pred_member_form_rule_source"].notna().sum()
            print(f"\n  pair_form_heads processed {overridden}/{len(result)} rows")
        # Regardless, output rows must equal input rows
        assert len(result) == len(sample)
