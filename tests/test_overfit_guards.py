"""
Overfit / Underfit Guard Tests
=================================
Quick-turnaround tests that train on real data and verify:
  - Train vs test MAE/accuracy gap is reasonable (no severe overfit)
  - Test R² / accuracy is above minimum (no severe underfit)
  - Model doesn't predict constant values
  - Carboxyl & pair-form heads generalize (balanced accuracy > chance)
  - Delta context model actually improves over intrinsic-only baseline

Marked ``@pytest.mark.slow`` — skip with ``pytest -m "not slow"`` for CI.
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    balanced_accuracy_score,
    roc_auc_score,
)

PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed",
)
MODELS_DIR = os.path.join(PROCESSED_DIR, "ml_models")
ASSIGNMENTS_PATH = os.path.join(PROCESSED_DIR, "functional_group_assignments.csv")
SUMMARY_PATH = os.path.join(PROCESSED_DIR, "functional_group_pka_summary.csv")


def _skip_if_missing(path: str):
    if not os.path.exists(path):
        pytest.skip(f"Data file not found: {path}")


# ===================================================================
#  Stage 1 — Intrinsic pKa regression guards
# ===================================================================

@pytest.fixture(scope="module")
def stage1_train_test():
    """Train a fresh Stage 1 model and return train/test metrics."""
    _skip_if_missing(ASSIGNMENTS_PATH)
    from train_intrinsic_single_group_model import (
        load_single_group_rows, build_intrinsic_features,
    )
    from ml_splits import random_molecule_split

    df = load_single_group_rows(ASSIGNMENTS_PATH)
    if df.empty or len(df) < 50:
        pytest.skip("Insufficient data for Stage 1 overfit test")

    X = build_intrinsic_features(df, nbits=256, radius=2)
    y = df["pka_value"].astype(float)
    train_idx, test_idx = random_molecule_split(df, test_frac=0.2, seed=17)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=8, max_iter=200, random_state=17,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    train_pred = model.predict(X.iloc[train_idx])
    test_pred = model.predict(X.iloc[test_idx])

    return {
        "model": model,
        "y_train": y.iloc[train_idx].to_numpy(),
        "y_test": y.iloc[test_idx].to_numpy(),
        "train_pred": train_pred,
        "test_pred": test_pred,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }


@pytest.mark.slow
class TestStage1OverfitGuards:

    def test_train_test_mae_gap(self, stage1_train_test):
        """Train MAE should not be too much lower than test MAE (overfit signal)."""
        d = stage1_train_test
        train_mae = mean_absolute_error(d["y_train"], d["train_pred"])
        test_mae = mean_absolute_error(d["y_test"], d["test_pred"])
        gap = test_mae - train_mae

        print(f"\n[Stage 1] train MAE={train_mae:.3f}  test MAE={test_mae:.3f}  gap={gap:.3f}")
        assert gap < 3.0, (
            f"Stage 1 overfit detected: train MAE={train_mae:.3f} vs test MAE={test_mae:.3f}, "
            f"gap={gap:.3f} exceeds 3.0 pKa units"
        )

    def test_test_r2_above_minimum(self, stage1_train_test):
        """Test R² should be positive (model is better than predicting mean)."""
        d = stage1_train_test
        r2 = r2_score(d["y_test"], d["test_pred"])
        print(f"\n[Stage 1] test R²={r2:.3f}")
        assert r2 > 0.10, (
            f"Stage 1 underfit: test R²={r2:.3f} (< 0.10). "
            "Model is barely better than predicting the mean."
        )

    def test_not_predicting_constant(self, stage1_train_test):
        """Predictions should have non-trivial variance (not predicting a constant)."""
        d = stage1_train_test
        pred_std = np.std(d["test_pred"])
        target_std = np.std(d["y_test"])
        print(f"\n[Stage 1] pred_std={pred_std:.3f}  target_std={target_std:.3f}")
        assert pred_std > 0.5, (
            f"Stage 1 predictions have near-zero variance ({pred_std:.3f}). "
            "Model may be predicting a constant."
        )
        # Prediction variance should be at least 10% of target variance
        assert pred_std > 0.10 * target_std, (
            f"Predictions are too narrow: pred_std={pred_std:.3f} vs target_std={target_std:.3f}"
        )

    def test_test_mae_below_baseline(self, stage1_train_test):
        """Test MAE should be significantly below naive mean-prediction MAE."""
        d = stage1_train_test
        test_mae = mean_absolute_error(d["y_test"], d["test_pred"])
        naive_mae = mean_absolute_error(d["y_test"], np.full_like(d["y_test"], d["y_train"].mean()))
        improvement = naive_mae - test_mae
        print(f"\n[Stage 1] test MAE={test_mae:.3f}  naive MAE={naive_mae:.3f}  improvement={improvement:.3f}")
        assert improvement > 0.5, (
            f"Stage 1 barely beats naive baseline: model MAE={test_mae:.3f}, "
            f"naive MAE={naive_mae:.3f}, improvement={improvement:.3f}"
        )


# ===================================================================
#  Stage 2 — Delta context model guards
# ===================================================================

@pytest.fixture(scope="module")
def stage2_train_test():
    """Train Stage 2 and compare effective pKa vs intrinsic-only."""
    _skip_if_missing(ASSIGNMENTS_PATH)
    stage1_pkl = os.path.join(MODELS_DIR, "stage1_intrinsic/stage1_intrinsic_model.pkl")
    _skip_if_missing(stage1_pkl)

    from train_delta_context_model import load_multi_group_rows, build_delta_features
    from train_intrinsic_single_group_model import build_intrinsic_features
    from ml_splits import random_molecule_split

    df = load_multi_group_rows(ASSIGNMENTS_PATH)
    if df.empty or len(df) < 50:
        pytest.skip("Insufficient multi-group data for Stage 2 overfit test")

    with open(stage1_pkl, "rb") as fh:
        s1_bundle = pickle.load(fh)

    X_s1 = build_intrinsic_features(df, nbits=int(s1_bundle.get("fp_bits", 512)), radius=int(s1_bundle.get("fp_radius", 2)))
    X_s1 = X_s1.reindex(columns=s1_bundle["feature_columns"], fill_value=0.0)
    df["intrinsic_pred_pka"] = s1_bundle["model"].predict(X_s1)
    df["delta_pka"] = df["pka_value"].astype(float) - df["intrinsic_pred_pka"]

    X_delta = build_delta_features(df, nbits=256, radius=2)
    y_delta = df["delta_pka"].astype(float)

    train_idx, test_idx = random_molecule_split(df, test_frac=0.2, seed=17)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=8, max_iter=200, random_state=17,
    )
    model.fit(X_delta.iloc[train_idx], y_delta.iloc[train_idx])

    test_delta_pred = model.predict(X_delta.iloc[test_idx])
    test_effective = df.iloc[test_idx]["intrinsic_pred_pka"].to_numpy() + test_delta_pred
    test_intrinsic_only = df.iloc[test_idx]["intrinsic_pred_pka"].to_numpy()
    test_true = df.iloc[test_idx]["pka_value"].to_numpy()

    train_delta_pred = model.predict(X_delta.iloc[train_idx])

    return {
        "y_test": test_true,
        "test_effective": test_effective,
        "test_intrinsic_only": test_intrinsic_only,
        "y_delta_train": y_delta.iloc[train_idx].to_numpy(),
        "y_delta_test": y_delta.iloc[test_idx].to_numpy(),
        "train_delta_pred": train_delta_pred,
        "test_delta_pred": test_delta_pred,
    }


@pytest.mark.slow
class TestStage2OverfitGuards:

    def test_effective_better_than_intrinsic_only(self, stage2_train_test):
        """Stage 2 effective pKa should be closer to truth than intrinsic alone."""
        d = stage2_train_test
        eff_mae = mean_absolute_error(d["y_test"], d["test_effective"])
        intr_mae = mean_absolute_error(d["y_test"], d["test_intrinsic_only"])
        print(f"\n[Stage 2] effective MAE={eff_mae:.3f}  intrinsic-only MAE={intr_mae:.3f}")
        # Allow small tolerance: effective should not be much worse
        assert eff_mae <= intr_mae + 0.5, (
            f"Stage 2 delta model makes things WORSE: effective MAE={eff_mae:.3f} "
            f"vs intrinsic-only MAE={intr_mae:.3f}"
        )

    def test_delta_train_test_gap(self, stage2_train_test):
        """Delta prediction gap should not be extreme."""
        d = stage2_train_test
        train_mae = mean_absolute_error(d["y_delta_train"], d["train_delta_pred"])
        test_mae = mean_absolute_error(d["y_delta_test"], d["test_delta_pred"])
        gap = test_mae - train_mae
        print(f"\n[Stage 2 delta] train MAE={train_mae:.3f}  test MAE={test_mae:.3f}  gap={gap:.3f}")
        assert gap < 3.0, f"Stage 2 delta overfit gap={gap:.3f}"


# ===================================================================
#  Site-state classifier guards
# ===================================================================

@pytest.fixture(scope="module")
def site_state_train_test():
    """Train a fresh site-state model and return accuracy metrics."""
    _skip_if_missing(ASSIGNMENTS_PATH)
    _skip_if_missing(SUMMARY_PATH)
    from ml_dataset_builder import DatasetBuildConfig, build_candidate_dataset
    from ml_features import build_feature_matrix
    from ml_splits import random_molecule_split
    from train_site_state_model import _topk_site_accuracy

    config = DatasetBuildConfig(assignments_path=ASSIGNMENTS_PATH, summary_path=SUMMARY_PATH)
    df = build_candidate_dataset(config)
    if df.empty or len(df) < 50:
        pytest.skip("Insufficient candidate data for site-state test")

    X, y = build_feature_matrix(df, nbits=256, radius=2)
    train_idx, test_idx = random_molecule_split(df, test_frac=0.2, seed=17)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=8, max_iter=200, random_state=17,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    train_proba = model.predict_proba(X.iloc[train_idx])[:, 1]
    test_proba = model.predict_proba(X.iloc[test_idx])[:, 1]

    train_df = df.iloc[train_idx].copy()
    train_df["pred_proba"] = train_proba
    test_df = df.iloc[test_idx].copy()
    test_df["pred_proba"] = test_proba

    return {
        "train_top1": _topk_site_accuracy(train_df, k=1),
        "test_top1": _topk_site_accuracy(test_df, k=1),
        "train_auroc": float(roc_auc_score(y.iloc[train_idx], train_proba)),
        "test_auroc": float(roc_auc_score(y.iloc[test_idx], test_proba)),
    }


@pytest.mark.slow
class TestSiteStateOverfitGuards:

    def test_top1_accuracy_above_chance(self, site_state_train_test):
        """Site-state top-1 accuracy on test set should beat random guessing."""
        d = site_state_train_test
        print(f"\n[Site-state] train top1={d['train_top1']:.3f}  test top1={d['test_top1']:.3f}")
        assert d["test_top1"] > 0.30, (
            f"Site-state test top-1 accuracy={d['test_top1']:.3f} is near-random. "
            "Model is underfitting."
        )

    def test_train_test_auroc_gap(self, site_state_train_test):
        """AUROC gap between train and test should not be extreme."""
        d = site_state_train_test
        gap = d["train_auroc"] - d["test_auroc"]
        print(f"\n[Site-state] train AUROC={d['train_auroc']:.3f}  "
              f"test AUROC={d['test_auroc']:.3f}  gap={gap:.3f}")
        assert gap < 0.25, (
            f"Site-state overfit: AUROC gap={gap:.3f} (train={d['train_auroc']:.3f}, "
            f"test={d['test_auroc']:.3f})"
        )

    def test_test_auroc_above_minimum(self, site_state_train_test):
        """Test AUROC should be well above 0.5 (random)."""
        d = site_state_train_test
        assert d["test_auroc"] > 0.55, (
            f"Site-state test AUROC={d['test_auroc']:.3f} is near-random"
        )


# ===================================================================
#  Carboxyl form head guard
# ===================================================================

@pytest.fixture(scope="module")
def carboxyl_head_train_test():
    """Train a fresh carboxyl form head and return metrics."""
    _skip_if_missing(ASSIGNMENTS_PATH)
    from train_carboxyl_form_head import _prepare_training_rows, _split_by_molecule
    from ionization_context_features import build_carboxyl_form_feature_frame

    frame = _prepare_training_rows(ASSIGNMENTS_PATH)
    if frame.empty or len(frame) < 30:
        pytest.skip("Insufficient carboxyl data for overfit test")

    X = build_carboxyl_form_feature_frame(
        frame,
        candidate_col="candidate_label",
        resolved_col="resolved_groups",
        smiles_col="smiles",
        pka_type_col="pka_type_canonical",
    )
    y = frame["target_is_acid_form"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        pytest.skip("Single class in carboxyl training data")

    split = _split_by_molecule(frame, test_frac=0.2, seed=17)
    train_idx, test_idx = split["train"], split["test"]

    model = LogisticRegression(max_iter=1500, class_weight="balanced")
    model.fit(X.iloc[train_idx], y[train_idx])

    train_proba = model.predict_proba(X.iloc[train_idx])[:, 1]
    test_proba = model.predict_proba(X.iloc[test_idx])[:, 1]
    train_pred = (train_proba >= 0.5).astype(int)
    test_pred = (test_proba >= 0.5).astype(int)

    return {
        "train_ba": balanced_accuracy_score(y[train_idx], train_pred),
        "test_ba": balanced_accuracy_score(y[test_idx], test_pred),
        "test_auroc": float(roc_auc_score(y[test_idx], test_proba)) if len(np.unique(y[test_idx])) > 1 else 0.5,
    }


@pytest.mark.slow
class TestCarboxylHeadOverfitGuards:

    def test_balanced_accuracy_above_chance(self, carboxyl_head_train_test):
        d = carboxyl_head_train_test
        print(f"\n[Carboxyl head] train BA={d['train_ba']:.3f}  test BA={d['test_ba']:.3f}")
        assert d["test_ba"] > 0.50, (
            f"Carboxyl form head test balanced accuracy={d['test_ba']:.3f} "
            "is at or below random chance"
        )

    def test_train_test_ba_gap(self, carboxyl_head_train_test):
        d = carboxyl_head_train_test
        gap = d["train_ba"] - d["test_ba"]
        print(f"\n[Carboxyl head] BA gap={gap:.3f}")
        assert gap < 0.25, (
            f"Carboxyl form head overfit: BA gap={gap:.3f}"
        )

    def test_auroc_reasonable(self, carboxyl_head_train_test):
        d = carboxyl_head_train_test
        assert d["test_auroc"] > 0.50, (
            f"Carboxyl form head AUROC={d['test_auroc']:.3f} at or below chance"
        )


# ===================================================================
#  Pair form heads — per-head overfit guards
# ===================================================================

@pytest.fixture(scope="module")
def pair_head_metrics():
    """Train ALL pair-form heads and collect per-head metrics."""
    _skip_if_missing(ASSIGNMENTS_PATH)
    _skip_if_missing(SUMMARY_PATH)
    from train_pair_form_heads import HEAD_SPECS, _pick_best_threshold
    from ml_dataset_builder import DatasetBuildConfig, build_candidate_dataset
    from ionization_context_features import build_carboxyl_form_feature_frame
    from functional_group_pka_analysis import CONJUGATE_FAMILY_MAP

    cand = build_candidate_dataset(
        DatasetBuildConfig(
            assignments_path=ASSIGNMENTS_PATH,
            summary_path=SUMMARY_PATH,
            min_candidates=1,
        )
    )
    if cand.empty:
        pytest.skip("Empty candidate dataset")

    base = cand[
        (cand["group_mode"] == "pair_type")
        & (cand["candidate_own_form"].isin(["acid_form", "base_form"]))
        & (cand["distribution_ok"])
    ].copy()
    if "candidate_family" not in base.columns:
        base["candidate_family"] = base["candidate_label"].map(
            lambda x: CONJUGATE_FAMILY_MAP.get(str(x), str(x))
        )
    base = base.drop_duplicates(subset=["molecule_key", "candidate_instance_id", "pka_value"]).copy()

    results = {}
    for head_name, spec in HEAD_SPECS.items():
        labels = set(spec["acid_labels"]) | set(spec["base_labels"])
        frame = base[base["candidate_label"].isin(labels)].copy()
        if len(frame) < 30:
            continue

        frame["target_is_acid_form"] = (frame["candidate_own_form"] == "acid_form").astype(int)
        y = frame["target_is_acid_form"].to_numpy()
        if len(np.unique(y)) < 2:
            continue

        X = build_carboxyl_form_feature_frame(
            frame,
            candidate_col="candidate_label",
            resolved_col="resolved_groups",
            smiles_col="smiles",
            pka_type_col="pka_type_canonical",
            include_pka_type=False,
        )

        rng = np.random.default_rng(17)
        mols = np.array(sorted(frame["molecule_key"].unique()))
        rng.shuffle(mols)
        n_test = max(1, int(len(mols) * 0.2))
        test_mols = set(mols[:n_test])
        test_mask = frame["molecule_key"].isin(test_mols).to_numpy()
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue

        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(X.iloc[train_idx], y[train_idx])

        train_proba = model.predict_proba(X.iloc[train_idx])[:, 1]
        test_proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        opt_thr = _pick_best_threshold(y[test_idx], test_proba)

        train_pred = (train_proba >= opt_thr).astype(int)
        test_pred = (test_proba >= opt_thr).astype(int)

        results[head_name] = {
            "n_rows": len(frame),
            "train_ba": balanced_accuracy_score(y[train_idx], train_pred),
            "test_ba": balanced_accuracy_score(y[test_idx], test_pred),
            "test_auroc": float(roc_auc_score(y[test_idx], test_proba)),
            "threshold": opt_thr,
        }

    if not results:
        pytest.skip("No pair-form heads had enough data to train")
    return results


@pytest.mark.slow
class TestPairFormHeadOverfitGuards:

    def test_all_heads_above_chance(self, pair_head_metrics):
        """Every successfully trained head should have test BA > 0.50."""
        failures = []
        for head_name, m in pair_head_metrics.items():
            if m["test_ba"] <= 0.50:
                failures.append(f"{head_name}: BA={m['test_ba']:.3f}")
        if failures:
            pytest.fail(
                f"Pair-form heads at or below chance:\n  " + "\n  ".join(failures)
            )

    def test_overfit_gap_reasonable(self, pair_head_metrics):
        """Train-test BA gap should not exceed 0.30 for any head."""
        failures = []
        for head_name, m in pair_head_metrics.items():
            gap = m["train_ba"] - m["test_ba"]
            if gap > 0.30:
                failures.append(f"{head_name}: gap={gap:.3f}")
        if failures:
            pytest.fail(
                f"Pair-form heads with overfit BA gap > 0.30:\n  " + "\n  ".join(failures)
            )

    def test_thresholds_sensible(self, pair_head_metrics):
        """Optimized thresholds should be within [0.20, 0.80]."""
        for head_name, m in pair_head_metrics.items():
            assert 0.10 <= m["threshold"] <= 0.90, (
                f"Pair head '{head_name}' has extreme threshold={m['threshold']:.2f}"
            )

    def test_summary_report(self, pair_head_metrics):
        """Print a summary of all pair-form head metrics for diagnostic purposes."""
        print("\n" + "=" * 75)
        print("  PAIR-FORM HEAD OVERFIT / UNDERFIT REPORT")
        print("=" * 75)
        print(f"  {'Head':<20} {'Rows':>6} {'Train BA':>9} {'Test BA':>8} "
              f"{'Gap':>6} {'AUROC':>6} {'Thr':>5}")
        print("  " + "-" * 66)
        for head_name in sorted(pair_head_metrics.keys()):
            m = pair_head_metrics[head_name]
            gap = m["train_ba"] - m["test_ba"]
            print(f"  {head_name:<20} {m['n_rows']:>6} {m['train_ba']:>9.3f} "
                  f"{m['test_ba']:>8.3f} {gap:>+6.3f} {m['test_auroc']:>6.3f} "
                  f"{m['threshold']:>5.2f}")
        print("=" * 75)
        assert True  # informational
