"""
Model Degradation Tests (Retrain on Subsets)
==============================================
These tests retrain Stage 1, Stage 2, and site-state models on reduced
data fractions (100%, 50%, 25%, 10%) and measure MAE / accuracy
degradation.  They answer the question: "How capable is pKa prediction
given less data, and how trustworthy are site/protonation predictions?"

Marked ``@pytest.mark.slow`` — skip with ``pytest -m "not slow"`` for
fast CI runs.  Full run takes ~2-5 minutes depending on hardware.
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score

import sys as _sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROCESSED_DIR, "ml_models")
SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
for _p in (_PROJECT_ROOT, SCRIPTS_DIR):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Lazy imports (only needed when these tests actually run)
# ---------------------------------------------------------------------------

ASSIGNMENTS_PATH = os.path.join(PROCESSED_DIR, "functional_group_assignments.csv")
SUMMARY_PATH = os.path.join(PROCESSED_DIR, "functional_group_pka_summary.csv")


def _skip_if_missing(path):
    if not os.path.exists(path):
        pytest.skip(f"Data file not found: {path}")


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def single_group_data():
    """Load single-group rows for Stage 1 training."""
    _skip_if_missing(ASSIGNMENTS_PATH)
    from train_intrinsic_single_group_model import load_single_group_rows, build_intrinsic_features
    from ml_splits import random_molecule_split

    df = load_single_group_rows(ASSIGNMENTS_PATH)
    if df.empty:
        pytest.skip("No single-group rows available")

    X = build_intrinsic_features(df, nbits=512, radius=2)
    y = df["pka_value"].astype(float)
    train_idx, test_idx = random_molecule_split(df, test_frac=0.2, seed=17)
    return df, X, y, train_idx, test_idx


@pytest.fixture(scope="module")
def stage1_baseline_mae(single_group_data):
    """Train Stage 1 on full training set and return test MAE as baseline."""
    df, X, y, train_idx, test_idx = single_group_data
    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=8, max_iter=400, random_state=17,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])
    mae = mean_absolute_error(y.iloc[test_idx], pred)
    r2 = r2_score(y.iloc[test_idx], pred)
    print(f"\n[BASELINE] Stage 1  MAE={mae:.3f}  R²={r2:.3f}  "
          f"(train={len(train_idx)}, test={len(test_idx)})")
    return mae


def _subsample_train(train_idx, fraction, seed=42):
    """Randomly subsample the training indices to a given fraction."""
    rng = np.random.default_rng(seed)
    n = max(1, int(len(train_idx) * fraction))
    return rng.choice(train_idx, size=n, replace=False)


# ===================================================================
# Stage 1 — Intrinsic pKa degradation
# ===================================================================

@pytest.mark.slow
def test_stage1_half_data_mae(single_group_data, stage1_baseline_mae):
    """Stage 1 retrained on 50% data should degrade by < 1.5 MAE."""
    df, X, y, train_idx, test_idx = single_group_data
    sub_train = _subsample_train(train_idx, 0.50)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=8, max_iter=400, random_state=17,
    )
    model.fit(X.iloc[sub_train], y.iloc[sub_train])
    pred = model.predict(X.iloc[test_idx])
    mae = mean_absolute_error(y.iloc[test_idx], pred)
    r2 = r2_score(y.iloc[test_idx], pred)
    degradation = mae - stage1_baseline_mae

    print(f"\n[50% DATA] Stage 1  MAE={mae:.3f}  R²={r2:.3f}  "
          f"degradation={degradation:+.3f}  (train={len(sub_train)})")

    assert degradation < 1.5, (
        f"Stage 1 at 50% data: MAE degradation {degradation:.3f} exceeds 1.5 budget.\n"
        f"Baseline MAE={stage1_baseline_mae:.3f}, 50% MAE={mae:.3f}\n"
        f"The model is too sensitive to data volume — consider stronger regularization."
    )


@pytest.mark.slow
def test_stage1_quarter_data_mae(single_group_data, stage1_baseline_mae):
    """Stage 1 retrained on 25% data — degradation should be < 3.0 MAE."""
    df, X, y, train_idx, test_idx = single_group_data
    sub_train = _subsample_train(train_idx, 0.25)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=8, max_iter=400, random_state=17,
    )
    model.fit(X.iloc[sub_train], y.iloc[sub_train])
    pred = model.predict(X.iloc[test_idx])
    mae = mean_absolute_error(y.iloc[test_idx], pred)
    degradation = mae - stage1_baseline_mae

    print(f"\n[25% DATA] Stage 1  MAE={mae:.3f}  degradation={degradation:+.3f}  "
          f"(train={len(sub_train)})")

    assert degradation < 3.0, (
        f"Stage 1 at 25% data: MAE degradation {degradation:.3f} exceeds 3.0 budget.\n"
        f"Baseline MAE={stage1_baseline_mae:.3f}, 25% MAE={mae:.3f}\n"
        f"WARNING: Quarter-data performance is dangerously poor."
    )


@pytest.mark.slow
def test_stage1_tenth_data_mae(single_group_data, stage1_baseline_mae):
    """Stage 1 retrained on 10% data — expect significant degradation.
    If MAE degrades > 5.0, the model is essentially guessing."""
    df, X, y, train_idx, test_idx = single_group_data
    sub_train = _subsample_train(train_idx, 0.10)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=8, max_iter=400, random_state=17,
    )
    model.fit(X.iloc[sub_train], y.iloc[sub_train])
    pred = model.predict(X.iloc[test_idx])
    mae = mean_absolute_error(y.iloc[test_idx], pred)
    degradation = mae - stage1_baseline_mae

    print(f"\n[10% DATA] Stage 1  MAE={mae:.3f}  degradation={degradation:+.3f}  "
          f"(train={len(sub_train)})")
    print(f"  *** DATA CLIFF REPORT ***")
    print(f"  100% train MAE = {stage1_baseline_mae:.3f}")
    print(f"  10%  train MAE = {mae:.3f}")
    print(f"  Degradation    = {degradation:+.3f} pKa units")

    assert degradation < 5.0, (
        f"Stage 1 at 10% data: MAE degradation {degradation:.3f} exceeds 5.0.\n"
        f"The model is essentially GUESSING with this little data.\n"
        f"Baseline MAE={stage1_baseline_mae:.3f}, 10% MAE={mae:.3f}"
    )


# ===================================================================
# Stage 2 — Delta context model degradation
# ===================================================================

@pytest.fixture(scope="module")
def multi_group_data():
    """Load multi-group rows for Stage 2 training."""
    _skip_if_missing(ASSIGNMENTS_PATH)
    stage1_pkl = os.path.join(MODELS_DIR, "stage1_intrinsic/stage1_intrinsic_model.pkl")
    _skip_if_missing(stage1_pkl)

    from train_delta_context_model import load_multi_group_rows
    from train_intrinsic_single_group_model import build_intrinsic_features
    from ml_splits import random_molecule_split

    df = load_multi_group_rows(ASSIGNMENTS_PATH)
    if df.empty or len(df) < 20:
        pytest.skip("Not enough multi-group rows for Stage 2 test")

    # Load Stage 1 model for computing intrinsic predictions
    with open(stage1_pkl, "rb") as fh:
        stage1_bundle = pickle.load(fh)

    stage1_model = stage1_bundle["model"]
    stage1_cols = stage1_bundle["feature_columns"]
    fp_bits = int(stage1_bundle.get("fp_bits", 512))
    fp_radius = int(stage1_bundle.get("fp_radius", 2))

    # Build Stage-1-like features for multi-group rows
    X_s1 = build_intrinsic_features(df, nbits=fp_bits, radius=fp_radius)
    X_s1 = X_s1.reindex(columns=stage1_cols, fill_value=0.0)
    df["intrinsic_pred_pka"] = stage1_model.predict(X_s1)
    df["delta_pka"] = df["pka_value"].astype(float) - df["intrinsic_pred_pka"]

    # Build Stage 2 features
    from train_delta_context_model import build_delta_features
    X = build_delta_features(df, nbits=fp_bits, radius=fp_radius)
    y = df["delta_pka"].astype(float)

    train_idx, test_idx = random_molecule_split(df, test_frac=0.2, seed=17)
    return df, X, y, train_idx, test_idx


@pytest.fixture(scope="module")
def stage2_baseline_mae(multi_group_data):
    """Train Stage 2 on full training set and return test MAE."""
    df, X, y, train_idx, test_idx = multi_group_data
    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=8, max_iter=500, random_state=17,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])
    mae = mean_absolute_error(y.iloc[test_idx], pred)
    # Also compute effective pKa MAE
    eff_pred = df.iloc[test_idx]["intrinsic_pred_pka"].to_numpy() + pred
    eff_mae = mean_absolute_error(df.iloc[test_idx]["pka_value"].to_numpy(), eff_pred)

    print(f"\n[BASELINE] Stage 2  ΔpKa MAE={mae:.3f}  effective pKa MAE={eff_mae:.3f}  "
          f"(train={len(train_idx)}, test={len(test_idx)})")
    return mae


@pytest.mark.slow
def test_stage2_half_data_mae(multi_group_data, stage2_baseline_mae):
    """Stage 2 at 50% data should degrade by < 2.0 MAE on delta."""
    df, X, y, train_idx, test_idx = multi_group_data
    sub_train = _subsample_train(train_idx, 0.50)

    model = HistGradientBoostingRegressor(
        learning_rate=0.05, max_depth=8, max_iter=500, random_state=17,
    )
    model.fit(X.iloc[sub_train], y.iloc[sub_train])
    pred = model.predict(X.iloc[test_idx])
    mae = mean_absolute_error(y.iloc[test_idx], pred)
    degradation = mae - stage2_baseline_mae

    print(f"\n[50% DATA] Stage 2  ΔpKa MAE={mae:.3f}  degradation={degradation:+.3f}")

    assert degradation < 2.0, (
        f"Stage 2 at 50% data: delta MAE degradation {degradation:.3f} exceeds 2.0.\n"
        f"Baseline={stage2_baseline_mae:.3f}, 50%={mae:.3f}"
    )


# ===================================================================
# Site-state classifier degradation
# ===================================================================

@pytest.fixture(scope="module")
def site_state_data(candidate_dataset):
    """Build feature matrix for site-state model from candidate dataset."""
    from ml_features import build_feature_matrix
    from ml_splits import random_molecule_split

    df = candidate_dataset
    if df.empty or len(df) < 30:
        pytest.skip("Not enough candidate rows for site-state test")

    X, y = build_feature_matrix(df, nbits=512, radius=2)
    train_idx, test_idx = random_molecule_split(df, test_frac=0.2, seed=17)
    return df, X, y, train_idx, test_idx


@pytest.fixture(scope="module")
def site_baseline_top1(site_state_data):
    """Train site-state model on full data and get top-1 accuracy."""
    df, X, y, train_idx, test_idx = site_state_data
    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=8, max_iter=300, random_state=17,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    test_df = df.iloc[test_idx].copy()
    test_df["pred_prob"] = model.predict_proba(X.iloc[test_idx])[:, 1]

    # Top-1 site accuracy: for each molecule, does the top-ranked candidate
    # have is_true_site == 1?
    top_picks = (
        test_df.sort_values("pred_prob", ascending=False)
        .groupby("molecule_key", as_index=False)
        .first()
    )
    top1_acc = float(top_picks["is_true_site"].mean())
    print(f"\n[BASELINE] Site-state  top1_acc={top1_acc:.3f}  "
          f"(train={len(train_idx)}, test={len(test_idx)})")
    return top1_acc


@pytest.mark.slow
def test_site_state_half_data(site_state_data, site_baseline_top1):
    """Site-state at 50% data should maintain top-1 accuracy above 0.50."""
    df, X, y, train_idx, test_idx = site_state_data
    sub_train = _subsample_train(train_idx, 0.50)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=8, max_iter=300, random_state=17,
    )
    model.fit(X.iloc[sub_train], y.iloc[sub_train])

    test_df = df.iloc[test_idx].copy()
    test_df["pred_prob"] = model.predict_proba(X.iloc[test_idx])[:, 1]
    top_picks = (
        test_df.sort_values("pred_prob", ascending=False)
        .groupby("molecule_key", as_index=False)
        .first()
    )
    top1_acc = float(top_picks["is_true_site"].mean())
    print(f"\n[50% DATA] Site-state  top1_acc={top1_acc:.3f}  "
          f"(baseline={site_baseline_top1:.3f})")

    assert top1_acc >= 0.50, (
        f"Site-state at 50% data: top1_acc={top1_acc:.3f} < 0.50 threshold.\n"
        f"Baseline={site_baseline_top1:.3f}.\n"
        f"Site ranking is unreliable with this little data."
    )


@pytest.mark.slow
def test_site_state_quarter_data(site_state_data, site_baseline_top1):
    """Site-state at 25% data should stay above 0.40 top-1 accuracy."""
    df, X, y, train_idx, test_idx = site_state_data
    sub_train = _subsample_train(train_idx, 0.25)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=8, max_iter=300, random_state=17,
    )
    model.fit(X.iloc[sub_train], y.iloc[sub_train])

    test_df = df.iloc[test_idx].copy()
    test_df["pred_prob"] = model.predict_proba(X.iloc[test_idx])[:, 1]
    top_picks = (
        test_df.sort_values("pred_prob", ascending=False)
        .groupby("molecule_key", as_index=False)
        .first()
    )
    top1_acc = float(top_picks["is_true_site"].mean())
    drop = site_baseline_top1 - top1_acc

    print(f"\n[25% DATA] Site-state  top1_acc={top1_acc:.3f}  "
          f"drop={drop:+.3f} from baseline")

    assert top1_acc >= 0.40, (
        f"Site-state at 25% data: top1_acc={top1_acc:.3f} < 0.40.\n"
        f"Site ranking is dangerously unreliable."
    )


# ===================================================================
# Summary report (session-end)
# ===================================================================

@pytest.mark.slow
def test_degradation_summary(
    single_group_data,
    stage1_baseline_mae,
):
    """Print a summary table of degradation results."""
    df, X, y, train_idx, test_idx = single_group_data
    fractions = [1.0, 0.50, 0.25, 0.10]
    results = []
    for frac in fractions:
        if frac == 1.0:
            sub_tr = train_idx
        else:
            sub_tr = _subsample_train(train_idx, frac)
        model = HistGradientBoostingRegressor(
            learning_rate=0.05, max_depth=8, max_iter=400, random_state=17,
        )
        model.fit(X.iloc[sub_tr], y.iloc[sub_tr])
        pred = model.predict(X.iloc[test_idx])
        mae = mean_absolute_error(y.iloc[test_idx], pred)
        r2 = r2_score(y.iloc[test_idx], pred)
        results.append({
            "fraction": frac,
            "train_n": len(sub_tr),
            "mae": mae,
            "r2": r2,
            "degradation": mae - stage1_baseline_mae,
        })

    print("\n" + "=" * 70)
    print("  STAGE 1 pKa PREDICTION — DATA FRACTION DEGRADATION REPORT")
    print("=" * 70)
    print(f"  {'Fraction':>10}  {'Train N':>8}  {'MAE':>7}  {'R²':>7}  {'Degradation':>12}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['fraction']:>10.0%}  {r['train_n']:>8d}  {r['mae']:>7.3f}  "
              f"{r['r2']:>7.3f}  {r['degradation']:>+12.3f}")
    print("=" * 70)
    # This test always passes — it's informational
    assert True
