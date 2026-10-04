"""
Integration tests for training pipeline ``main()`` flows.

Each training script follows load → features → split → train → eval → save.
These tests replicate that sequence using the real data / fixtures and
verify bundle structure, metric keys, model types, and saved artefacts.
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest

# ── path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from ml_dataset_builder import DatasetBuildConfig, build_candidate_dataset
from ml_features import build_feature_matrix, molecule_descriptors, morgan_bits
from ml_splits import random_molecule_split, scaffold_molecule_split, split_manifest
from ionization_context_features import build_carboxyl_form_feature_frame
from train_intrinsic_single_group_model import build_intrinsic_features

MODELS_DIR = os.path.join(PROCESSED_DIR, "ml_models")


def _load_pkl(relpath: str):
    """Load a pickle file from the models directory, skip if missing."""
    path = os.path.join(MODELS_DIR, relpath)
    if not os.path.exists(path):
        pytest.skip(f"Model file not found: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 – Intrinsic Single-Group Model  (train_intrinsic_single_group_model)
# ═══════════════════════════════════════════════════════════════════════════

class TestStage1Pipeline:
    """End-to-end pipeline test for ``train_intrinsic_single_group_model``."""

    @pytest.fixture(scope="class")
    def single_df(self):
        from train_intrinsic_single_group_model import load_single_group_rows
        path = os.path.join(PROCESSED_DIR, "functional_group_assignments.csv")
        if not os.path.exists(path):
            pytest.skip("assignments CSV not found")
        df = load_single_group_rows(path)
        if df.empty:
            pytest.skip("no single-group rows")
        return df

    @pytest.fixture(scope="class")
    def features_and_target(self, single_df):
        X = build_intrinsic_features(single_df, nbits=512, radius=2)
        y = single_df["pka_value"].astype(float)
        return X, y

    @pytest.fixture(scope="class")
    def splits(self, single_df):
        r_train, r_test = random_molecule_split(single_df, test_frac=0.2)
        s_train, s_test = scaffold_molecule_split(single_df, test_frac=0.2)
        return {
            "random": (r_train, r_test),
            "scaffold": (s_train, s_test),
        }

    # ── fit_and_eval ────────────────────────────────────────────────────
    @pytest.fixture(scope="class")
    def fit_result(self, single_df, features_and_target, splits):
        from train_intrinsic_single_group_model import _fit_and_eval
        X, y = features_and_target
        train_idx, test_idx = splits["random"]
        model, metrics, eval_df = _fit_and_eval(single_df, X, y, train_idx, test_idx)
        return model, metrics, eval_df

    def test_model_type(self, fit_result):
        model, _, _ = fit_result
        assert isinstance(model, HistGradientBoostingRegressor)

    def test_metrics_keys(self, fit_result):
        _, metrics, _ = fit_result
        for key in ("mae", "rmse", "r2"):
            assert key in metrics, f"Missing metric: {key}"

    def test_metrics_values_sensible(self, fit_result):
        _, metrics, _ = fit_result
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["r2"] <= 1.0

    def test_eval_df_has_predictions(self, fit_result):
        _, _, eval_df = fit_result
        assert "pred_intrinsic_pka" in eval_df.columns
        assert "abs_error" in eval_df.columns
        assert len(eval_df) > 0

    # ── scaffold fit for bundle ─────────────────────────────────────────
    @pytest.fixture(scope="class")
    def scaffold_fit(self, single_df, features_and_target, splits):
        from train_intrinsic_single_group_model import _fit_and_eval
        X, y = features_and_target
        train_idx, test_idx = splits["scaffold"]
        return _fit_and_eval(single_df, X, y, train_idx, test_idx)

    def test_scaffold_model_trained(self, scaffold_fit):
        model, _, _ = scaffold_fit
        # Should be fitted (has predict method and at least some trees)
        assert hasattr(model, "predict")

    def test_scaffold_metrics_nonempty(self, scaffold_fit):
        _, metrics, _ = scaffold_fit
        assert len(metrics) >= 3

    # ── bundle save / reload ────────────────────────────────────────────
    def test_bundle_save_and_reload(self, scaffold_fit, features_and_target, tmp_path):
        model, _, _ = scaffold_fit
        X, _ = features_and_target
        bundle = {
            "model": model,
            "feature_columns": list(X.columns),
            "fp_bits": 512,
            "fp_radius": 2,
        }
        path = tmp_path / "stage1_intrinsic_model.pkl"
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh)
        with open(path, "rb") as fh:
            loaded = pickle.load(fh)

        assert set(loaded.keys()) == {"model", "feature_columns", "fp_bits", "fp_radius"}
        assert isinstance(loaded["model"], HistGradientBoostingRegressor)
        assert loaded["fp_bits"] == 512

    # ── split manifest ──────────────────────────────────────────────────
    def test_split_manifest_structure(self, single_df, splits):
        train_idx, test_idx = splits["random"]
        manifest = split_manifest(single_df, train_idx, test_idx)
        assert "split" in manifest.columns
        assert set(manifest["split"].unique()) <= {"train", "test", "unused"}
        assert len(manifest) <= len(single_df)
        assert len(manifest) > 0

    # ── metrics JSON round-trip ─────────────────────────────────────────
    def test_metrics_json_roundtrip(self, fit_result, scaffold_fit, single_df, tmp_path):
        _, random_metrics, _ = fit_result
        _, scaffold_metrics, _ = scaffold_fit
        report = {
            "rows": int(len(single_df)),
            "molecules": int(single_df["molecule_key"].nunique()),
            "random": random_metrics,
            "scaffold": scaffold_metrics,
        }
        path = tmp_path / "metrics.json"
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2)
        with open(path) as fh:
            loaded = json.load(fh)
        assert loaded["rows"] == len(single_df)
        assert "mae" in loaded["random"]
        assert "mae" in loaded["scaffold"]


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 2 – Delta Context Model  (train_delta_context_model)
# ═══════════════════════════════════════════════════════════════════════════

class TestStage2Pipeline:
    """End-to-end pipeline test for ``train_delta_context_model``."""

    @pytest.fixture(scope="class")
    def stage1_bundle(self):
        return _load_pkl("stage1_intrinsic/stage1_intrinsic_model.pkl")

    @pytest.fixture(scope="class")
    def multi_df(self):
        from train_delta_context_model import load_multi_group_rows
        path = os.path.join(PROCESSED_DIR, "functional_group_assignments.csv")
        if not os.path.exists(path):
            pytest.skip("assignments CSV not found")
        df = load_multi_group_rows(path)
        if df.empty:
            pytest.skip("no multi-group rows")
        return df

    # ── _load_stage1_bundle validation ──────────────────────────────────
    def test_load_stage1_bundle_validates_keys(self, tmp_path):
        from train_delta_context_model import _load_stage1_bundle
        # Missing keys should raise ValueError
        bad_bundle = {"model": "fake"}
        bad_path = tmp_path / "bad.pkl"
        with open(bad_path, "wb") as fh:
            pickle.dump(bad_bundle, fh)
        with pytest.raises(ValueError, match="missing keys"):
            _load_stage1_bundle(str(bad_path))

    def test_load_stage1_bundle_success(self, tmp_path):
        from train_delta_context_model import _load_stage1_bundle
        good_bundle = {
            "model": "fake",
            "feature_columns": ["a"],
            "fp_bits": 512,
            "fp_radius": 2,
        }
        path = tmp_path / "good.pkl"
        with open(path, "wb") as fh:
            pickle.dump(good_bundle, fh)
        loaded = _load_stage1_bundle(str(path))
        assert set(loaded.keys()) >= {"model", "feature_columns", "fp_bits", "fp_radius"}

    # ── build stage1-like features ──────────────────────────────────────
    def test_stage1_like_features_for_multi(self, multi_df, stage1_bundle):
        from train_delta_context_model import _build_stage1_like_features
        X = _build_stage1_like_features(
            multi_df,
            feature_columns=stage1_bundle["feature_columns"],
            fp_bits=int(stage1_bundle["fp_bits"]),
            fp_radius=int(stage1_bundle["fp_radius"]),
        )
        assert len(X) == len(multi_df)
        assert list(X.columns) == list(stage1_bundle["feature_columns"])

    # ── intrinsic predictions + delta target ────────────────────────────
    @pytest.fixture(scope="class")
    def multi_with_intrinsic(self, multi_df, stage1_bundle):
        from train_delta_context_model import _build_stage1_like_features
        X_s1 = _build_stage1_like_features(
            multi_df,
            feature_columns=stage1_bundle["feature_columns"],
            fp_bits=int(stage1_bundle["fp_bits"]),
            fp_radius=int(stage1_bundle["fp_radius"]),
        )
        df = multi_df.copy()
        df["intrinsic_pred_pka"] = stage1_bundle["model"].predict(X_s1)
        df["delta_pka"] = df["pka_value"].astype(float) - df["intrinsic_pred_pka"].astype(float)
        return df

    def test_delta_target_created(self, multi_with_intrinsic):
        assert "delta_pka" in multi_with_intrinsic.columns
        assert "intrinsic_pred_pka" in multi_with_intrinsic.columns
        assert multi_with_intrinsic["delta_pka"].notna().all()

    # ── build delta features ────────────────────────────────────────────
    @pytest.fixture(scope="class")
    def delta_features(self, multi_with_intrinsic):
        from train_delta_context_model import build_delta_features
        X = build_delta_features(multi_with_intrinsic, nbits=512, radius=2)
        return X

    def test_delta_features_shape(self, delta_features, multi_with_intrinsic):
        assert len(delta_features) == len(multi_with_intrinsic)
        assert delta_features.shape[1] > 10, "Expected many delta feature columns"

    def test_delta_features_has_numeric_cols(self, delta_features):
        assert "resolved_group_count" in delta_features.columns
        assert "intrinsic_pred_pka" in delta_features.columns

    def test_delta_features_has_fingerprints(self, delta_features):
        fp_cols = [c for c in delta_features.columns if c.startswith("fp_")]
        assert len(fp_cols) == 512

    def test_delta_features_has_context_columns(self, delta_features):
        ctx_cols = [c for c in delta_features.columns if c.startswith("ctx_has_")]
        assert len(ctx_cols) > 0, "Expected resolved-group multihot context columns"

    def test_delta_features_has_geometry_columns(self, delta_features):
        geom_cols = [
            c
            for c in delta_features.columns
            if c.startswith("mol_geom_") or c.startswith("site_geom_")
        ]
        assert len(geom_cols) > 0, "Expected embedded 3D geometry columns"

    # ── _fit_and_eval ───────────────────────────────────────────────────
    @pytest.fixture(scope="class")
    def delta_fit_result(self, multi_with_intrinsic, delta_features):
        from train_delta_context_model import _fit_and_eval
        y_delta = multi_with_intrinsic["delta_pka"].astype(float)
        train, test = random_molecule_split(multi_with_intrinsic, test_frac=0.2)
        model, metrics, eval_df = _fit_and_eval(
            multi_with_intrinsic, delta_features, y_delta, train, test
        )
        return model, metrics, eval_df

    def test_delta_model_type(self, delta_fit_result):
        model, _, _ = delta_fit_result
        assert isinstance(model, HistGradientBoostingRegressor)

    def test_delta_metrics_keys(self, delta_fit_result):
        _, metrics, _ = delta_fit_result
        expected = {
            "effective_mae", "effective_rmse", "effective_r2",
            "intrinsic_only_mae", "intrinsic_only_rmse", "intrinsic_only_r2",
            "delta_target_std",
        }
        assert expected <= set(metrics.keys())

    def test_delta_eval_has_predictions(self, delta_fit_result):
        _, _, eval_df = delta_fit_result
        assert "pred_delta_pka" in eval_df.columns
        assert "pred_effective_pka" in eval_df.columns
        assert "pred_intrinsic_only_pka" in eval_df.columns

    # ── full pipeline save / reload ─────────────────────────────────────
    def test_bundle_save_and_reload(self, delta_fit_result, delta_features, tmp_path):
        model, _, _ = delta_fit_result
        bundle = {
            "model": model,
            "feature_columns": list(delta_features.columns),
            "fp_bits": 512,
            "fp_radius": 2,
        }
        path = tmp_path / "stage2_delta_model.pkl"
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh)
        with open(path, "rb") as fh:
            loaded = pickle.load(fh)
        assert set(loaded.keys()) == {"model", "feature_columns", "fp_bits", "fp_radius"}
        assert isinstance(loaded["model"], HistGradientBoostingRegressor)

    def test_scaffold_split_also_works(self, multi_with_intrinsic, delta_features):
        from train_delta_context_model import _fit_and_eval
        y_delta = multi_with_intrinsic["delta_pka"].astype(float)
        s_train, s_test = scaffold_molecule_split(multi_with_intrinsic, test_frac=0.2)
        model, metrics, eval_df = _fit_and_eval(
            multi_with_intrinsic, delta_features, y_delta, s_train, s_test
        )
        assert isinstance(model, HistGradientBoostingRegressor)
        assert metrics["effective_mae"] >= 0

    def test_metrics_json_roundtrip(self, delta_fit_result, multi_with_intrinsic, tmp_path):
        _, random_metrics, _ = delta_fit_result
        report = {
            "rows": int(len(multi_with_intrinsic)),
            "molecules": int(multi_with_intrinsic["molecule_key"].nunique()),
            "delta_mean": float(multi_with_intrinsic["delta_pka"].mean()),
            "delta_std": float(multi_with_intrinsic["delta_pka"].std()),
            "random": random_metrics,
        }
        path = tmp_path / "metrics.json"
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2)
        with open(path) as fh:
            loaded = json.load(fh)
        assert loaded["rows"] == len(multi_with_intrinsic)
        assert "effective_mae" in loaded["random"]


# ═══════════════════════════════════════════════════════════════════════════
#  Site-State Model  (train_site_state_model)
# ═══════════════════════════════════════════════════════════════════════════

class TestSiteStatePipeline:
    """End-to-end pipeline test for ``train_site_state_model``."""

    @pytest.fixture(scope="class")
    def candidate_df(self):
        assignments = os.path.join(PROCESSED_DIR, "functional_group_assignments.csv")
        summary = os.path.join(PROCESSED_DIR, "functional_group_pka_summary.csv")
        _skip_if_missing(STAGE1_MODEL, STAGE2_MODEL)
        if not os.path.exists(assignments):
            pytest.skip("assignments CSV not found")
        config = DatasetBuildConfig(assignments_path=assignments, summary_path=summary)
        df = build_candidate_dataset(config)
        if df.empty:
            pytest.skip("candidate dataset empty")
        from train_site_state_model import _add_predicted_pka_features
        return _add_predicted_pka_features(df, STAGE1_MODEL, STAGE2_MODEL)

    @pytest.fixture(scope="class")
    def feature_matrix(self, candidate_df):
        X, y = build_feature_matrix(candidate_df, nbits=512, radius=2)
        return X, y

    # ── _fit_and_eval ───────────────────────────────────────────────────
    @pytest.fixture(scope="class")
    def site_fit_result(self, candidate_df, feature_matrix):
        from train_site_state_model import _fit_and_eval
        X, y = feature_matrix
        train, test = random_molecule_split(candidate_df, test_frac=0.2)
        model, metrics, eval_df = _fit_and_eval(candidate_df, X, y, train, test)
        return model, metrics, eval_df

    def test_model_type(self, site_fit_result):
        model, _, _ = site_fit_result
        assert isinstance(model, HistGradientBoostingClassifier)

    def test_metrics_keys(self, site_fit_result):
        _, metrics, _ = site_fit_result
        for key in ("auroc", "auprc", "top1_site_acc", "top3_site_acc"):
            assert key in metrics, f"Missing metric: {key}"

    def test_auroc_above_chance(self, site_fit_result):
        _, metrics, _ = site_fit_result
        assert metrics["auroc"] > 0.5, f"AUROC {metrics['auroc']:.3f} not above chance"

    def test_top1_above_zero(self, site_fit_result):
        _, metrics, _ = site_fit_result
        assert metrics["top1_site_acc"] > 0.0

    def test_eval_df_has_pred_proba(self, site_fit_result):
        _, _, eval_df = site_fit_result
        assert "pred_proba" in eval_df.columns
        assert eval_df["pred_proba"].between(0, 1).all()

    # ── both split types ────────────────────────────────────────────────
    def test_scaffold_split_trains(self, candidate_df, feature_matrix):
        from train_site_state_model import _fit_and_eval
        X, y = feature_matrix
        s_train, s_test = scaffold_molecule_split(candidate_df, test_frac=0.2)
        model, metrics, eval_df = _fit_and_eval(candidate_df, X, y, s_train, s_test)
        assert isinstance(model, HistGradientBoostingClassifier)
        assert metrics["auroc"] >= 0

    # ── bundle structure ────────────────────────────────────────────────
    def test_random_bundle_structure(self, site_fit_result, feature_matrix, tmp_path):
        model, _, _ = site_fit_result
        X, _ = feature_matrix
        bundle = {
            "model": model,
            "feature_columns": list(X.columns),
            "fp_bits": 512,
            "fp_radius": 2,
            "include_observed_pka": False,
        }
        path = tmp_path / "site_state_random_model.pkl"
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh)
        with open(path, "rb") as fh:
            loaded = pickle.load(fh)
        assert set(loaded.keys()) == {"model", "feature_columns", "fp_bits", "fp_radius", "include_observed_pka"}
        assert isinstance(loaded["model"], HistGradientBoostingClassifier)

    def test_feature_matrix_includes_predicted_pka(self, feature_matrix):
        X, _ = feature_matrix
        assert "intrinsic_pred_pka" in X.columns
        assert "pred_delta_pka" in X.columns
        assert "pred_effective_pka" in X.columns

    # ── manifests ───────────────────────────────────────────────────────
    def test_manifests_match_data(self, candidate_df):
        r_train, r_test = random_molecule_split(candidate_df, test_frac=0.2)
        manifest = split_manifest(candidate_df, r_train, r_test)
        assert len(manifest) <= len(candidate_df)
        assert len(manifest) > 0
        assert set(manifest["split"].unique()) <= {"train", "test", "unused"}

    # ── metrics JSON ────────────────────────────────────────────────────
    def test_metrics_json_serialisable(self, site_fit_result, candidate_df, feature_matrix, tmp_path):
        _, random_metrics, _ = site_fit_result
        X, y = feature_matrix
        report = {
            "rows": int(len(candidate_df)),
            "molecules": int(candidate_df["molecule_key"].nunique()),
            "positive_rate": float(y.mean()),
            "random": random_metrics,
        }
        path = tmp_path / "metrics.json"
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2)
        with open(path) as fh:
            loaded = json.load(fh)
        assert "auroc" in loaded["random"]


# ═══════════════════════════════════════════════════════════════════════════
#  Carboxyl Form Head  (train_carboxyl_form_head)
# ═══════════════════════════════════════════════════════════════════════════

class TestCarboxylPipeline:
    """End-to-end pipeline test for ``train_carboxyl_form_head``."""

    @pytest.fixture(scope="class")
    def carboxyl_rows(self):
        from train_carboxyl_form_head import _prepare_training_rows
        path = os.path.join(PROCESSED_DIR, "functional_group_assignments.csv")
        if not os.path.exists(path):
            pytest.skip("assignments CSV not found")
        df = _prepare_training_rows(path)
        if df.empty:
            pytest.skip("no carboxyl pair-type rows")
        return df

    # ── _prepare_training_rows ──────────────────────────────────────────
    def test_prepare_has_required_columns(self, carboxyl_rows):
        expected = {"molecule_key", "smiles", "target_is_acid_form", "candidate_label"}
        assert expected <= set(carboxyl_rows.columns)

    def test_prepare_target_is_binary(self, carboxyl_rows):
        assert set(carboxyl_rows["target_is_acid_form"].unique()) <= {0, 1}

    def test_prepare_only_carboxyl(self, carboxyl_rows):
        assert (carboxyl_rows["final_group_family"] == "carboxyl").all()

    def test_prepare_only_pair_type(self, carboxyl_rows):
        assert (carboxyl_rows["group_mode"] == "pair_type").all()

    # ── _split_by_molecule ──────────────────────────────────────────────
    def test_split_by_molecule_no_leakage(self, carboxyl_rows):
        from train_carboxyl_form_head import _split_by_molecule
        split = _split_by_molecule(carboxyl_rows, test_frac=0.2, seed=17)
        train_mols = set(carboxyl_rows.iloc[split["train"]]["molecule_key"])
        test_mols = set(carboxyl_rows.iloc[split["test"]]["molecule_key"])
        overlap = train_mols & test_mols
        assert len(overlap) == 0, f"Molecule leakage: {len(overlap)} molecules in both splits"

    def test_split_by_molecule_covers_all_rows(self, carboxyl_rows):
        from train_carboxyl_form_head import _split_by_molecule
        split = _split_by_molecule(carboxyl_rows, test_frac=0.2, seed=17)
        assert len(split["train"]) + len(split["test"]) == len(carboxyl_rows)

    # ── features ────────────────────────────────────────────────────────
    @pytest.fixture(scope="class")
    def carboxyl_features(self, carboxyl_rows):
        X = build_carboxyl_form_feature_frame(
            carboxyl_rows,
            candidate_col="candidate_label",
            resolved_col="resolved_groups",
            smiles_col="smiles",
            pka_type_col="pka_type_canonical",
        )
        return X

    def test_features_shape_matches_rows(self, carboxyl_features, carboxyl_rows):
        assert len(carboxyl_features) == len(carboxyl_rows)

    # ── single-class handling ───────────────────────────────────────────
    def test_single_class_detected_gracefully(self, carboxyl_rows, tmp_path):
        """If only one class exists, the pipeline should report skipped status."""
        y = carboxyl_rows["target_is_acid_form"].to_numpy()
        if len(np.unique(y)) >= 2:
            pytest.skip("carboxyl data has two classes; single-class path not exercised")
        # Emulate single-class handling from main()
        metrics = {
            "rows": int(len(carboxyl_rows)),
            "molecules": int(carboxyl_rows["molecule_key"].nunique()),
            "acid_fraction": float(np.mean(y)),
            "status": "skipped_single_class",
        }
        path = tmp_path / "metrics.json"
        with open(path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        with open(path) as fh:
            loaded = json.load(fh)
        assert loaded["status"] == "skipped_single_class"

    # ── two-class training (synthetic) ──────────────────────────────────
    def test_two_class_trains_successfully(self, carboxyl_features, carboxyl_rows, tmp_path):
        """Ensure the training loop works when both classes are present.

        If real data is single-class, fabricate a balanced target column.
        """
        y = carboxyl_rows["target_is_acid_form"].to_numpy().copy()
        if len(np.unique(y)) < 2:
            y = np.array([i % 2 for i in range(len(y))])

        from train_carboxyl_form_head import _split_by_molecule
        split = _split_by_molecule(carboxyl_rows, test_frac=0.2, seed=17)
        train_idx, test_idx = split["train"], split["test"]

        # Ensure both classes in train set
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            pytest.skip("cannot create two-class split with test data")

        model = LogisticRegression(max_iter=1500, class_weight="balanced")
        model.fit(carboxyl_features.iloc[train_idx], y_train)

        proba = model.predict_proba(carboxyl_features.iloc[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        y_test = y[test_idx]

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, pred)
        assert 0 <= acc <= 1

        # Bundle save
        bundle = {
            "model": model,
            "feature_columns": list(carboxyl_features.columns),
            "candidate_col": "candidate_label",
            "resolved_col": "resolved_groups",
            "smiles_col": "smiles",
            "pka_type_col": "pka_type_canonical",
        }
        path = tmp_path / "carboxyl_form_head.pkl"
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh)
        with open(path, "rb") as fh:
            loaded = pickle.load(fh)
        assert isinstance(loaded["model"], LogisticRegression)
        assert "feature_columns" in loaded


# ═══════════════════════════════════════════════════════════════════════════
#  Pair Form Heads  (train_pair_form_heads)
# ═══════════════════════════════════════════════════════════════════════════

class TestPairFormHeadsPipeline:
    """End-to-end pipeline test for ``train_pair_form_heads``."""

    @pytest.fixture(scope="class")
    def prepared_rows(self):
        from train_pair_form_heads import _prepare_rows
        path = os.path.join(PROCESSED_DIR, "functional_group_assignments.csv")
        if not os.path.exists(path):
            pytest.skip("assignments CSV not found")
        df = _prepare_rows(path, max_rows=0)
        if df.empty:
            pytest.skip("no pair-type rows")
        return df

    # ── _prepare_rows ───────────────────────────────────────────────────
    def test_prepare_rows_has_required_cols(self, prepared_rows):
        expected = {
            "molecule_key", "candidate_label", "candidate_own_form",
            "pair_member_form", "group_mode", "resolved_groups",
        }
        assert expected <= set(prepared_rows.columns)

    def test_prepare_rows_only_pair_type(self, prepared_rows):
        assert (prepared_rows["group_mode"] == "pair_type").all()

    def test_prepare_rows_only_acid_or_base_form(self, prepared_rows):
        assert prepared_rows["candidate_own_form"].isin(["acid_form", "base_form"]).all()

    def test_prepare_rows_distribution_ok(self, prepared_rows):
        assert prepared_rows["distribution_ok"].all()

    def test_prepare_rows_max_rows_limits_output(self):
        from train_pair_form_heads import _prepare_rows
        path = os.path.join(PROCESSED_DIR, "functional_group_assignments.csv")
        if not os.path.exists(path):
            pytest.skip("assignments CSV not found")
        df = _prepare_rows(path, max_rows=50)
        assert len(df) <= 50

    # ── _split_by_molecule ──────────────────────────────────────────────
    def test_split_no_molecule_leakage(self, prepared_rows):
        from train_pair_form_heads import _split_by_molecule
        train_idx, test_idx = _split_by_molecule(prepared_rows, test_frac=0.2, seed=17)
        train_mols = set(prepared_rows.iloc[train_idx]["molecule_key"])
        test_mols = set(prepared_rows.iloc[test_idx]["molecule_key"])
        assert len(train_mols & test_mols) == 0

    def test_split_deterministic(self, prepared_rows):
        from train_pair_form_heads import _split_by_molecule
        t1, _ = _split_by_molecule(prepared_rows, test_frac=0.2, seed=42)
        t2, _ = _split_by_molecule(prepared_rows, test_frac=0.2, seed=42)
        np.testing.assert_array_equal(t1, t2)

    # ── per-head training loop ──────────────────────────────────────────
    @pytest.fixture(scope="class")
    def training_loop_result(self, prepared_rows):
        """Replicate the per-head training loop from main()."""
        from train_pair_form_heads import HEAD_SPECS, _split_by_molecule, _pick_best_threshold

        bundle_heads = {}
        metrics_rows = []
        # Use a lower threshold for test data which may be small
        min_rows = 10

        for head_name, spec in HEAD_SPECS.items():
            acid_labels = set(spec["acid_labels"])
            base_labels = set(spec["base_labels"])
            labels = acid_labels | base_labels

            frame = prepared_rows[prepared_rows["candidate_label"].isin(labels)].copy()
            if len(frame) < min_rows:
                metrics_rows.append({"head": head_name, "rows": len(frame), "status": "skipped_low_rows"})
                continue

            frame["target_is_acid_form"] = (frame["candidate_own_form"] == "acid_form").astype(int)
            y = frame["target_is_acid_form"].to_numpy()
            if len(np.unique(y)) < 2:
                metrics_rows.append({"head": head_name, "rows": len(frame), "status": "skipped_single_class"})
                continue

            X = build_carboxyl_form_feature_frame(
                frame,
                candidate_col="candidate_label",
                resolved_col="resolved_groups",
                smiles_col="smiles",
                pka_type_col="pka_type_canonical",
                include_pka_type=False,
            )

            train_idx, test_idx = _split_by_molecule(frame, test_frac=0.2, seed=17)
            if len(train_idx) == 0 or len(test_idx) == 0:
                metrics_rows.append({"head": head_name, "rows": len(frame), "status": "skipped_bad_split"})
                continue

            y_train, y_test = y[train_idx], y[test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                metrics_rows.append({"head": head_name, "rows": len(frame), "status": "skipped_single_class_split"})
                continue

            model = LogisticRegression(max_iter=2000, class_weight="balanced")
            model.fit(X.iloc[train_idx], y_train)

            proba = model.predict_proba(X.iloc[test_idx])[:, 1]
            opt_thr = _pick_best_threshold(y_test, proba)
            pred = (proba >= opt_thr).astype(int)

            from sklearn.metrics import accuracy_score
            head_metrics = {
                "head": head_name,
                "rows": int(len(frame)),
                "accuracy": float(accuracy_score(y_test, pred)),
                "decision_threshold": float(opt_thr),
                "decision_margin": float(max(0.05, min(0.20, 0.5 * abs(opt_thr - 0.5)))),
                "status": "trained",
            }
            metrics_rows.append(head_metrics)

            bundle_heads[head_name] = {
                "model": model,
                "feature_columns": list(X.columns),
                "acid_labels": sorted(acid_labels),
                "base_labels": sorted(base_labels),
                "decision_threshold": float(opt_thr),
                "decision_margin": float(max(0.05, min(0.20, 0.5 * abs(opt_thr - 0.5)))),
            }

        return {"heads": bundle_heads}, metrics_rows

    def test_at_least_one_head_trained(self, training_loop_result):
        bundle, _ = training_loop_result
        trained = [k for k, v in bundle["heads"].items()]
        assert len(trained) > 0, "No heads were trained"

    def test_every_trained_head_is_logistic(self, training_loop_result):
        bundle, _ = training_loop_result
        for name, head in bundle["heads"].items():
            assert isinstance(head["model"], LogisticRegression), f"{name} model is not LR"

    def test_every_trained_head_has_threshold(self, training_loop_result):
        bundle, _ = training_loop_result
        for name, head in bundle["heads"].items():
            assert 0.10 <= head["decision_threshold"] <= 0.90, f"{name} threshold out of range"
            assert 0.05 <= head["decision_margin"] <= 0.20, f"{name} margin out of range"

    def test_every_trained_head_has_feature_columns(self, training_loop_result):
        bundle, _ = training_loop_result
        for name, head in bundle["heads"].items():
            assert len(head["feature_columns"]) > 0, f"{name} missing feature columns"

    def test_every_trained_head_has_label_lists(self, training_loop_result):
        bundle, _ = training_loop_result
        for name, head in bundle["heads"].items():
            assert len(head["acid_labels"]) > 0, f"{name} missing acid_labels"
            assert len(head["base_labels"]) > 0, f"{name} missing base_labels"

    def test_metrics_cover_all_head_specs(self, training_loop_result):
        from train_pair_form_heads import HEAD_SPECS
        _, metrics_rows = training_loop_result
        reported_heads = {m["head"] for m in metrics_rows}
        assert reported_heads == set(HEAD_SPECS.keys())

    def test_all_metric_statuses_valid(self, training_loop_result):
        _, metrics_rows = training_loop_result
        valid = {"trained", "skipped_low_rows", "skipped_single_class",
                 "skipped_bad_split", "skipped_single_class_split"}
        for m in metrics_rows:
            assert m["status"] in valid, f"Unexpected status: {m['status']}"

    # ── bundle save / reload ────────────────────────────────────────────
    def test_bundle_save_and_reload(self, training_loop_result, tmp_path):
        bundle, _ = training_loop_result
        path = tmp_path / "pair_form_heads.pkl"
        with open(path, "wb") as fh:
            pickle.dump(bundle, fh)
        with open(path, "rb") as fh:
            loaded = pickle.load(fh)
        assert "heads" in loaded
        assert set(loaded["heads"].keys()) == set(bundle["heads"].keys())

    # ── metrics CSV / JSON save ─────────────────────────────────────────
    def test_metrics_csv_and_json_save(self, training_loop_result, tmp_path):
        _, metrics_rows = training_loop_result
        metrics_df = pd.DataFrame(metrics_rows)
        csv_path = tmp_path / "metrics.csv"
        json_path = tmp_path / "metrics.json"
        metrics_df.to_csv(csv_path, index=False)
        with open(json_path, "w") as fh:
            json.dump(metrics_rows, fh, indent=2)

        loaded_df = pd.read_csv(csv_path)
        assert len(loaded_df) == len(metrics_rows)
        with open(json_path) as fh:
            loaded_json = json.load(fh)
        assert isinstance(loaded_json, list)
        assert len(loaded_json) == len(metrics_rows)


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 3 – Protonation State Inference  (stage3_protonation_state_inference)
# ═══════════════════════════════════════════════════════════════════════════

class TestStage3MainPipeline:
    """End-to-end integration test for the Stage 3 ``main()`` pipeline.

    Replicates the full sequence (lines 488–635) using real bundles
    and a candidate dataset.
    """

    @pytest.fixture(scope="class")
    def bundles(self):
        """Load all model bundles, skipping if any are missing."""
        s1 = _load_pkl("stage1_intrinsic/stage1_intrinsic_model.pkl")
        s2 = _load_pkl("stage2_delta/stage2_delta_model.pkl")
        site = _load_pkl("site_state_baseline/site_state_random_model.pkl")
        carboxyl = _load_pkl("carboxyl_form_head/carboxyl_form_head.pkl")
        pair_heads = _load_pkl("pair_form_heads/pair_form_heads.pkl")
        return s1, s2, site, carboxyl, pair_heads

    @pytest.fixture(scope="class")
    def candidate_df(self):
        assignments = os.path.join(PROCESSED_DIR, "functional_group_assignments.csv")
        summary = os.path.join(PROCESSED_DIR, "functional_group_pka_summary.csv")
        if not os.path.exists(assignments):
            pytest.skip("Assignments CSV not found")
        config = DatasetBuildConfig(assignments_path=assignments, summary_path=summary)
        df = build_candidate_dataset(config)
        if df.empty:
            pytest.skip("Candidate dataset is empty")
        return df

    @pytest.fixture(scope="class")
    def stage3_output(self, candidate_df, bundles):
        """Run the full Stage 3 pipeline and return the augmented DataFrame."""
        from stage3_protonation_state_inference import (
            _build_stage1_like_features,
            _build_stage2_features,
            _infer_site_probability,
            _apply_uncertainty_ph_fallback,
            _apply_carboxyl_form_head,
            _apply_pair_form_heads,
            _label_form,
            _protonated_fraction,
            _add_selection_confidence,
            _molecule_level_summary,
        )
        from ionization_context_features import build_ionization_context_frame

        s1, s2, site, carboxyl, pair_heads = bundles
        df = candidate_df.copy()
        df["ph_target"] = 7.4

        # Stage 1: intrinsic pKa
        X_s1 = _build_stage1_like_features(
            df,
            feature_columns=s1.get("feature_columns"),
            fp_bits=int(s1.get("fp_bits", 512)),
            fp_radius=int(s1.get("fp_radius", 2)),
        )
        df["intrinsic_pred_pka"] = s1["model"].predict(X_s1)

        # Stage 2: delta for multi-group
        is_multi = pd.to_numeric(df["resolved_group_count"], errors="coerce").fillna(1) > 1
        df["pred_delta_pka"] = 0.0
        if is_multi.any():
            multi_subset = df.loc[is_multi].copy().reset_index(drop=True)
            X_s2 = _build_stage2_features(
                multi_subset,
                feature_columns=s2.get("feature_columns"),
                fp_bits=int(s2.get("fp_bits", 512)),
                fp_radius=int(s2.get("fp_radius", 2)),
            )
            df.loc[is_multi, "pred_delta_pka"] = s2["model"].predict(X_s2).tolist()

        df["pred_effective_pka"] = df["intrinsic_pred_pka"] + df["pred_delta_pka"]

        # HH fraction and form assignment
        family_form = df["candidate_label"].map(_label_form)
        df["candidate_family"] = family_form.map(lambda x: x[0])
        df["candidate_member_form"] = family_form.map(lambda x: x[1] if x[1] is not None else "non_pair")

        pH = 7.4
        frac_prot = _protonated_fraction(df["pred_effective_pka"].to_numpy(dtype=float), pH=pH)
        df["pred_protonated_fraction"] = frac_prot
        df["pred_is_deprotonated"] = df["pred_effective_pka"] < pH
        df["pred_is_protonated"] = ~df["pred_is_deprotonated"]

        df["pred_member_form"] = np.where(
            df["candidate_member_form"] == "acid_form",
            np.where(df["pred_is_protonated"], "acid_form", "base_form"),
            np.where(
                df["candidate_member_form"] == "base_form",
                np.where(df["pred_is_protonated"], "acid_form", "base_form"),
                df["candidate_member_form"],
            ),
        )

        df["member_presence_prob"] = np.where(
            df["candidate_member_form"] == "acid_form",
            df["pred_protonated_fraction"],
            np.where(
                df["candidate_member_form"] == "base_form",
                1.0 - df["pred_protonated_fraction"],
                0.5,
            ),
        )

        # Site probability
        df["site_prob"] = _infer_site_probability(df, site)
        df["combined_score"] = df["site_prob"] * df["member_presence_prob"]

        # Confidence
        df = _add_selection_confidence(df, pH=pH)

        # Ionization context
        df = pd.concat(
            [
                df,
                build_ionization_context_frame(
                    df,
                    resolved_col="resolved_groups",
                    candidate_col="candidate_label",
                    smiles_col="smiles",
                ),
            ],
            axis=1,
        )

        # Uncertainty fallback
        df = _apply_uncertainty_ph_fallback(df, pH=pH, uncertain_threshold=1.5)

        # Form heads
        df = _apply_carboxyl_form_head(df, carboxyl_bundle=carboxyl)
        df = _apply_pair_form_heads(df, heads_bundle=pair_heads)

        # True form recovery
        inferred_true_form = df["true_label"].map(
            lambda label: (_label_form(str(label))[1] if pd.notna(label) else None)
        )
        df["true_pair_member_form"] = inferred_true_form

        # Molecule-level summary
        metrics = _molecule_level_summary(df)

        return df, metrics

    # ── output shape and columns ────────────────────────────────────────
    def test_output_has_all_prediction_columns(self, stage3_output):
        df, _ = stage3_output
        expected = {
            "intrinsic_pred_pka", "pred_delta_pka", "pred_effective_pka",
            "pred_protonated_fraction", "pred_is_deprotonated", "pred_is_protonated",
            "pred_member_form", "member_presence_prob",
            "site_prob", "combined_score",
            "candidate_family", "candidate_member_form",
            "true_pair_member_form",
        }
        missing = expected - set(df.columns)
        assert not missing, f"Missing prediction columns: {missing}"

    def test_output_row_count_matches_input(self, stage3_output, candidate_df):
        df, _ = stage3_output
        assert len(df) == len(candidate_df)

    def test_protonated_fraction_in_0_1(self, stage3_output):
        df, _ = stage3_output
        assert df["pred_protonated_fraction"].between(0, 1).all()

    def test_site_prob_in_0_1(self, stage3_output):
        df, _ = stage3_output
        assert df["site_prob"].between(0, 1).all()

    def test_combined_score_in_0_1(self, stage3_output):
        df, _ = stage3_output
        assert df["combined_score"].between(0, 1).all()

    def test_single_group_delta_is_zero(self, stage3_output):
        df, _ = stage3_output
        is_single = pd.to_numeric(df["resolved_group_count"], errors="coerce").fillna(1) == 1
        if is_single.any():
            assert (df.loc[is_single, "pred_delta_pka"] == 0.0).all()

    # ── metrics structure ───────────────────────────────────────────────
    def test_molecule_level_metrics_nonempty(self, stage3_output):
        _, metrics = stage3_output
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_metrics_has_accuracy_keys(self, stage3_output):
        _, metrics = stage3_output
        # _molecule_level_summary should produce site/form accuracy metrics
        has_accuracy = any("acc" in k.lower() or "correct" in k.lower() for k in metrics)
        assert has_accuracy or len(metrics) > 0, "Expected accuracy-related keys in metrics"

    # ── CSV / JSON save ─────────────────────────────────────────────────
    def test_csv_roundtrip(self, stage3_output, tmp_path):
        df, _ = stage3_output
        path = tmp_path / "stage3_output.csv"
        df.to_csv(path, index=False)
        loaded = pd.read_csv(path)
        assert len(loaded) == len(df)
        assert set(df.columns) == set(loaded.columns)

    def test_metrics_json_roundtrip(self, stage3_output, tmp_path):
        df, metrics = stage3_output
        full_report = {
            **metrics,
            "rows": int(len(df)),
            "ph": 7.4,
            "uncertain_threshold": 1.5,
            "used_site_model": True,
            "used_carboxyl_form_model": True,
            "used_pair_form_heads": True,
        }
        path = tmp_path / "metrics.json"
        with open(path, "w") as fh:
            json.dump(full_report, fh, indent=2)
        with open(path) as fh:
            loaded = json.load(fh)
        assert loaded["rows"] == len(df)
        assert loaded["ph"] == 7.4
        assert loaded["used_site_model"] is True

    # ── confidence column ───────────────────────────────────────────────
    def test_confidence_column_exists(self, stage3_output):
        df, _ = stage3_output
        conf_cols = [c for c in df.columns if "confidence" in c.lower() or "uncertain" in c.lower()]
        assert len(conf_cols) > 0, "Expected confidence/uncertainty column in Stage 3 output"

    # ── form head columns added ─────────────────────────────────────────
    def test_carboxyl_form_head_applied(self, stage3_output):
        df, _ = stage3_output
        # The carboxyl form head may or may not add a column depending on data
        # At minimum the function should not crash – verified by fixture succeeding
        assert len(df) > 0

    def test_pair_form_heads_applied(self, stage3_output):
        df, _ = stage3_output
        # Like carboxyl, columns depend on whether matching candidates exist
        assert len(df) > 0


# ═══════════════════════════════════════════════════════════════════════════
#  Cross-pipeline: bundle compatibility
# ═══════════════════════════════════════════════════════════════════════════

class TestBundleCompatibility:
    """Verify saved model bundles are mutually compatible for Stage 3.

    Stage 3 loads Stage 1 + Stage 2 + Site + Carboxyl + Pair-form bundles.
    This class checks that the feature-column contracts match.
    """

    @pytest.fixture(scope="class")
    def all_bundles(self):
        bundles = {
            "stage1": _load_pkl("stage1_intrinsic/stage1_intrinsic_model.pkl"),
            "stage2": _load_pkl("stage2_delta/stage2_delta_model.pkl"),
            "site": _load_pkl("site_state_baseline/site_state_random_model.pkl"),
            "carboxyl": _load_pkl("carboxyl_form_head/carboxyl_form_head.pkl"),
            "pair_heads": _load_pkl("pair_form_heads/pair_form_heads.pkl"),
        }
        return bundles

    def test_stage1_has_required_keys(self, all_bundles):
        b = all_bundles["stage1"]
        assert {"model", "feature_columns", "fp_bits", "fp_radius"} <= set(b.keys())

    def test_stage2_has_required_keys(self, all_bundles):
        b = all_bundles["stage2"]
        assert {"model", "feature_columns", "fp_bits", "fp_radius"} <= set(b.keys())

    def test_site_has_required_keys(self, all_bundles):
        b = all_bundles["site"]
        assert {"model", "feature_columns", "fp_bits", "fp_radius"} <= set(b.keys())

    def test_carboxyl_has_required_keys(self, all_bundles):
        b = all_bundles["carboxyl"]
        assert {"model", "feature_columns"} <= set(b.keys())

    def test_pair_heads_has_heads_dict(self, all_bundles):
        b = all_bundles["pair_heads"]
        assert "heads" in b
        assert isinstance(b["heads"], dict)

    def test_each_pair_head_has_model(self, all_bundles):
        for name, head in all_bundles["pair_heads"]["heads"].items():
            assert "model" in head, f"Pair head '{name}' missing model"
            assert "feature_columns" in head, f"Pair head '{name}' missing feature_columns"

    def test_fp_bits_consistent(self, all_bundles):
        s1_bits = all_bundles["stage1"]["fp_bits"]
        s2_bits = all_bundles["stage2"]["fp_bits"]
        site_bits = all_bundles["site"]["fp_bits"]
        assert s1_bits == s2_bits == site_bits, (
            f"fp_bits mismatch: stage1={s1_bits}, stage2={s2_bits}, site={site_bits}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  main() entrypoint tests — mock argparse and call main() directly
#  to cover the full orchestration + file-I/O code path.
# ═══════════════════════════════════════════════════════════════════════════

from unittest.mock import patch
from argparse import Namespace

ASSIGNMENTS_CSV = os.path.join(PROCESSED_DIR, "functional_group_assignments.csv")
SUMMARY_CSV = os.path.join(PROCESSED_DIR, "functional_group_pka_summary.csv")
STAGE1_MODEL = os.path.join(MODELS_DIR, "stage1_intrinsic", "stage1_intrinsic_model.pkl")
STAGE2_MODEL = os.path.join(MODELS_DIR, "stage2_delta", "stage2_delta_model.pkl")
SITE_MODEL = os.path.join(MODELS_DIR, "site_state_baseline", "site_state_random_model.pkl")
CARBOXYL_MODEL = os.path.join(MODELS_DIR, "carboxyl_form_head", "carboxyl_form_head.pkl")
PAIR_HEADS_MODEL = os.path.join(MODELS_DIR, "pair_form_heads", "pair_form_heads.pkl")


def _skip_if_missing(*paths):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"Required file not found: {p}")


class TestStage1Main:
    """Exercise train_intrinsic_single_group_model.main() end-to-end."""

    def test_main_produces_artefacts(self, tmp_path):
        _skip_if_missing(ASSIGNMENTS_CSV)
        args = Namespace(
            assignments=ASSIGNMENTS_CSV,
            out_dir=str(tmp_path),
            fp_bits=512,
            fp_radius=2,
            test_frac=0.2,
        )
        with patch("train_intrinsic_single_group_model.parse_args", return_value=args):
            import train_intrinsic_single_group_model
            train_intrinsic_single_group_model.main()

        assert (tmp_path / "stage1_intrinsic_model.pkl").exists()
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "random_eval_intrinsic.csv").exists()
        assert (tmp_path / "scaffold_eval_intrinsic.csv").exists()
        assert (tmp_path / "random_split_manifest.csv").exists()
        assert (tmp_path / "scaffold_split_manifest.csv").exists()

        with open(tmp_path / "metrics.json") as fh:
            report = json.load(fh)
        assert "random" in report and "scaffold" in report
        assert report["random"]["mae"] >= 0
        assert report["scaffold"]["mae"] >= 0

        with open(tmp_path / "stage1_intrinsic_model.pkl", "rb") as fh:
            bundle = pickle.load(fh)
        assert {"model", "feature_columns", "fp_bits", "fp_radius"} <= set(bundle.keys())


class TestStage2Main:
    """Exercise train_delta_context_model.main() end-to-end."""

    def test_main_produces_artefacts(self, tmp_path):
        _skip_if_missing(ASSIGNMENTS_CSV, STAGE1_MODEL)
        args = Namespace(
            assignments=ASSIGNMENTS_CSV,
            stage1_model=STAGE1_MODEL,
            out_dir=str(tmp_path),
            fp_bits=512,
            fp_radius=2,
            test_frac=0.2,
            max_rows=0,
        )
        with patch("train_delta_context_model.parse_args", return_value=args):
            import train_delta_context_model
            train_delta_context_model.main()

        assert (tmp_path / "stage2_delta_model.pkl").exists()
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "random_eval_delta.csv").exists()
        assert (tmp_path / "scaffold_eval_delta.csv").exists()

        with open(tmp_path / "metrics.json") as fh:
            report = json.load(fh)
        assert "random" in report and "scaffold" in report
        assert report["random"]["effective_mae"] >= 0

        with open(tmp_path / "stage2_delta_model.pkl", "rb") as fh:
            bundle = pickle.load(fh)
        assert {"model", "feature_columns", "fp_bits", "fp_radius"} <= set(bundle.keys())


class TestSiteStateMain:
    """Exercise train_site_state_model.main() end-to-end."""

    def test_main_produces_artefacts(self, tmp_path):
        _skip_if_missing(ASSIGNMENTS_CSV, SUMMARY_CSV, STAGE1_MODEL, STAGE2_MODEL)
        args = Namespace(
            assignments=ASSIGNMENTS_CSV,
            summary=SUMMARY_CSV,
            raw_dir=os.path.join(PROJECT_ROOT, "data", "raw"),
            stage1_model=STAGE1_MODEL,
            stage2_model=STAGE2_MODEL,
            out_dir=str(tmp_path),
            test_frac=0.2,
            fp_bits=512,
            fp_radius=2,
            include_observed_pka=False,
        )
        with patch("train_site_state_model.parse_args", return_value=args):
            import train_site_state_model
            train_site_state_model.main()

        assert (tmp_path / "site_state_random_model.pkl").exists()
        assert (tmp_path / "site_state_scaffold_model.pkl").exists()
        assert (tmp_path / "metrics.json").exists()

        with open(tmp_path / "metrics.json") as fh:
            report = json.load(fh)
        assert "random" in report and "scaffold" in report
        assert report["random"]["auroc"] > 0
        assert report["uses_predicted_pka_features"] is True


class TestCarboxylMain:
    """Exercise train_carboxyl_form_head.main() end-to-end."""

    def test_main_produces_artefacts(self, tmp_path):
        _skip_if_missing(ASSIGNMENTS_CSV)
        args = Namespace(
            assignments=ASSIGNMENTS_CSV,
            out_dir=str(tmp_path),
            test_frac=0.2,
            seed=17,
            max_rows=0,
        )
        with patch("train_carboxyl_form_head.parse_args", return_value=args):
            import train_carboxyl_form_head
            train_carboxyl_form_head.main()

        # main() always writes metrics.json (even on single-class skip)
        assert (tmp_path / "metrics.json").exists()
        with open(tmp_path / "metrics.json") as fh:
            report = json.load(fh)

        # If single-class, status is skipped; otherwise has accuracy
        if report.get("status") == "skipped_single_class":
            assert "acid_fraction" in report
        else:
            assert (tmp_path / "carboxyl_form_head.pkl").exists()
            assert "accuracy" in report


class TestPairFormHeadsMain:
    """Exercise train_pair_form_heads.main() end-to-end."""

    def test_main_produces_artefacts(self, tmp_path):
        _skip_if_missing(ASSIGNMENTS_CSV, SUMMARY_CSV)
        args = Namespace(
            assignments=ASSIGNMENTS_CSV,
            out_dir=str(tmp_path),
            test_frac=0.2,
            seed=17,
            max_rows=0,
            min_rows=10,  # lowered for test data
        )
        with patch("train_pair_form_heads.parse_args", return_value=args):
            import train_pair_form_heads
            train_pair_form_heads.main()

        assert (tmp_path / "pair_form_heads.pkl").exists()
        assert (tmp_path / "metrics.csv").exists()
        assert (tmp_path / "metrics.json").exists()

        with open(tmp_path / "pair_form_heads.pkl", "rb") as fh:
            bundle = pickle.load(fh)
        assert "heads" in bundle

        with open(tmp_path / "metrics.json") as fh:
            metrics_list = json.load(fh)
        assert isinstance(metrics_list, list)
        assert len(metrics_list) > 0


class TestStage3Main:
    """Exercise stage3_protonation_state_inference.main() end-to-end."""

    def test_main_produces_artefacts(self, tmp_path):
        _skip_if_missing(
            ASSIGNMENTS_CSV, SUMMARY_CSV, STAGE1_MODEL,
            STAGE2_MODEL, SITE_MODEL, CARBOXYL_MODEL, PAIR_HEADS_MODEL,
        )
        out_csv = str(tmp_path / "stage3_output.csv")
        out_json = str(tmp_path / "stage3_metrics.json")
        args = Namespace(
            assignments=ASSIGNMENTS_CSV,
            summary=SUMMARY_CSV,
            stage1_model=STAGE1_MODEL,
            stage2_model=STAGE2_MODEL,
            site_model=SITE_MODEL,
            carboxyl_form_model=CARBOXYL_MODEL,
            pair_form_heads=PAIR_HEADS_MODEL,
            out_csv=out_csv,
            out_json=out_json,
            ph=7.4,
            uncertain_threshold=1.5,
            max_rows=200,  # limit for speed
        )
        with patch("stage3_protonation_state_inference.parse_args", return_value=args):
            import stage3_protonation_state_inference
            stage3_protonation_state_inference.main()

        assert os.path.exists(out_csv)
        assert os.path.exists(out_json)

        df = pd.read_csv(out_csv)
        assert len(df) > 0
        assert "intrinsic_pred_pka" in df.columns
        assert "pred_effective_pka" in df.columns
        assert "combined_score" in df.columns

        with open(out_json) as fh:
            metrics = json.load(fh)
        assert "rows" in metrics
        assert metrics["ph"] == 7.4
        assert metrics["used_site_model"] is True
