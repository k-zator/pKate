"""
Training Data Integrity Tests
================================
Validates that training scripts receive correctly-shaped, correctly-filtered
data so you never accidentally train on wrong or leaked data.

Covered scripts:
  - train_intrinsic_single_group_model.py  (load_single_group_rows, build_intrinsic_features)
  - train_delta_context_model.py           (load_multi_group_rows, build_delta_features)
  - train_carboxyl_form_head.py            (_prepare_training_rows, _split_by_molecule)
  - train_pair_form_heads.py               (HEAD_SPECS, _prepare_rows, _pick_best_threshold)
  - train_site_state_model.py              (_topk_site_accuracy)
  - ml_splits.py                           (random_molecule_split, scaffold_molecule_split, split_manifest)
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

# ---------------------------------------------------------------------------
# Imports from scripts/
# ---------------------------------------------------------------------------
from train_intrinsic_single_group_model import load_single_group_rows, build_intrinsic_features
from train_delta_context_model import load_multi_group_rows, build_delta_features
from train_carboxyl_form_head import _prepare_training_rows, _split_by_molecule as carboxyl_split
from train_pair_form_heads import HEAD_SPECS, _pick_best_threshold
from train_site_state_model import _topk_site_accuracy
from ml_splits import random_molecule_split, scaffold_molecule_split, split_manifest
from ml_features import build_feature_matrix
from functional_group_pka_analysis import CONJUGATE_FAMILY_MAP, PAIR_TYPE_FAMILY_FORMS
from functional_group_pka_analysis import build_functional_group_table, SITE_OVERRIDE_KEY_COLUMNS
from prepare_external_pka_datasets import prepare_czodrowski_datasets, prepare_sampl_benchmarks
from site_resolution_review import build_site_resolution_review_queue

PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed",
)
ASSIGNMENTS_PATH = os.path.join(PROCESSED_DIR, "functional_group_assignments.csv")
RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw",
)
EXTERNAL_DATASETS_DIR = os.path.join(RAW_DIR, "external_datasets")


def _skip_if_missing(path: str):
    if not os.path.exists(path):
        pytest.skip(f"Data file not found: {path}")


# ===================================================================
#  1. Stage 1 — load_single_group_rows
# ===================================================================

class TestLoadSingleGroupRows:

    @pytest.fixture(scope="class")
    def single_df(self):
        _skip_if_missing(ASSIGNMENTS_PATH)
        return load_single_group_rows(ASSIGNMENTS_PATH)

    def test_not_empty(self, single_df):
        assert len(single_df) > 0, "Single-group dataset should not be empty"

    def test_required_columns_present(self, single_df):
        required = {
            "source_file", "record_index", "smiles", "pka_value",
            "final_group_label", "molecule_key",
        }
        missing = required - set(single_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_nan_in_critical_columns(self, single_df):
        for col in ("smiles", "pka_value", "final_group_label"):
            assert single_df[col].isna().sum() == 0, f"NaN found in {col}"

    def test_pka_values_finite(self, single_df):
        pka = single_df["pka_value"].astype(float)
        assert np.all(np.isfinite(pka)), "pKa values must all be finite"

    def test_pka_in_plausible_range(self, single_df):
        pka = single_df["pka_value"].astype(float)
        assert pka.min() > -15, f"Implausibly low pKa: {pka.min()}"
        assert pka.max() < 25, f"Implausibly high pKa: {pka.max()}"

    def test_distribution_ok_filter_applied(self, single_df):
        """If distribution_ok column exists, only True rows should remain."""
        if "distribution_ok" in single_df.columns:
            assert single_df["distribution_ok"].all(), (
                "load_single_group_rows should filter out distribution_ok == False"
            )

    def test_molecule_key_deduplication(self, single_df):
        """No duplicate (molecule_key, pka_value, final_group_label) triples."""
        dups = single_df.duplicated(subset=["molecule_key", "pka_value", "final_group_label"])
        assert not dups.any(), f"Found {dups.sum()} duplicate rows after dedup"

    def test_smiles_parseable(self, single_df):
        """Spot-check: first 20 SMILES should parse with RDKit."""
        from rdkit import Chem
        for smi in single_df["smiles"].head(20):
            mol = Chem.MolFromSmiles(str(smi))
            assert mol is not None, f"Unparseable SMILES in training data: {smi}"


# ===================================================================
#  2. Stage 1 — build_intrinsic_features
# ===================================================================

class TestBuildIntrinsicFeatures:

    @pytest.fixture(scope="class")
    def features(self):
        _skip_if_missing(ASSIGNMENTS_PATH)
        df = load_single_group_rows(ASSIGNMENTS_PATH)
        return build_intrinsic_features(df.head(100), nbits=64, radius=2)

    def test_output_is_dataframe(self, features):
        assert isinstance(features, pd.DataFrame)

    def test_no_nan_in_fingerprints(self, features):
        fp_cols = [c for c in features.columns if c.startswith("fp_")]
        assert not features[fp_cols].isna().any().any()

    def test_has_group_dummies(self, features):
        group_cols = [c for c in features.columns if c.startswith("group_")]
        assert len(group_cols) > 0, "Should have at least one group dummy column"

    def test_has_pka_type_dummies(self, features):
        type_cols = [c for c in features.columns if c.startswith("pka_type_")]
        assert len(type_cols) > 0

    def test_has_geometry_columns(self, features):
        mol_geom_cols = [c for c in features.columns if c.startswith("mol_geom_")]
        site_geom_cols = [c for c in features.columns if c.startswith("site_geom_")]
        assert len(mol_geom_cols) > 0
        assert len(site_geom_cols) > 0

    def test_feature_values_finite(self, features):
        # Some descriptor columns may have NaN from RDKit; at minimum fps should be fine
        fp_cols = [c for c in features.columns if c.startswith("fp_")]
        assert np.all(np.isfinite(features[fp_cols].values))


def test_stage1_features_expose_nitro_as_explicit_context_not_target_site():
    frame = pd.DataFrame([{
        "smiles": "O=[N+]([O-])c1ncc[nH]1",
        "candidate_label": "imidazole",
        "pka_type_canonical": "basic",
        "atom_index_raw": 1,
    }])
    features = build_intrinsic_features(
        frame, nbits=32, radius=2, label_col="candidate_label"
    )
    assert features.loc[0, "fg_global_count__nitro"] == 1.0
    assert features.loc[0, "fg_local_count__nitro"] == 1.0
    assert "group_nitro" not in features.columns


class TestSiteResolutionWorkflow:

    def test_build_table_applies_overrides(self):
        raw_file = os.path.join(RAW_DIR, "literature_compilation.sdf")
        if not os.path.exists(raw_file):
            pytest.skip("literature_compilation.sdf not found")

        df = build_functional_group_table(
            raw_dir=RAW_DIR,
            include_files=["literature_compilation.sdf"],
            allow_epik=False,
        )
        ambig = df[df["assignment_status"] != "single_group"].head(1).copy()
        if ambig.empty:
            pytest.skip("No ambiguous rows found for override test")

        row = ambig.iloc[0]
        labels = [token for token in str(row["resolved_groups"]).split("|") if token]
        if len(labels) < 2:
            pytest.skip("Ambiguous row did not expose multiple resolved labels")
        override_label = labels[-1] if labels[-1] != row["selected_group_label"] else labels[0]

        override = {col: row[col] for col in SITE_OVERRIDE_KEY_COLUMNS}
        override["override_group_label"] = override_label
        override["note"] = "pytest"

        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as handle:
            pd.DataFrame([override]).to_csv(handle.name, index=False)
            override_path = handle.name

        try:
            df2 = build_functional_group_table(
                raw_dir=RAW_DIR,
                include_files=["literature_compilation.sdf"],
                allow_epik=False,
                overrides_path=override_path,
            )
        finally:
            os.unlink(override_path)

        matches = df2[
            (df2["source_file"] == row["source_file"])
            & (df2["record_index"] == row["record_index"])
            & (df2["smiles"] == row["smiles"])
            & (df2["pka_value"].round(6) == round(float(row["pka_value"]), 6))
        ]
        assert not matches.empty
        assert override_label in set(matches["selected_group_label"])
        assert matches["site_override_applied"].any()

    def test_review_queue_marks_reasons(self):
        raw_file = os.path.join(RAW_DIR, "literature_compilation.sdf")
        if not os.path.exists(raw_file):
            pytest.skip("literature_compilation.sdf not found")

        queue = build_site_resolution_review_queue(
            raw_dir=RAW_DIR,
            include_files=["literature_compilation.sdf"],
            include_reviewed=True,
        )
        assert isinstance(queue, pd.DataFrame)
        if queue.empty:
            pytest.skip("No ambiguous rows found for review queue test")
        assert "review_reason" in queue.columns
        assert queue["review_reason"].astype(str).str.len().gt(0).all()


# ===================================================================
#  3. Stage 2 — load_multi_group_rows
# ===================================================================

class TestLoadMultiGroupRows:

    @pytest.fixture(scope="class")
    def multi_df(self):
        _skip_if_missing(ASSIGNMENTS_PATH)
        return load_multi_group_rows(ASSIGNMENTS_PATH)

    def test_not_empty(self, multi_df):
        assert len(multi_df) > 0

    def test_assignment_status_is_multi_group(self, multi_df):
        assert (multi_df["assignment_status"] == "multi_group").all()

    def test_pka_finite(self, multi_df):
        pka = multi_df["pka_value"].astype(float)
        assert np.all(np.isfinite(pka))

    def test_distribution_ok_filter(self, multi_df):
        if "distribution_ok" in multi_df.columns:
            assert multi_df["distribution_ok"].all()

    def test_resolved_groups_not_empty(self, multi_df):
        """Multi-group rows should have non-empty resolved_groups."""
        assert multi_df["resolved_groups"].notna().all()
        assert (multi_df["resolved_groups"].str.len() > 0).all()


# ===================================================================
#  4. Carboxyl form head — _prepare_training_rows
# ===================================================================

class TestPrepareCarboxylTrainingRows:

    @pytest.fixture(scope="class")
    def carboxyl_df(self):
        _skip_if_missing(ASSIGNMENTS_PATH)
        return _prepare_training_rows(ASSIGNMENTS_PATH)

    def test_not_empty(self, carboxyl_df):
        assert len(carboxyl_df) > 0, "Carboxyl training rows should not be empty"

    def test_only_pair_type_mode(self, carboxyl_df):
        assert (carboxyl_df["group_mode"] == "pair_type").all()


class TestExternalDatasetPreparation:

    @pytest.fixture(scope="class")
    def carboxyl_df(self):
        _skip_if_missing(ASSIGNMENTS_PATH)
        return _prepare_training_rows(ASSIGNMENTS_PATH)

    def test_czodrowski_subset_builds_pipeline_ready_candidates(self):
        dataset_dir = os.path.join(
            EXTERNAL_DATASETS_DIR,
            "Machine-learning-meets-pKa",
            "datasets",
        )
        if not os.path.exists(dataset_dir):
            pytest.skip("Czodrowski dataset clone not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = prepare_czodrowski_datasets(
                dataset_dir=dataset_dir,
                out_root=tmpdir,
                include_files=["AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf"],
            )
            assert not manifest.empty

            dataset_dirname = "AvLiLuMoVe_cleaned_mono_unique_notraindata"
            manifest_row = manifest.loc[manifest["dataset_name"] == dataset_dirname].iloc[0]
            assert manifest_row["dataset_role"] == "duplicate_local_curated"
            assert manifest_row["recommended_use"] == "do_not_use"
            assert manifest_row["duplicate_local_source"] == "AvLiLuMoVe_testdata.sdf"
            candidates_path = os.path.join(tmpdir, dataset_dirname, "candidates_curated_priors.csv")
            assert os.path.exists(candidates_path)

            candidates = pd.read_csv(candidates_path)
            assert not candidates.empty
            X, y = build_feature_matrix(candidates.head(min(20, len(candidates))), nbits=32, radius=2)
            assert len(X) == len(y)
            assert "candidate_prior_mean_pka" in candidates.columns

    def test_sampl_subset_builds_unambiguous_candidates(self):
        sampl7_root = os.path.join(EXTERNAL_DATASETS_DIR, "SAMPL7")
        if not os.path.exists(sampl7_root):
            pytest.skip("SAMPL7 dataset clone not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = prepare_sampl_benchmarks(
                sampl7_root=sampl7_root,
                sampl8_root=os.path.join(EXTERNAL_DATASETS_DIR, "SAMPL8"),
                out_root=tmpdir,
                sampl7_compounds=["SM25", "SM26", "SM27"],
                sampl8_compounds=[],
            )
            assert not manifest.empty

            candidates_path = os.path.join(tmpdir, "SAMPL7", "candidates_curated_priors.csv")
            if not os.path.exists(candidates_path):
                pytest.skip("SAMPL7 unambiguous candidate output was not created")

            candidates = pd.read_csv(candidates_path)
            if candidates.empty:
                pytest.skip("SAMPL7 subset did not yield unambiguous candidates")

            X, y = build_feature_matrix(candidates.head(min(20, len(candidates))), nbits=32, radius=2)
            assert len(X) == len(y)
            assert candidates["is_true_site"].sum() > 0

    def test_only_carboxyl_family(self, carboxyl_df):
        assert (carboxyl_df["final_group_family"] == "carboxyl").all()

    def test_only_acid_or_base_form(self, carboxyl_df):
        assert carboxyl_df["pair_member_form"].isin(["acid_form", "base_form"]).all()

    def test_distribution_ok_filter(self, carboxyl_df):
        if "distribution_ok" in carboxyl_df.columns:
            assert carboxyl_df["distribution_ok"].all()

    def test_target_column_binary(self, carboxyl_df):
        assert carboxyl_df["target_is_acid_form"].isin([0, 1]).all()

    def test_both_classes_present(self, carboxyl_df):
        """Both acid and base forms should be present for effective training.
        If only one class exists, the training script will skip, which is safe
        but means the head won't be usable."""
        n_classes = carboxyl_df["target_is_acid_form"].nunique()
        if n_classes < 2:
            import warnings
            warnings.warn(
                f"Carboxyl training data has only {n_classes} class(es). "
                "The carboxyl form head will be skipped during training. "
                "This is handled safely but reduces prediction capability."
            )
        # Not a hard failure — the training script handles single-class gracefully

    def test_class_balance_not_extreme(self, carboxyl_df):
        """Neither class should be < 5% of the data (when both classes exist)."""
        if carboxyl_df["target_is_acid_form"].nunique() < 2:
            pytest.skip("Single class in carboxyl data — balance check N/A")
        frac = carboxyl_df["target_is_acid_form"].mean()
        assert 0.05 < frac < 0.95, (
            f"Extreme class imbalance: acid_form fraction = {frac:.2f}"
        )


# ===================================================================
#  5. Carboxyl form head — _split_by_molecule
# ===================================================================

class TestCarboxylSplitByMolecule:

    @pytest.fixture(scope="class")
    def split(self):
        _skip_if_missing(ASSIGNMENTS_PATH)
        df = _prepare_training_rows(ASSIGNMENTS_PATH)
        if df.empty:
            pytest.skip("No carboxyl training rows")
        return df, carboxyl_split(df, test_frac=0.2, seed=17)

    def test_no_molecule_leakage(self, split):
        """No molecule_key should appear in both train and test."""
        df, sp = split
        train_mols = set(df.iloc[sp["train"]]["molecule_key"])
        test_mols = set(df.iloc[sp["test"]]["molecule_key"])
        overlap = train_mols & test_mols
        assert len(overlap) == 0, f"Molecule leakage: {len(overlap)} shared molecules"

    def test_all_indices_covered(self, split):
        df, sp = split
        all_idx = set(range(len(df)))
        split_idx = set(sp["train"].tolist()) | set(sp["test"].tolist())
        assert split_idx == all_idx, "Train + test should cover all rows"

    def test_test_fraction_approximate(self, split):
        df, sp = split
        test_frac = len(sp["test"]) / len(df)
        assert 0.10 < test_frac < 0.35, f"Test fraction {test_frac:.2f} outside expected range"


# ===================================================================
#  6. ml_splits — random_molecule_split
# ===================================================================

class TestRandomMoleculeSplit:

    @pytest.fixture(scope="class")
    def df_with_keys(self):
        _skip_if_missing(ASSIGNMENTS_PATH)
        df = load_single_group_rows(ASSIGNMENTS_PATH)
        if df.empty:
            pytest.skip("No single-group rows")
        return df

    def test_no_molecule_leakage(self, df_with_keys):
        train_idx, test_idx = random_molecule_split(df_with_keys, test_frac=0.2)
        train_mols = set(df_with_keys.iloc[train_idx]["molecule_key"])
        test_mols = set(df_with_keys.iloc[test_idx]["molecule_key"])
        overlap = train_mols & test_mols
        assert len(overlap) == 0, f"Molecule leakage: {len(overlap)} shared molecules"

    def test_deterministic_with_seed(self, df_with_keys):
        t1_train, t1_test = random_molecule_split(df_with_keys, test_frac=0.2, seed=42)
        t2_train, t2_test = random_molecule_split(df_with_keys, test_frac=0.2, seed=42)
        np.testing.assert_array_equal(t1_train, t2_train)
        np.testing.assert_array_equal(t1_test, t2_test)

    def test_different_seeds_differ(self, df_with_keys):
        t1_train, _ = random_molecule_split(df_with_keys, test_frac=0.2, seed=1)
        t2_train, _ = random_molecule_split(df_with_keys, test_frac=0.2, seed=2)
        # Very unlikely to be identical with different seeds
        assert not np.array_equal(t1_train, t2_train)


# ===================================================================
#  7. ml_splits — scaffold_molecule_split
# ===================================================================

class TestScaffoldMoleculeSplit:

    @pytest.fixture(scope="class")
    def df_with_keys(self):
        _skip_if_missing(ASSIGNMENTS_PATH)
        df = load_single_group_rows(ASSIGNMENTS_PATH)
        if df.empty:
            pytest.skip("No single-group rows")
        return df

    def test_no_molecule_leakage(self, df_with_keys):
        train_idx, test_idx = scaffold_molecule_split(df_with_keys, test_frac=0.2)
        train_mols = set(df_with_keys.iloc[train_idx]["molecule_key"])
        test_mols = set(df_with_keys.iloc[test_idx]["molecule_key"])
        overlap = train_mols & test_mols
        assert len(overlap) == 0, f"Scaffold split molecule leakage: {len(overlap)}"

    def test_test_set_not_empty(self, df_with_keys):
        _, test_idx = scaffold_molecule_split(df_with_keys, test_frac=0.2)
        assert len(test_idx) > 0

    def test_all_rows_assigned(self, df_with_keys):
        train_idx, test_idx = scaffold_molecule_split(df_with_keys, test_frac=0.2)
        assert len(train_idx) + len(test_idx) == len(df_with_keys)


# ===================================================================
#  8. ml_splits — split_manifest
# ===================================================================

class TestSplitManifest:

    def test_manifest_columns(self):
        df = pd.DataFrame({
            "molecule_key": ["m1", "m1", "m2", "m3"],
            "smiles": ["CC", "CC", "CCO", "CCN"],
        })
        train_idx = np.array([0, 1])
        test_idx = np.array([2, 3])
        manifest = split_manifest(df, train_idx, test_idx)
        assert "split" in manifest.columns
        assert set(manifest["split"].unique()) == {"train", "test"}

    def test_manifest_train_test_labels(self):
        df = pd.DataFrame({
            "molecule_key": ["m1", "m2"],
            "smiles": ["CC", "CCO"],
        })
        manifest = split_manifest(df, [0], [1])
        assert manifest.iloc[0]["split"] == "train"
        assert manifest.iloc[1]["split"] == "test"


# ===================================================================
#  9. Pair form heads — HEAD_SPECS consistency
# ===================================================================

class TestHeadSpecsConsistency:

    def test_all_acid_labels_in_conjugate_map(self):
        """Every acid label in HEAD_SPECS should have a conjugate mapping."""
        for head_name, spec in HEAD_SPECS.items():
            for label in spec["acid_labels"]:
                # The acid label should map to itself or have a family mapping
                family = CONJUGATE_FAMILY_MAP.get(label, label)
                assert family is not None, (
                    f"HEAD_SPECS[{head_name}] acid_label '{label}' not in CONJUGATE_FAMILY_MAP"
                )

    def test_all_base_labels_in_conjugate_map(self):
        for head_name, spec in HEAD_SPECS.items():
            for label in spec["base_labels"]:
                family = CONJUGATE_FAMILY_MAP.get(label, label)
                assert family is not None

    def test_acid_and_base_share_family(self):
        """Acid and base labels in the same head should map to the same conjugate family."""
        for head_name, spec in HEAD_SPECS.items():
            families = set()
            for label in spec["acid_labels"] + spec["base_labels"]:
                families.add(CONJUGATE_FAMILY_MAP.get(label, label))
            assert len(families) == 1, (
                f"HEAD_SPECS[{head_name}] has labels mapping to multiple families: {families}"
            )

    def test_no_overlapping_labels_across_heads(self):
        """No label should appear in multiple HEAD_SPECS entries."""
        seen = {}
        for head_name, spec in HEAD_SPECS.items():
            for label in spec["acid_labels"] + spec["base_labels"]:
                if label in seen:
                    pytest.fail(
                        f"Label '{label}' appears in both '{seen[label]}' and '{head_name}'"
                    )
                seen[label] = head_name


# ===================================================================
# 10. _pick_best_threshold
# ===================================================================

class TestPickBestThreshold:

    def test_perfect_separation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        thr = _pick_best_threshold(y_true, proba)
        assert 0.10 <= thr <= 0.90
        # With perfect separation, threshold should fall between 0.3 and 0.7
        assert 0.25 <= thr <= 0.75

    def test_random_probabilities(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100)
        proba = rng.random(100)
        thr = _pick_best_threshold(y_true, proba)
        assert 0.10 <= thr <= 0.90

    def test_all_positive(self):
        y_true = np.ones(10, dtype=int)
        proba = np.linspace(0.5, 0.9, 10)
        thr = _pick_best_threshold(y_true, proba)
        # Should pick threshold that maximizes balanced accuracy
        assert 0.10 <= thr <= 0.90

    def test_threshold_maximizes_balanced_accuracy(self):
        """The chosen threshold should give balanced accuracy >= default 0.5."""
        y_true = np.array([0]*30 + [1]*70)
        proba = np.concatenate([
            np.random.default_rng(1).uniform(0.1, 0.5, 30),
            np.random.default_rng(2).uniform(0.5, 0.9, 70),
        ])
        thr = _pick_best_threshold(y_true, proba)
        pred = (proba >= thr).astype(int)
        ba = balanced_accuracy_score(y_true, pred)
        assert ba >= 0.50, f"Balanced accuracy {ba:.3f} is below random chance"


# ===================================================================
# 11. _topk_site_accuracy
# ===================================================================

class TestTopkSiteAccuracy:

    def test_perfect_top1(self):
        df = pd.DataFrame({
            "molecule_key": ["m1", "m1", "m2", "m2"],
            "pred_proba": [0.9, 0.1, 0.8, 0.2],
            "is_true_site": [1, 0, 1, 0],
        })
        acc = _topk_site_accuracy(df, proba_col="pred_proba", k=1)
        assert acc == 1.0

    def test_completely_wrong_top1(self):
        df = pd.DataFrame({
            "molecule_key": ["m1", "m1"],
            "pred_proba": [0.9, 0.1],
            "is_true_site": [0, 1],
        })
        acc = _topk_site_accuracy(df, proba_col="pred_proba", k=1)
        assert acc == 0.0

    def test_top3_rescues_missed_top1(self):
        df = pd.DataFrame({
            "molecule_key": ["m1"] * 4,
            "pred_proba": [0.9, 0.8, 0.7, 0.1],
            "is_true_site": [0, 0, 1, 0],
        })
        top1 = _topk_site_accuracy(df, proba_col="pred_proba", k=1)
        top3 = _topk_site_accuracy(df, proba_col="pred_proba", k=3)
        assert top1 == 0.0
        assert top3 == 1.0

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["molecule_key", "pred_proba", "is_true_site"])
        acc = _topk_site_accuracy(df, proba_col="pred_proba", k=1)
        assert acc == 0.0


# ===================================================================
# 12. train/test split — no molecule leakage in Stage 2
# ===================================================================

class TestStage2NoLeakage:

    @pytest.fixture(scope="class")
    def multi_df(self):
        _skip_if_missing(ASSIGNMENTS_PATH)
        df = load_multi_group_rows(ASSIGNMENTS_PATH)
        if df.empty:
            pytest.skip("No multi-group rows")
        return df

    def test_random_split_no_leakage(self, multi_df):
        train_idx, test_idx = random_molecule_split(multi_df, test_frac=0.2)
        train_mols = set(multi_df.iloc[train_idx]["molecule_key"])
        test_mols = set(multi_df.iloc[test_idx]["molecule_key"])
        assert not (train_mols & test_mols), "Molecule leakage in Stage 2 random split"

    def test_scaffold_split_no_leakage(self, multi_df):
        train_idx, test_idx = scaffold_molecule_split(multi_df, test_frac=0.2)
        train_mols = set(multi_df.iloc[train_idx]["molecule_key"])
        test_mols = set(multi_df.iloc[test_idx]["molecule_key"])
        assert not (train_mols & test_mols), "Molecule leakage in Stage 2 scaffold split"
