"""Data-independent tests for canonical Stage 1 and Stage 2 model training."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import stage1_train_intrinsic_pka as stage1_train
import stage2_train_network_context as stage2_train


def test_stage1_metrics_report_individual_and_tail_pka_errors():
    metrics = stage1_train._metrics(
        np.array([1.0, 5.0, 10.0, 15.0]),
        np.array([1.0, 3.0, 6.0, 9.0]),
        np.ones(4),
    )

    assert metrics["mae"] == pytest.approx(3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(14.0))
    assert metrics["max_abs_error"] == pytest.approx(6.0)
    assert metrics["count_abs_error_gt_2"] == 2
    assert metrics["count_abs_error_gt_3"] == 2
    assert metrics["count_abs_error_gt_5"] == 1


def test_stage1_scaffold_oof_holds_each_scaffold_together():
    features = pd.DataFrame({"signal": np.arange(8, dtype=float)})
    target = 2.0 * features["signal"].to_numpy() + 1.0
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])

    predicted, folds = stage1_train.scaffold_oof_predictions(
        LinearRegression(), features, target, np.ones(8), groups, folds=4
    )

    assert predicted == pytest.approx(target)
    assert all(len(set(folds[groups == group])) == 1 for group in set(groups))
    assert len(set(folds)) == 4


def test_stage1_scaffold_oof_requires_two_independent_groups():
    with pytest.raises(ValueError, match="At least two scaffold groups"):
        stage1_train.scaffold_oof_predictions(
            LinearRegression(),
            pd.DataFrame({"signal": [0.0, 1.0]}),
            np.array([1.0, 3.0]),
            np.ones(2),
            np.array(["same", "same"]),
            folds=5,
        )


def test_stage1_supplement_loader_skips_headerless_empty_sources(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    missing = tmp_path / "missing.csv"
    canonical = pd.DataFrame({
        "smiles": ["CC"],
        "sample_weight": [1.0],
    })

    loaded, report = stage1_train._load_supplemental_sites(
        f"{empty},{missing}", canonical
    )

    assert loaded.empty
    assert report["status"] == "available_but_empty"
    assert report["candidate_rows"] == 0


def test_stage1_label_interpolation_oof_is_complete_and_reproducible():
    features = pd.DataFrame({"signal": np.arange(8, dtype=float)})
    target = 3.0 * features["signal"].to_numpy() - 2.0
    labels = np.array(["amine", "acid"] * 4)

    first_prediction, first_folds = stage1_train.label_interpolation_oof_predictions(
        LinearRegression(), features, target, np.ones(8), labels, folds=4
    )
    second_prediction, second_folds = stage1_train.label_interpolation_oof_predictions(
        LinearRegression(), features, target, np.ones(8), labels, folds=4
    )

    assert first_prediction == pytest.approx(target)
    assert first_prediction == pytest.approx(second_prediction)
    assert first_folds.tolist() == second_folds.tolist()
    assert set(first_folds) == {0, 1, 2, 3}


def test_stage1_fold_support_distinguishes_supported_and_zero_shot_labels():
    frame = pd.DataFrame({
        "candidate_label": ["amine", "amine", "rare"],
        "scaffold_group": ["a", "b", "c"],
        "fold": [0, 1, 0],
    })

    annotated = stage1_train._add_fold_support(frame, "fold")

    assert annotated.loc[0, "validation_regime"] == "label_supported"
    assert annotated.loc[1, "validation_regime"] == "label_supported"
    assert annotated.loc[2, "validation_regime"] == "zero_shot_exact_label_absent"
    assert annotated.loc[2, "exact_label_training_rows_in_fold"] == 0


def test_stage1_error_calibration_preserves_family_and_exact_label_strata():
    frame = pd.DataFrame({
        "site_family": ["amine", "amine", "carboxyl"],
        "candidate_label": ["primary_amine", "primary_amine", "carboxylic_acid"],
        "abs_error": [0.5, 1.5, 2.0],
    })

    global_calibration, families, labels = stage1_train._error_calibration(frame)

    assert global_calibration["mae"] == pytest.approx(4.0 / 3.0)
    assert families["amine"]["rows"] == 2
    assert families["amine"]["mae"] == pytest.approx(1.0)
    assert labels["carboxylic_acid"]["p95_abs_error"] == pytest.approx(2.0)


def _synthetic_stage1_training_frame():
    signal = np.arange(8, dtype=float)
    return pd.DataFrame({
        "molecule_id": [f"mol_{idx}" for idx in range(8)],
        "site_id": ["site_1"] * 8,
        "measurement_method": ["experimental"] * 8,
        "sample_weight": np.ones(8),
        "scaffold_group": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "pka_value": 2.0 * signal + 1.0,
        "candidate_label": ["primary_amine", "carboxylic_acid"] * 4,
        "site_family": ["amine", "carboxyl"] * 4,
        "experimental_measurement_count": np.ones(8, dtype=int),
        "structure_key": [f"structure_{idx}" for idx in range(8)],
        "smiles": [f"synthetic_{idx}" for idx in range(8)],
        "functional_group_detector_schema_versions_json": [
            '["detector-test"]'
        ] * 8,
        "synthetic_signal": signal,
    })


def test_stage1_training_refits_exact_model_on_every_eligible_site(
    tmp_path, monkeypatch
):
    training = _synthetic_stage1_training_frame()
    network_path = tmp_path / "network.csv"
    network_path.write_text("synthetic-network\n")
    out_dir = tmp_path / "stage1"

    monkeypatch.setattr(
        stage1_train,
        "load_stage1_training_sites",
        lambda *args, **kwargs: (
            training.copy(),
            pd.DataFrame(columns=["quarantine_reason"]),
            {"eligible_training_sites": len(training)},
        ),
    )
    monkeypatch.setattr(
        stage1_train,
        "build_intrinsic_features",
        lambda frame, **kwargs: pd.DataFrame({
            "synthetic_signal": frame["synthetic_signal"].to_numpy(dtype=float)
        }),
    )
    monkeypatch.setattr(
        stage1_train,
        "_model_candidates",
        lambda: {"reference_hist_absolute_leaf15": LinearRegression()},
    )
    monkeypatch.setattr(
        stage1_train,
        "DEFAULT_REFERENCE_PANEL",
        str(tmp_path / "absent-reference-panel.csv"),
    )

    report, oof = stage1_train.train_stage1(
        str(network_path),
        str(out_dir),
        fp_bits=8,
        selection_splits=2,
        scaffold_folds=4,
        test_fraction=0.25,
    )

    with (out_dir / "stage1_intrinsic_model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    deployment_prediction = bundle["model"].predict(
        pd.DataFrame({"synthetic_signal": training["synthetic_signal"]})
    )

    assert deployment_prediction == pytest.approx(training["pka_value"])
    assert oof["pred_intrinsic_pka"].to_numpy() == pytest.approx(
        training["pka_value"].to_numpy()
    )
    assert report["rows"] == len(training)
    assert bundle["training_rows"] == len(training)
    assert bundle["deployment_fit_rows"] == len(training)
    assert bundle["trained_on_all_eligible_rows"] is True
    assert bundle["training_measurement_method"] == "experimental"
    assert bundle["functional_group_detector_schema_versions"] == ["detector-test"]
    assert report["scaffold"]["mae"] < 1e-10
    assert report["label_interpolation"]["mae"] < 1e-10


def test_stage1_training_rejects_nonexperimental_rows(tmp_path, monkeypatch):
    training = _synthetic_stage1_training_frame()
    training.loc[0, "measurement_method"] = "marvin"
    network_path = tmp_path / "network.csv"
    network_path.write_text("synthetic-network\n")
    monkeypatch.setattr(
        stage1_train,
        "load_stage1_training_sites",
        lambda *args, **kwargs: (training, pd.DataFrame(), {}),
    )

    with pytest.raises(ValueError, match="not experimental-only"):
        stage1_train.train_stage1(str(network_path), str(tmp_path / "out"))


def test_stage2_metrics_report_exact_residual_error_thresholds():
    metrics = stage2_train._metrics(
        np.array([0.0, 2.0, 5.0]),
        np.array([0.0, -1.0, 0.0]),
        np.ones(3),
    )

    assert metrics["mae"] == pytest.approx(8.0 / 3.0)
    assert metrics["max_abs_error"] == pytest.approx(5.0)
    assert metrics["count_abs_error_gt_2"] == 2
    assert metrics["count_abs_error_gt_3"] == 1
    assert metrics["count_abs_error_gt_5"] == 0


def _two_site_thermodynamic_payload():
    sites = [
        {"site_id": "a", "family": "guanidine_like"},
        {"site_id": "b", "family": "guanidine_like"},
    ]
    nodes = [
        {
            "node_id": "00",
            "protonated_site_count": 0,
            "site_forms": {"a": "base_form", "b": "base_form"},
        },
        {
            "node_id": "10",
            "protonated_site_count": 1,
            "site_forms": {"a": "acid_form", "b": "base_form"},
        },
        {
            "node_id": "01",
            "protonated_site_count": 1,
            "site_forms": {"a": "base_form", "b": "acid_form"},
        },
        {
            "node_id": "11",
            "protonated_site_count": 2,
            "site_forms": {"a": "acid_form", "b": "acid_form"},
        },
    ]
    return json.dumps(sites), json.dumps(nodes)


def _synthetic_stage2_site_table():
    sites_json, nodes_json = _two_site_thermodynamic_payload()
    rows = []
    for molecule_index, molecule in enumerate(("mol_1", "mol_2")):
        for site_id, baseline, residual, step in (
            ("a", 9.0, 0.4, 1),
            ("b", 4.0, 0.2, 2),
        ):
            rows.append({
                "molecule_key": molecule,
                "molecule_id": molecule,
                "site_id": site_id,
                "smiles": "NCCN" if molecule_index == 0 else "NCCCN",
                "site_label": "guanidine",
                "site_family": "guanidine_like",
                "stage1_intrinsic_pka": baseline,
                "stage1_network_macro_pka": baseline,
                "stage1_macro_step": step,
                "experimental_anchor_pka": baseline + residual,
                "stage2_delta_target": residual,
                "sample_weight": 1.0,
                "sample_weight_raw": 1.0,
                "experimental_measurement_count": 1.0,
                "experimental_measurement_range": 0.0,
                "network_sites_json": sites_json,
                "microstate_nodes_json": nodes_json,
                "synthetic_signal": residual,
            })
    frame = pd.DataFrame(rows)
    frame["full_site_position"] = np.arange(len(frame), dtype=int)
    return frame


def test_stage2_fold_fits_exact_individual_residual_then_closes_free_energy():
    site_table = _synthetic_stage2_site_table()
    features = pd.DataFrame({
        "synthetic_signal": site_table["synthetic_signal"].to_numpy(dtype=float)
    })

    metrics, evaluation, model = stage2_train._fit_evaluate(
        LinearRegression(),
        site_table,
        site_table,
        features,
        train_idx=np.array([0, 1]),
        test_idx=np.array([2, 3]),
        minimum_family_training_rows=1,
    )

    assert model.predict(features.iloc[[2, 3]]) == pytest.approx([0.4, 0.2])
    assert evaluation["stage2_predicted_macro_pka"].to_numpy() == pytest.approx(
        [9.4, 4.2]
    )
    assert evaluation["stage2_abs_error"].to_numpy() == pytest.approx([0.0, 0.0])
    assert np.isfinite(
        evaluation["stage2_supported_pairwise_free_energy_macro_pka"]
    ).all()
    assert metrics["stage2_corrected_raw"]["mae"] < 1e-12
    assert metrics["full_ladder_sites_predicted"] == 2


def test_stage2_split_manifest_preserves_site_level_membership():
    site_table = _synthetic_stage2_site_table()
    manifest = stage2_train._split_manifest(
        site_table, np.array([0, 1]), np.array([2, 3])
    )

    assert manifest.loc[:1, "split"].tolist() == ["train", "train"]
    assert manifest.loc[2:, "split"].tolist() == ["test", "test"]
    assert set(manifest.columns) == {"molecule_key", "site_id", "smiles", "split"}


def test_stage2_training_fits_deployment_residual_model_on_all_eligible_sites(
    tmp_path, monkeypatch
):
    full_site_table = _synthetic_stage2_site_table()
    network_path = tmp_path / "network.csv"
    pd.DataFrame([{
        "stage1_training_measurement_method": "experimental",
        "stage1_trained_on_all_eligible_rows": True,
        "marvin_values_used": False,
        "epik_values_used": False,
        "stage1_model_sha256": "stage1-hash",
        "stage1_schema_version": "stage1-test",
        "stage1_target_definition": "intrinsic-test",
        "dataset_schema_version": "network-test",
        "functional_group_detector_schema_versions_json": '["detector-test"]',
    }]).to_csv(network_path, index=False)
    out_dir = tmp_path / "stage2"

    monkeypatch.setattr(
        stage2_train,
        "build_stage2_site_table",
        lambda *args, **kwargs: full_site_table.copy(),
    )
    monkeypatch.setattr(
        stage2_train,
        "prepare_stage2_training_sites",
        lambda frame, **kwargs: (
            frame.copy(),
            pd.DataFrame(columns=["quarantine_reason"]),
            {"accepted_training_sites": len(frame)},
        ),
    )
    monkeypatch.setattr(
        stage2_train,
        "build_stage2_feature_matrix",
        lambda frame, **kwargs: pd.DataFrame({
            "synthetic_signal": frame["synthetic_signal"].to_numpy(dtype=float)
        }),
    )
    monkeypatch.setattr(
        stage2_train,
        "build_stage2_weak_training_sites",
        lambda *args, **kwargs: (
            full_site_table.iloc[0:0].copy(),
            pd.DataFrame(columns=["blocked_reason"]),
        ),
    )
    split = (np.array([0, 1]), np.array([2, 3]))
    monkeypatch.setattr(stage2_train, "random_molecule_split", lambda *a, **k: split)
    monkeypatch.setattr(stage2_train, "scaffold_molecule_split", lambda *a, **k: split)
    selection_detail = pd.DataFrame([{
        "candidate": "linear_test",
        "selection_fold": 0,
        "stage2_supported_mae": 0.0,
        "stage2_supported_weighted_mae": 0.0,
        "mae_improvement": 0.3,
    }])
    monkeypatch.setattr(
        stage2_train,
        "_select_model",
        lambda *args, **kwargs: (
            "linear_test", LinearRegression(), selection_detail.copy()
        ),
    )

    report, evaluation = stage2_train.train_stage2(
        str(network_path), str(out_dir), fp_bits=8, selection_splits=1
    )

    with (out_dir / "stage2_network_context_model.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    deployment_prediction = bundle["model"].predict(pd.DataFrame({
        "synthetic_signal": full_site_table["synthetic_signal"]
    }))

    assert deployment_prediction == pytest.approx(
        full_site_table["stage2_delta_target"]
    )
    assert bundle["training_rows"] == len(full_site_table)
    assert bundle["deployment_fit_rows"] == len(full_site_table)
    assert bundle["trained_on_all_eligible_rows"] is True
    assert bundle["supported_families"] == ["guanidine_like"]
    assert bundle["stage1_model_sha256"] == "stage1-hash"
    assert report["scaffold_split"]["stage2_corrected_raw"]["mae"] < 1e-12
    assert evaluation["stage2_abs_error"].to_numpy() == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    "column, value, message",
    [
        (
            "stage1_training_measurement_method",
            "marvin",
            "does not use an experimental-only Stage 1 model",
        ),
        (
            "stage1_trained_on_all_eligible_rows",
            False,
            "Stage 1 was not fit on all eligible rows",
        ),
        ("epik_values_used", True, "prohibited Marvin or Epik labels"),
    ],
)
def test_stage2_training_rejects_untrusted_upstream_data(
    tmp_path, column, value, message
):
    row = {
        "stage1_training_measurement_method": "experimental",
        "stage1_trained_on_all_eligible_rows": True,
        "marvin_values_used": False,
        "epik_values_used": False,
    }
    row[column] = value
    path = tmp_path / "network.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        stage2_train.train_stage2(str(path), str(tmp_path / "out"))

