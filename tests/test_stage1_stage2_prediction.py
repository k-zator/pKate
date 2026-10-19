"""Prediction-contract tests for the Stage 1 -> Stage 2 pKa path.

The fixtures deliberately use tiny deterministic regressors.  These tests are
about the scientific plumbing around a fitted model: every site must retain its
identity, receive exactly one finite intrinsic prediction, and receive a Stage
2 correction only when its family and network are eligible.
"""

from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import pytest

import stage1_network_inference
from stage1_network_inference import (
    load_stage1_bundle,
    predict_stage1_tasks,
    stage1_applicability,
)
from stage2_apply_network_context import apply_stage2
from stage2_free_energy_coupling import (
    STAGE2_FREE_ENERGY_METHOD,
    STAGE2_FREE_ENERGY_SCHEMA_VERSION,
)
from stage2_network_context import (
    STAGE2_TARGET_DEFINITION,
    STAGE2_TRAINING_SCHEMA_VERSION,
)


class FixedRegressor:
    """Minimal pickle-safe regressor returning one exact residual or pKa."""

    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.value, dtype=float)


class WrongLengthRegressor(FixedRegressor):
    """Broken model used to prove silent site/prediction truncation is rejected."""

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(max(0, len(features) - 1), self.value, dtype=float)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_stage1_bundle(tmp_path, *, value=7.25, model=None, **updates):
    bundle = {
        "model": model or FixedRegressor(value),
        "feature_columns": ["stage1_reference_pka"],
        "fp_bits": 32,
        "fp_radius": 2,
        "training_measurement_method": "experimental",
        "stage1_schema_version": "stage1-test",
        "trained_on_all_eligible_rows": True,
        "deployment_fit_rows": 12,
        "training_rows": 12,
    }
    bundle.update(updates)
    path = tmp_path / "stage1.pkl"
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return path, bundle


def _stage1_tasks():
    return [
        {
            "task_id": "mol:amine",
            "molecule_id": "mol",
            "smiles": "CN",
            "candidate_label": "primary_amine",
            "pka_type_canonical": "basic",
            "atom_index_raw": 1,
        },
        {
            "task_id": "mol:carboxyl",
            "molecule_id": "mol",
            "smiles": "CC(=O)O",
            "candidate_label": "carboxylic_acid",
            "pka_type_canonical": "acidic",
            "atom_index_raw": 3,
        },
    ]


def test_stage1_predicts_one_exact_finite_pka_per_site(tmp_path):
    model_path, _ = _write_stage1_bundle(tmp_path, value=7.25)

    predictions = predict_stage1_tasks(_stage1_tasks(), str(model_path))

    assert predictions == {
        "mol:amine": pytest.approx(7.25),
        "mol:carboxyl": pytest.approx(7.25),
    }


def test_stage1_parallel_chunking_preserves_site_identity(tmp_path, monkeypatch):
    model_path, _ = _write_stage1_bundle(tmp_path, value=6.75)

    class InlineExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, payloads):
            return [function(payload) for payload in payloads]

    monkeypatch.setattr(stage1_network_inference, "ProcessPoolExecutor", InlineExecutor)
    tasks = _stage1_tasks() + [{
        **_stage1_tasks()[0],
        "task_id": "other:amine",
        "molecule_id": "other",
    }]

    predictions = predict_stage1_tasks(tasks, str(model_path), workers=2)

    assert set(predictions) == {task["task_id"] for task in tasks}
    assert list(predictions.values()) == pytest.approx([6.75, 6.75, 6.75])


def test_stage1_permission_failure_falls_back_to_serial_prediction(
    tmp_path, monkeypatch
):
    model_path, _ = _write_stage1_bundle(tmp_path, value=8.1)

    class UnavailableExecutor:
        def __init__(self, max_workers):
            pass

        def __enter__(self):
            raise PermissionError("process creation disabled")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        stage1_network_inference, "ProcessPoolExecutor", UnavailableExecutor
    )

    predictions = predict_stage1_tasks(
        _stage1_tasks(), str(model_path), workers=2
    )

    assert list(predictions.values()) == pytest.approx([8.1, 8.1])


@pytest.mark.parametrize(
    "tasks, message",
    [
        ([{**_stage1_tasks()[0], "task_id": ""}], "non-empty task_id"),
        (
            [_stage1_tasks()[0], {**_stage1_tasks()[1], "task_id": "mol:amine"}],
            "Duplicate Stage 1 task_id",
        ),
    ],
)
def test_stage1_rejects_tasks_that_cannot_be_mapped_one_to_one(
    tmp_path, tasks, message
):
    model_path, _ = _write_stage1_bundle(tmp_path)
    with pytest.raises(ValueError, match=message):
        predict_stage1_tasks(tasks, str(model_path))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_stage1_rejects_nonfinite_individual_pka_predictions(tmp_path, value):
    model_path, _ = _write_stage1_bundle(tmp_path, value=value)
    with pytest.raises(ValueError, match="non-finite pKa"):
        predict_stage1_tasks(_stage1_tasks(), str(model_path))


def test_stage1_rejects_model_output_with_wrong_number_of_sites(tmp_path):
    model_path, _ = _write_stage1_bundle(
        tmp_path, model=WrongLengthRegressor(7.0)
    )
    with pytest.raises(RuntimeError, match="different number of pKa predictions"):
        predict_stage1_tasks(_stage1_tasks(), str(model_path))


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"training_measurement_method": "marvin"}, "not certified experimental-only"),
        ({"trained_on_all_eligible_rows": False}, "not fit on all eligible rows"),
        ({"deployment_fit_rows": 11}, "provenance is inconsistent"),
    ],
)
def test_stage1_rejects_untrusted_or_partial_deployment_bundles(
    tmp_path, updates, message
):
    model_path, _ = _write_stage1_bundle(tmp_path, **updates)
    with pytest.raises(ValueError, match=message):
        load_stage1_bundle(str(model_path))


def test_stage1_applicability_covers_high_similarity_and_sparse_extrapolation():
    high = stage1_applicability(
        "CN",
        "primary_amine",
        {
            "fp_bits": 32,
            "fp_radius": 2,
            "training_site_label_counts": {"primary_amine": 5},
            "training_smiles_by_label": {"primary_amine": ["CN"] * 5},
        },
    )
    sparse = stage1_applicability(
        "CN",
        "primary_amine",
        {
            "fp_bits": 32,
            "fp_radius": 2,
            "training_site_label_counts": {"primary_amine": 1},
            "training_smiles_by_label": {"primary_amine": ["CN"]},
        },
    )

    assert high["stage1_applicability_domain"] == "interpolation_high_similarity"
    assert high["stage1_nearest_same_label_tanimoto"] == pytest.approx(1.0)
    assert sparse["stage1_applicability_domain"] == (
        "extrapolation_low_similarity_or_sparse_label"
    )


def _two_site_network_row(*, incomplete=False):
    sites = [
        {
            "site_id": "amine",
            "label": "primary_amine",
            "family": "amine",
            "atom_maps": [1],
            "center_maps": [1],
            "stage1_intrinsic_pka": 9.5,
            "stage1_predicted_pka_ci_low": 8.5,
            "stage1_predicted_pka_ci_high": 10.5,
            "rank_associated_macro_step": 1,
            "rank_associated_macro_pka": 9.3,
            "stage1_group_feature_present": True,
            "experimental_anchor_pka": None,
            "experimental_measurements": [],
        },
        {
            "site_id": "carboxyl",
            "label": "carboxylic_acid",
            "family": "carboxyl",
            "atom_maps": [3, 4, 5],
            "center_maps": [5],
            "stage1_intrinsic_pka": 4.2,
            "stage1_predicted_pka_ci_low": 3.7,
            "stage1_predicted_pka_ci_high": 4.7,
            "rank_associated_macro_step": 2,
            "rank_associated_macro_pka": 4.4,
            "stage1_group_feature_present": True,
            "experimental_anchor_pka": None,
            "experimental_measurements": [],
        },
    ]
    nodes = [
        {
            "node_id": "00",
            "protonated_site_count": 0,
            "site_forms": {"amine": "base_form", "carboxyl": "base_form"},
        },
        {
            "node_id": "10",
            "protonated_site_count": 1,
            "site_forms": {"amine": "acid_form", "carboxyl": "base_form"},
        },
        {
            "node_id": "01",
            "protonated_site_count": 1,
            "site_forms": {"amine": "base_form", "carboxyl": "acid_form"},
        },
        {
            "node_id": "11",
            "protonated_site_count": 2,
            "site_forms": {"amine": "acid_form", "carboxyl": "acid_form"},
        },
    ]
    edges = [
        {
            "edge_id": "amine_0",
            "site_id": "amine",
            "acid_node_id": "10",
            "base_node_id": "00",
        },
        {
            "edge_id": "amine_1",
            "site_id": "amine",
            "acid_node_id": "11",
            "base_node_id": "01",
        },
        {
            "edge_id": "carboxyl_0",
            "site_id": "carboxyl",
            "acid_node_id": "01",
            "base_node_id": "00",
        },
        {
            "edge_id": "carboxyl_1",
            "site_id": "carboxyl",
            "acid_node_id": "11",
            "base_node_id": "10",
        },
    ]
    macro_steps = [
        {
            "protonation_step": 1,
            "predicted_macro_pka": 9.3,
            "rank_associated_site_id": "amine",
        },
        {
            "protonation_step": 2,
            "predicted_macro_pka": 4.4,
            "rank_associated_site_id": "carboxyl",
        },
    ]
    return {
        "dataset_schema_version": "network-test",
        "functional_group_detector_schema_versions_json": json.dumps(
            ["detector-test"]
        ),
        "stage1_schema_version": "stage1-test",
        "stage1_model_sha256": "stage1-hash",
        "stage1_trained_on_all_eligible_rows": True,
        "marvin_values_used": False,
        "epik_values_used": False,
        "molecule_id": "glycine-test",
        "representative_input_smiles": "NCC(=O)O",
        "representative_atom_mapped_smiles": (
            "[NH2:1][CH2:2][C:3](=[O:4])[OH:5]"
        ),
        "detected_site_count": 2,
        "protonation_state_count": 4,
        "protonation_edge_count": 4,
        "protonation_state_enumeration_truncated": False,
        "incomplete_site_ids_json": json.dumps(
            ["carboxyl"] if incomplete else []
        ),
        "sites_json": json.dumps(sites),
        "microstate_nodes_json": json.dumps(nodes),
        "microstate_edges_json": json.dumps(edges),
        "independent_site_macro_pka_steps_json": json.dumps(macro_steps),
        "baseline_macro_pka_steps_json": json.dumps(macro_steps),
    }


def _write_stage2_inputs(
    tmp_path,
    *,
    residual=0.7,
    supported_families=("amine",),
    priority_families=(),
    incomplete=False,
    bundle_updates=None,
    row_updates=None,
):
    row = _two_site_network_row(incomplete=incomplete)
    row.update(row_updates or {})
    network_path = tmp_path / "network.csv"
    pd.DataFrame([row]).to_csv(network_path, index=False)
    bundle = {
        "model": FixedRegressor(residual),
        "feature_columns": ["stage1_network_macro_pka"],
        "fp_bits": 32,
        "fp_radius": 2,
        "target_definition": STAGE2_TARGET_DEFINITION,
        "training_measurement_method": "experimental",
        "stage1_model_sha256": "stage1-hash",
        "stage2_schema_version": STAGE2_TRAINING_SCHEMA_VERSION,
        "network_dataset_sha256": _sha256(network_path),
        "network_dataset_schema_versions": ["network-test"],
        "functional_group_detector_schema_versions": ["detector-test"],
        "stage1_schema_versions": ["stage1-test"],
        "trained_on_all_eligible_rows": True,
        "deployment_fit_rows": 100,
        "training_rows": 100,
        "stage2_free_energy_schema_version": STAGE2_FREE_ENERGY_SCHEMA_VERSION,
        "stage2_free_energy_method": STAGE2_FREE_ENERGY_METHOD,
        "supported_families": list(supported_families),
        "priority_supported_families": list(priority_families),
        "family_training_counts": {"amine": 100, "carboxyl": 100},
        "minimum_family_training_rows": 50,
        "scaffold_validation_mae": 0.3,
        "scaffold_abs_error_p90": 0.35,
        "scaffold_abs_error_p95": 0.4,
        "selected_model": "fixed_test_regressor",
    }
    bundle.update(bundle_updates or {})
    model_path = tmp_path / "stage2.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return network_path, model_path


def _run_stage2(tmp_path, **kwargs):
    network_path, model_path = _write_stage2_inputs(tmp_path, **kwargs)
    output_path = tmp_path / "stage2.csv"
    report_path = tmp_path / "stage2-report.json"
    result, report = apply_stage2(
        str(network_path),
        str(model_path),
        str(output_path),
        report_path=str(report_path),
    )
    return result, report, output_path, report_path


def test_stage2_applies_exact_residual_only_to_supported_site_family(tmp_path):
    result, report, output_path, report_path = _run_stage2(
        tmp_path, residual=0.7, supported_families=("amine",)
    )
    sites = {
        site["site_id"]: site
        for site in json.loads(result.iloc[0]["sites_json"])
    }
    steps = {
        step["rank_associated_site_id"]: step
        for step in json.loads(result.iloc[0]["stage2_macro_pka_steps_json"])
    }

    assert sites["amine"]["stage2_applied"] is True
    assert sites["amine"]["stage2_predicted_delta"] == pytest.approx(0.7)
    assert sites["amine"]["stage2_predicted_macro_pka"] == pytest.approx(10.0)
    assert steps["amine"]["stage2_predicted_macro_pka"] == pytest.approx(10.0)

    assert sites["carboxyl"]["stage2_applied"] is False
    assert sites["carboxyl"]["stage2_raw_model_delta"] == pytest.approx(0.7)
    assert sites["carboxyl"]["stage2_predicted_delta"] == pytest.approx(0.0)
    assert sites["carboxyl"]["stage2_predicted_macro_pka"] == pytest.approx(4.4)
    assert sites["carboxyl"]["stage2_prediction_confidence"] == (
        "fallback_stage1_insufficient_family_training_support"
    )
    assert report["complex_site_predictions"] == 2
    assert result.iloc[0]["stage2_applied_site_count"] == 1
    assert output_path.is_file()
    assert report_path.is_file()
    assert json.loads(report_path.read_text())["output_sha256"] == _sha256(output_path)


def test_stage2_combines_stage1_and_context_uncertainty_in_quadrature(tmp_path):
    result, _, _, _ = _run_stage2(
        tmp_path, residual=0.7, supported_families=("amine",)
    )
    sites = {
        site["site_id"]: site
        for site in json.loads(result.iloc[0]["sites_json"])
    }
    expected_half_width = np.sqrt(1.0**2 + 0.4**2)

    assert sites["amine"]["stage2_predicted_macro_pka_ci_low"] == pytest.approx(
        10.0 - expected_half_width
    )
    assert sites["amine"]["stage2_predicted_macro_pka_ci_high"] == pytest.approx(
        10.0 + expected_half_width
    )
    assert sites["carboxyl"]["stage2_predicted_macro_pka_ci_low"] == pytest.approx(
        3.9
    )
    assert sites["carboxyl"]["stage2_predicted_macro_pka_ci_high"] == pytest.approx(
        4.9
    )


def test_stage2_projects_context_correction_without_moving_fixed_site(tmp_path):
    result, report, _, _ = _run_stage2(
        tmp_path,
        residual=6.0,
        supported_families=("carboxyl",),
    )
    sites = {
        site["site_id"]: site
        for site in json.loads(result.iloc[0]["sites_json"])
    }

    assert sites["amine"]["stage2_predicted_macro_pka"] == pytest.approx(9.3)
    assert sites["amine"]["stage2_monotonic_projection_applied"] is False
    assert sites["carboxyl"]["stage2_raw_predicted_macro_pka"] == pytest.approx(10.4)
    assert sites["carboxyl"]["stage2_predicted_macro_pka"] == pytest.approx(9.3)
    assert sites["carboxyl"]["stage2_monotonic_projection_applied"] is True
    assert report["molecules_with_macro_order_violation"] == 1
    assert report["sites_adjusted_by_monotonic_projection"] == 1


def test_stage2_marks_priority_family_with_limited_empirical_support(tmp_path):
    result, _, _, _ = _run_stage2(
        tmp_path,
        supported_families=("amine",),
        priority_families=("amine",),
        bundle_updates={"family_training_counts": {"amine": 12, "carboxyl": 100}},
    )
    sites = {
        site["site_id"]: site
        for site in json.loads(result.iloc[0]["sites_json"])
    }

    assert sites["amine"]["stage2_prediction_confidence"] == (
        "model_estimate_priority_family_limited_support"
    )
    assert sites["amine"]["stage2_family_training_count"] == 12


def test_incomplete_network_never_exposes_numeric_stage2_pka(tmp_path):
    result, report, _, _ = _run_stage2(tmp_path, incomplete=True)
    sites = json.loads(result.iloc[0]["sites_json"])
    steps = json.loads(result.iloc[0]["stage2_macro_pka_steps_json"])
    edges = json.loads(result.iloc[0]["microstate_edges_json"])

    assert all(site["stage2_predicted_macro_pka"] is None for site in sites)
    assert all(
        site["stage2_prediction_confidence"]
        == "unavailable_incomplete_or_truncated_network"
        for site in sites
    )
    assert all(step["stage2_predicted_macro_pka"] is None for step in steps)
    assert all(edge["stage2_contextual_edge_pka"] is None for edge in edges)
    assert report["complex_site_predictions"] == 0
    assert report["mean_free_energy_macro_reconstruction_mae"] is None


@pytest.mark.parametrize(
    "bundle_updates, message",
    [
        ({"training_measurement_method": "epik"}, "not experimental-only"),
        ({"stage1_model_sha256": "another-stage1"}, "different Stage 1 model"),
        ({"trained_on_all_eligible_rows": False}, "not fit on all eligible rows"),
        ({"deployment_fit_rows": 99}, "row count does not match"),
        ({"stage2_schema_version": "old"}, "bundle schema is stale"),
    ],
)
def test_stage2_rejects_incompatible_or_untrusted_model_provenance(
    tmp_path, bundle_updates, message
):
    network_path, model_path = _write_stage2_inputs(
        tmp_path, bundle_updates=bundle_updates
    )
    with pytest.raises(ValueError, match=message):
        apply_stage2(
            str(network_path),
            str(model_path),
            str(tmp_path / "out.csv"),
            report_path=str(tmp_path / "report.json"),
        )


def test_stage2_requires_exact_snapshot_unless_compatibility_is_explicit(tmp_path):
    network_path, model_path = _write_stage2_inputs(
        tmp_path, bundle_updates={"network_dataset_sha256": "different-snapshot"}
    )
    with pytest.raises(ValueError, match="different network snapshot"):
        apply_stage2(
            str(network_path),
            str(model_path),
            str(tmp_path / "strict.csv"),
            report_path=str(tmp_path / "strict-report.json"),
        )

    compatible, _ = apply_stage2(
        str(network_path),
        str(model_path),
        str(tmp_path / "compatible.csv"),
        allow_compatible_network=True,
        report_path=str(tmp_path / "compatible-report.json"),
    )
    assert len(compatible) == 1

