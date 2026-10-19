"""End-to-end contract tests for the deployable Stage 1 feature/model bundle."""

from __future__ import annotations

import copy
import pickle
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import pkate
import train_intrinsic_single_group_model as intrinsic


def _synthetic_assignments() -> pd.DataFrame:
    """Return small, independent structures suitable for both split policies."""
    smiles = [
        "CN",
        "CCN",
        "CCCN",
        "CCCCN",
        "CC(C)N",
        "NCCO",
        "NCCCO",
        "NCCCl",
        "NCCBr",
        "NC1CCCCC1",
    ]
    return pd.DataFrame(
        {
            "source_file": ["synthetic_stage1.sdf"] * len(smiles),
            "record_index": range(len(smiles)),
            "smiles": smiles,
            "pka_value": np.linspace(8.1, 10.0, len(smiles)),
            "final_group_label": ["primary_amine"] * len(smiles),
            "assignment_status": ["single_group"] * len(smiles),
            "single_group_strict": [True] * len(smiles),
            "pka_type_canonical": ["basic"] * len(smiles),
            "pka_source_method": ["experimental"] * len(smiles),
            "distribution_ok": [True] * len(smiles),
            "site_family": ["amine_aminium"] * len(smiles),
        }
    )


@pytest.fixture(scope="module")
def trained_roundtrip(tmp_path_factory):
    """Train through the real legacy entry point and capture its pre-pickle bundle."""
    work_dir = tmp_path_factory.mktemp("intrinsic-roundtrip")
    assignments_path = work_dir / "assignments.csv"
    output_dir = work_dir / "model"
    _synthetic_assignments().to_csv(assignments_path, index=False)

    args = Namespace(
        assignments=str(assignments_path),
        raw_dir=str(work_dir / "unused-raw"),
        out_dir=str(output_dir),
        fp_bits=32,
        fp_radius=2,
        test_frac=0.2,
        measurement_method="experimental",
    )
    real_pickle_dump = pickle.dump
    captured = {}

    def capture_bundle(value, handle, *dump_args, **dump_kwargs):
        captured["bundle"] = value
        return real_pickle_dump(value, handle, *dump_args, **dump_kwargs)

    with (
        patch.object(intrinsic, "parse_args", return_value=args),
        patch.object(
            intrinsic,
            "ensure_curated_training_data",
            return_value=(str(assignments_path), str(work_dir / "unused-summary.csv")),
        ),
        patch.object(intrinsic.pickle, "dump", side_effect=capture_bundle),
    ):
        intrinsic.main()

    rows = intrinsic.load_single_group_rows(
        str(assignments_path), measurement_method="experimental"
    )
    features = intrinsic.build_intrinsic_features(rows, nbits=32, radius=2)
    model_path = output_dir / "stage1_intrinsic_model.pkl"
    return {
        "rows": rows,
        "features": features,
        "output_dir": output_dir,
        "model_path": model_path,
        "before_bundle": captured["bundle"],
    }


def _write_bundle(path: Path, bundle: dict) -> Path:
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return path


def test_complete_feature_assembly_includes_geometry_and_site_context(
    trained_roundtrip,
):
    features = trained_roundtrip["features"]

    expected_columns = {
        "desc_mol_wt",
        "site_desc_mol_wt",
        "mol_geom_embed_success",
        "mol_geom_radius_of_gyration",
        "site_geom_embed_success",
        "site_geom_source_centroid_to_mol_centroid",
        "site_match_found",
        "fg_global_count__primary_amine",
        "fg_local_count__primary_amine",
        "stage1_reference_pka",
        "stage1_reference_uncertainty",
        "stage1_reference_prior_present",
        "group_primary_amine",
        "pka_type_basic",
        "fp_0",
        "site_fp_0",
    }
    assert expected_columns <= set(features.columns)
    assert not features.columns.duplicated().any()
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    assert features["site_match_found"].eq(1.0).all()
    assert features["mol_geom_embed_success"].eq(1.0).all()
    assert features["site_geom_embed_success"].eq(1.0).all()
    assert (features["mol_geom_radius_of_gyration"] > 0.0).all()


def test_tiny_training_run_saves_a_complete_loadable_bundle(trained_roundtrip):
    output_dir = trained_roundtrip["output_dir"]
    model_path = trained_roundtrip["model_path"]

    assert model_path.is_file()
    assert (output_dir / "metrics.json").is_file()
    assert (output_dir / "random_eval_intrinsic.csv").is_file()
    assert (output_dir / "scaffold_eval_intrinsic.csv").is_file()
    assert (output_dir / "random_split_manifest.csv").is_file()
    assert (output_dir / "scaffold_split_manifest.csv").is_file()

    loaded = pkate.load_stage1_bundle(str(model_path))
    assert loaded["stage1_feature_schema_version"] == (
        intrinsic.STAGE1_FEATURE_SCHEMA_VERSION
    )
    assert loaded["training_measurement_method"] == "experimental"
    assert loaded["training_rows"] == len(trained_roundtrip["rows"])
    assert loaded["feature_columns"] == list(trained_roundtrip["features"].columns)


def test_pkate_reload_preserves_predictions_exactly(trained_roundtrip):
    before = trained_roundtrip["before_bundle"]
    loaded = pkate.load_stage1_bundle(str(trained_roundtrip["model_path"]))
    features = trained_roundtrip["features"]

    before_predictions = before["model"].predict(features)
    after_predictions = loaded["model"].predict(features)
    np.testing.assert_array_equal(after_predictions, before_predictions)


def test_serialized_model_predicts_through_pkate_stage1_entry_point(
    trained_roundtrip,
):
    tasks = [
        {
            "task_id": "query:amine",
            "molecule_id": "query",
            "smiles": "CCCN",
            "candidate_label": "primary_amine",
            "pka_type_canonical": "basic",
            "atom_index_raw": 3,
        }
    ]
    loaded = pkate.load_stage1_bundle(str(trained_roundtrip["model_path"]))
    expected_features = intrinsic.build_intrinsic_features(
        pd.DataFrame(tasks), nbits=32, radius=2, label_col="candidate_label"
    ).reindex(columns=loaded["feature_columns"], fill_value=0.0)
    expected = float(loaded["model"].predict(expected_features)[0])

    actual = pkate.predict_stage1_tasks(
        tasks, str(trained_roundtrip["model_path"]), workers=1
    )

    assert actual == {"query:amine": pytest.approx(expected)}


def test_loader_rejects_feature_columns_that_differ_from_fitted_model(
    trained_roundtrip, tmp_path
):
    bundle = copy.copy(trained_roundtrip["before_bundle"])
    bundle["feature_columns"] = list(bundle["feature_columns"][:-1])
    path = _write_bundle(tmp_path / "bad-columns.pkl", bundle)

    with pytest.raises(ValueError, match="do not match the fitted model"):
        pkate.load_stage1_bundle(str(path))


def test_loader_rejects_incompatible_feature_schema_version(
    trained_roundtrip, tmp_path
):
    bundle = copy.copy(trained_roundtrip["before_bundle"])
    bundle["stage1_feature_schema_version"] = "999.0.0"
    path = _write_bundle(tmp_path / "bad-version.pkl", bundle)

    with pytest.raises(ValueError, match="feature schema version mismatch"):
        pkate.load_stage1_bundle(str(path))
