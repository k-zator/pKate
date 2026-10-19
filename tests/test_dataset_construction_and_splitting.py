"""Consistency tests for candidate construction, splits, and data refreshes."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import functional_group_pka_analysis
import training_data_resolver
from ml_dataset_builder import (
    DatasetBuildConfig,
    _build_candidate_rows,
    _build_family_priors,
    _candidate_transition_labels,
    _deduplicate_measurement_rows,
    build_candidate_dataset,
)
from ml_splits import (
    _murcko_scaffold,
    random_molecule_split,
    scaffold_molecule_split,
    split_manifest,
)


def _assignment_row(**updates):
    row = {
        "source_file": "source.sdf",
        "record_index": 1,
        "smiles": "c1ncc[nH]1",
        "pka_value": 7.0,
        "pka_type_canonical": "basic",
        "resolved_groups": "imidazole|ketone",
        "resolved_group_instances": "imidazole|ketone",
        "resolved_group_count": 2,
        "assignment_status": "multi_group",
        "selected_group_label": "imidazole_basicity",
        "final_group_label": "imidazole_basicity",
        "final_group_family": "imidazole_basicity",
        "pair_member_form": "base_form",
        "group_mode": "pair_type",
        "distribution_ok": True,
        "molecule_formal_charge": 0,
        "neutral_input_risk": False,
    }
    row.update(updates)
    return row


def _summary_rows():
    return pd.DataFrame([
        {
            "final_group_label": "imidazole_basicity",
            "mean": 6.9,
            "median": 7.0,
            "iqr": 0.8,
            "count": 20,
        },
        {
            "final_group_label": "primary_amine",
            "mean": 10.2,
            "median": 10.3,
            "iqr": 1.0,
            "count": 10,
        },
    ])


def test_transition_specific_azole_label_is_retained_as_candidate():
    labels = _candidate_transition_labels(
        "imidazole", "imidazole_basicity", "basic"
    )
    assert labels == ["imidazole_basicity"]


def test_untyped_azole_exposes_both_serial_transition_candidates():
    labels = _candidate_transition_labels(
        "imidazole", "imidazole_basicity", None
    )
    assert labels == ["imidazole_basicity", "imidazole_acidity"]


def test_unresolved_ring_level_selection_remains_ring_level_candidate():
    assert _candidate_transition_labels("imidazole", "imidazole", None) == [
        "imidazole"
    ]


def test_candidate_rows_assign_exactly_one_true_transition_per_measurement():
    frame = pd.DataFrame([_assignment_row()])
    candidates = _build_candidate_rows(frame)

    assert set(candidates["candidate_label"]) == {"imidazole_basicity", "ketone"}
    assert candidates["is_true_site"].sum() == 1
    selected = candidates[candidates["is_true_site"] == 1].iloc[0]
    assert selected["candidate_label"] == "imidazole_basicity"
    assert selected["candidate_instance_id"] == "imidazole_basicity#0"


def test_duplicate_site_labels_still_create_only_one_positive_instance():
    frame = pd.DataFrame([_assignment_row(
        smiles="NCCN",
        selected_group_label="primary_amine",
        final_group_label="primary_amine",
        final_group_family="amine",
        resolved_groups="primary_amine",
        resolved_group_instances="primary_amine|primary_amine",
        resolved_group_count=1,
    )])
    candidates = _build_candidate_rows(frame)

    assert len(candidates) == 2
    assert candidates["is_true_site"].tolist() == [1, 0]
    assert candidates["candidate_instance_id"].is_unique


def test_candidate_construction_rejects_missing_selected_site():
    frame = pd.DataFrame([_assignment_row(
        selected_group_label="primary_amine",
        final_group_label="primary_amine",
    )])
    with pytest.raises(ValueError, match="absent from the resolved candidate set"):
        _build_candidate_rows(frame)


def test_duplicate_measurement_prefers_typed_site_assignment():
    untyped = _assignment_row(pka_type_canonical=None)
    typed = _assignment_row(
        pka_type_canonical="acidic",
        selected_group_label="imidazole_acidity",
        final_group_label="imidazole_acidity",
    )

    retained = _deduplicate_measurement_rows(pd.DataFrame([untyped, typed]))

    assert len(retained) == 1
    assert retained.iloc[0]["pka_type_canonical"] == "acidic"
    assert retained.iloc[0]["selected_group_label"] == "imidazole_acidity"


def test_duplicate_measurement_rejects_equally_supported_conflicting_sites():
    basic = _assignment_row()
    acidic = _assignment_row(
        pka_type_canonical="acidic",
        selected_group_label="imidazole_acidity",
        final_group_label="imidazole_acidity",
    )

    with pytest.raises(ValueError, match="Contradictory equally supported"):
        _deduplicate_measurement_rows(pd.DataFrame([basic, acidic]))


def test_family_priors_use_count_weighted_mean_and_dominant_label_median(tmp_path):
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame([
        {
            "final_group_label": "primary_amine",
            "mean": 10.0,
            "median": 10.1,
            "iqr": 1.0,
            "count": 30,
        },
        {
            "final_group_label": "primary_ammonium",
            "mean": 9.0,
            "median": 9.2,
            "iqr": 1.2,
            "count": 10,
        },
    ]).to_csv(summary_path, index=False)

    priors = _build_family_priors(str(summary_path)).set_index("candidate_family")

    assert priors.loc["amine", "family_prior_mean_pka"] == pytest.approx(9.75)
    assert priors.loc["amine", "family_prior_median_pka"] == pytest.approx(10.1)
    assert priors.loc["amine", "family_prior_count"] == pytest.approx(40.0)


def test_complete_candidate_dataset_is_deterministic_and_preserves_targets(tmp_path):
    assignments_path = tmp_path / "assignments.csv"
    summary_path = tmp_path / "summary.csv"
    assignments = pd.DataFrame([
        _assignment_row(),
        _assignment_row(
            record_index=2,
            smiles="CN",
            pka_value=10.6,
            selected_group_label="primary_amine",
            final_group_label="primary_amine",
            final_group_family="amine",
            resolved_groups="primary_amine",
            resolved_group_instances="primary_amine",
            resolved_group_count=1,
            assignment_status="single_group",
        ),
    ])
    assignments.to_csv(assignments_path, index=False)
    _summary_rows().to_csv(summary_path, index=False)
    config = DatasetBuildConfig(
        assignments_path=str(assignments_path),
        summary_path=str(summary_path),
    )

    first = build_candidate_dataset(config)
    second = build_candidate_dataset(config)

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("molecule_key")["is_true_site"].sum().eq(1).all()
    assert set(first.loc[first["is_true_site"] == 1, "pka_value"]) == {7.0, 10.6}
    assert first["candidate_prior_mean_pka"].notna().all()
    assert first["candidate_prior_median_pka"].notna().all()
    assert first["candidate_global_count"].ge(1).all()


def test_candidate_minimum_count_filter_operates_at_whole_molecule_level(tmp_path):
    assignments_path = tmp_path / "assignments.csv"
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame([
        _assignment_row(),
        _assignment_row(
            record_index=2,
            smiles="CN",
            selected_group_label="primary_amine",
            final_group_label="primary_amine",
            final_group_family="amine",
            resolved_groups="primary_amine",
            resolved_group_instances="primary_amine",
            resolved_group_count=1,
            assignment_status="single_group",
        ),
    ]).to_csv(assignments_path, index=False)
    _summary_rows().to_csv(summary_path, index=False)

    result = build_candidate_dataset(DatasetBuildConfig(
        assignments_path=str(assignments_path),
        summary_path=str(summary_path),
        min_candidates=2,
    ))

    assert result["molecule_key"].nunique() == 1
    assert set(result["candidate_label"]) == {"imidazole_basicity", "ketone"}


def test_duplicate_summary_keys_are_rejected_instead_of_multiplying_rows(tmp_path):
    assignments_path = tmp_path / "assignments.csv"
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame([_assignment_row()]).to_csv(assignments_path, index=False)
    summary = _summary_rows()
    pd.concat([summary, summary.iloc[[0]]], ignore_index=True).to_csv(
        summary_path, index=False
    )

    with pytest.raises(pd.errors.MergeError, match="not a many-to-one merge"):
        build_candidate_dataset(DatasetBuildConfig(
            assignments_path=str(assignments_path),
            summary_path=str(summary_path),
        ))


def _split_frame():
    return pd.DataFrame({
        "molecule_key": ["ring_a", "ring_a", "ring_b", "acyclic_a", "acyclic_b"],
        "smiles": ["Cc1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "CCO", "CCCO"],
    })


def test_random_split_is_deterministic_and_never_splits_one_molecule():
    frame = _split_frame()
    train_a, test_a = random_molecule_split(frame, test_frac=0.5, seed=23)
    train_b, test_b = random_molecule_split(frame, test_frac=0.5, seed=23)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(test_a, test_b)
    assert set(frame.iloc[train_a]["molecule_key"]).isdisjoint(
        set(frame.iloc[test_a]["molecule_key"])
    )
    assert sorted(np.concatenate([train_a, test_a]).tolist()) == list(range(len(frame)))


@pytest.mark.parametrize("test_frac", [0.0, 1.0, -0.1, 1.1])
def test_split_fraction_must_define_two_partitions(test_frac):
    with pytest.raises(ValueError, match="strictly between zero and one"):
        random_molecule_split(_split_frame(), test_frac=test_frac)


def test_split_rejects_missing_or_insufficient_molecule_keys():
    with pytest.raises(ValueError, match="non-missing"):
        random_molecule_split(
            pd.DataFrame({"molecule_key": ["a", None], "smiles": ["CC", "CCC"]})
        )
    with pytest.raises(ValueError, match="At least two distinct molecules"):
        random_molecule_split(
            pd.DataFrame({"molecule_key": ["a", "a"], "smiles": ["CC", "CC"]})
        )


def test_scaffold_split_groups_ring_analogues_but_not_all_acyclic_molecules():
    frame = _split_frame()
    train_idx, test_idx = scaffold_molecule_split(frame, test_frac=0.5)
    partition = {
        key: "test" if index in set(test_idx) else "train"
        for index, key in enumerate(frame["molecule_key"])
    }

    assert _murcko_scaffold("Cc1ccccc1") == _murcko_scaffold("Oc1ccccc1")
    assert partition["ring_a"] == partition["ring_b"]
    assert _murcko_scaffold("CCO") != _murcko_scaffold("CCCO")
    assert set(frame.iloc[train_idx]["molecule_key"]).isdisjoint(
        set(frame.iloc[test_idx]["molecule_key"])
    )


def test_scaffold_split_rejects_inconsistent_structure_for_one_molecule_key():
    frame = pd.DataFrame({
        "molecule_key": ["same", "same", "other"],
        "smiles": ["CC", "CCC", "c1ccccc1"],
    })
    with pytest.raises(ValueError, match="exactly one SMILES"):
        scaffold_molecule_split(frame)


def test_scaffold_split_fails_loudly_when_no_independent_scaffold_exists():
    frame = pd.DataFrame({
        "molecule_key": ["a", "b"],
        "smiles": ["Cc1ccccc1", "Oc1ccccc1"],
    })
    with pytest.raises(ValueError, match="empty partition"):
        scaffold_molecule_split(frame)


def test_split_manifest_uses_positional_indices_with_nondefault_dataframe_index():
    frame = pd.DataFrame({
        "molecule_key": ["a", "b", "c"],
        "smiles": ["CC", "CCC", "CCCC"],
    }, index=[10, 20, 30])

    manifest = split_manifest(frame, train_idx=[0, 1], test_idx=[2])

    assert manifest["split"].tolist() == ["train", "train", "test"]


def test_split_manifest_rejects_row_or_molecule_leakage():
    frame = pd.DataFrame({
        "molecule_key": ["same", "same", "other"],
        "smiles": ["CC", "CC", "CCC"],
    })
    with pytest.raises(ValueError, match="indices overlap"):
        split_manifest(frame, train_idx=[0], test_idx=[0, 2])
    with pytest.raises(ValueError, match="both train and test"):
        split_manifest(frame, train_idx=[0], test_idx=[1, 2])


def _write_trusted_raw_files(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in training_data_resolver.TRUSTED_TRAINING_SDF_FILES:
        (raw_dir / name).write_text("placeholder\n")


def test_training_data_resolver_requires_every_declared_raw_source(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / training_data_resolver.TRUSTED_TRAINING_SDF_FILES[0]).write_text(
        "placeholder\n"
    )

    with pytest.raises(FileNotFoundError, match="Missing expected trusted"):
        training_data_resolver._trusted_raw_paths(str(raw_dir))


def test_fresh_curated_outputs_are_reused_without_rebuilding(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    _write_trusted_raw_files(raw_dir)
    assignments = tmp_path / "processed" / "assignments.csv"
    summary = tmp_path / "processed" / "summary.csv"
    assignments.parent.mkdir()
    assignments.write_text("pka_source_method\nexperimental\n")
    summary.write_text("ready\n")
    future = max(path.stat().st_mtime for path in raw_dir.iterdir()) + 100.0
    os.utime(assignments, (future, future))
    os.utime(summary, (future, future))
    monkeypatch.setattr(training_data_resolver, "_dependency_paths", lambda **k: [])
    monkeypatch.setattr(
        functional_group_pka_analysis,
        "build_functional_group_table",
        lambda **kwargs: pytest.fail("fresh outputs should not rebuild"),
    )

    resolved = training_data_resolver.ensure_curated_training_data(
        assignments_path=str(assignments),
        summary_path=str(summary),
        raw_dir=str(raw_dir),
        site_overrides_path=str(tmp_path / "overrides.csv"),
    )

    assert resolved == (str(assignments), str(summary))


def test_stale_curated_outputs_rebuild_from_experimental_only_inputs(
    tmp_path, monkeypatch
):
    raw_dir = tmp_path / "raw"
    _write_trusted_raw_files(raw_dir)
    assignments = tmp_path / "processed" / "assignments.csv"
    summary = tmp_path / "processed" / "summary.csv"
    overrides = tmp_path / "overrides.csv"
    overrides.write_text("override\n")
    captured = {}
    source = pd.DataFrame({
        "single_group_strict": [True, True, False],
        "final_group_summary_label": ["primary_amine", "phenol", "ketone"],
        "distribution_ok": [True, False, True],
        "pka_value": [10.2, 9.9, 20.0],
        "pka_source_method": ["experimental", "experimental", "experimental"],
    })

    def fake_build(**kwargs):
        captured.update(kwargs)
        return source.copy()

    def fake_summary(frame):
        captured["summary_input"] = frame.copy()
        return pd.DataFrame({
            "final_group_label": ["primary_amine"],
            "mean": [10.2],
            "median": [10.2],
            "iqr": [0.0],
            "count": [1],
        })

    monkeypatch.setattr(training_data_resolver, "_dependency_paths", lambda **k: [])
    monkeypatch.setattr(
        functional_group_pka_analysis, "build_functional_group_table", fake_build
    )
    monkeypatch.setattr(
        functional_group_pka_analysis, "build_summary_table", fake_summary
    )

    resolved = training_data_resolver.ensure_curated_training_data(
        assignments_path=str(assignments),
        summary_path=str(summary),
        raw_dir=str(raw_dir),
        overlap_threshold=0.8,
        site_overrides_path=str(overrides),
    )

    assert resolved == (str(assignments), str(summary))
    assert captured["allow_epik"] is False
    assert captured["overlap_threshold"] == pytest.approx(0.8)
    assert captured["include_files"] == list(
        training_data_resolver.TRUSTED_TRAINING_SDF_FILES
    )
    assert len(captured["summary_input"]) == 1
    assert captured["summary_input"].iloc[0]["final_group_label"] == "primary_amine"
    assert assignments.is_file()
    assert summary.is_file()

