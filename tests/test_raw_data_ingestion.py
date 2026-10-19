"""Data-independent tests for raw pKa source ingestion and provenance gates."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import pytest
from rdkit import Chem

import build_chembl_pka_truth_dataset as chembl
import functional_group_pka_analysis as group_analysis
import prepare_external_pka_datasets as external
import training_data_resolver


def _raw_observation(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "activity_id": 1,
        "record_id": 10,
        "molecule_chembl_id": "CHEMBL1",
        "parent_molecule_chembl_id": "CHEMBL1",
        "standard_value": "7.0",
        "standard_relation": "=",
        "relation": None,
        "potential_duplicate": 0,
        "standard_flag": 1,
        "data_validity_comment": "",
        "assay_confidence_score": 8,
        "canonical_smiles": "CCN",
        "assay_type": "B",
        "src_id": 7,
        "document_chembl_id": "CHEMBL_DOC_1",
    }
    row.update(updates)
    return row


def _microstate_frame(reference: str, other: str, other_charge: int) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dataset_name": "SYNTHETIC",
            "compound_id": "CMP1",
            "microstate_id": "CMP1_micro000",
            "smiles": reference,
            "formal_charge": 0,
            "is_reference": True,
            "microstate_csv": "CMP1_microstates.csv",
        },
        {
            "dataset_name": "SYNTHETIC",
            "compound_id": "CMP1",
            "microstate_id": "CMP1_micro001",
            "smiles": other,
            "formal_charge": other_charge,
            "is_reference": False,
            "microstate_csv": "CMP1_microstates.csv",
        },
    ])


def test_chembl_scalar_helpers_handle_missing_and_deterministic_ties() -> None:
    assert chembl.safe_float(None) is None
    assert chembl.safe_float("  ") is None
    assert chembl.safe_float("bad") is None
    assert chembl.safe_float(" 4.25 ") == pytest.approx(4.25)
    assert chembl.has_formal_charge_token("C[NH3+]")
    assert chembl.has_formal_charge_token("CC[N-2]")
    assert not chembl.has_formal_charge_token("CCN")
    assert chembl.mode_or_none([None, "", "b", "a"]) == "a"
    assert chembl.mode_or_none([None, " "]) is None


def test_chembl_summary_statistics_cover_weighted_and_unweighted_boundaries() -> None:
    assert chembl.percentile([], 0.5) is None
    assert chembl.percentile([4.0], 0.25) == 4.0
    assert chembl.percentile([0.0, 10.0], 0.25) == pytest.approx(2.5)
    assert chembl.weighted_median([], []) is None
    assert chembl.weighted_median([1.0], [1.0, 2.0]) is None
    assert chembl.weighted_median([1.0, 5.0, 9.0], [0.0, 5.0, 1.0]) == 5.0
    assert chembl.weighted_median([1.0, 9.0], [0.0, 0.0]) == 5.0


def test_chembl_url_building_and_pagination_normalization() -> None:
    url = chembl.build_url("activity.json", {"limit": 50, "x": ["a", "b"]})
    assert url.startswith(f"{chembl.BASE_API}/activity.json?")
    assert "limit=50" in url and "x=a" in url and "x=b" in url
    assert chembl.normalize_next_url(None, 100) is None

    official = "https://www.ebi.ac.uk/chembl/api/data/activity.json?offset=10"
    assert chembl.normalize_next_url(official, 100) == official
    normalized = chembl.normalize_next_url("/chembl/api/data/activity.json?offset=10", 100)
    assert normalized == f"{chembl.BASE_API}/activity.json?offset=10&limit=100"


def test_checkpoint_round_trip_supports_filename_without_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert chembl.load_checkpoint("missing.json") is None
    chembl.save_checkpoint("checkpoint.json", {"next_url": "next", "records_fetched": 2})
    assert chembl.load_checkpoint("checkpoint.json") == {
        "next_url": "next",
        "records_fetched": 2,
    }


def test_api_get_json_retries_network_error_without_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sleeps: list[float] = []

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"ok": true}'

    def fake_urlopen(_request: object, timeout: float) -> Response:
        nonlocal calls
        calls += 1
        assert timeout == 3.0
        if calls == 1:
            raise URLError("temporary")
        return Response()

    monkeypatch.setattr(chembl, "urlopen", fake_urlopen)
    monkeypatch.setattr(chembl.time, "sleep", sleeps.append)
    monkeypatch.setattr(chembl.random, "random", lambda: 0.0)

    assert chembl.api_get_json("https://example.invalid", 3.0, 1, 0.25) == {"ok": True}
    assert calls == 2
    assert sleeps == [pytest.approx(1.25)]


def test_fetch_activity_rows_paginates_checkpoints_and_skips_malformed_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = chembl.build_url("activity.json", {"limit": 2})
    second = "/chembl/api/data/activity.json?offset=2"
    payloads = {
        start: {
            "activities": [{"activity_id": 1}, "malformed", {"activity_id": 2}],
            "page_meta": {"next": second},
        },
        f"{chembl.BASE_API}/activity.json?offset=2&limit=2": {
            "activities": [{"activity_id": 3}],
            "page_meta": {"next": None},
        },
    }
    monkeypatch.setattr(chembl, "api_get_json", lambda url, *_args: payloads[url])
    checkpoint = tmp_path / "pull.json"

    rows, state = chembl.fetch_activity_rows(
        start_url=start,
        timeout=1,
        max_retries=0,
        sleep_seconds=0,
        checkpoint_path=str(checkpoint),
        checkpoint_state=None,
        max_pages=0,
        max_records=0,
    )

    assert [row["activity_id"] for row in rows] == [1, 2, 3]
    assert state["pages_fetched"] == 2
    assert state["records_fetched"] == 3
    assert state["next_url"] is None
    assert state["activities"] == rows
    assert json.loads(checkpoint.read_text())["activities"] == rows


def test_fetch_activity_rows_resumes_with_previously_fetched_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    next_url = chembl.build_url("activity.json", {"limit": 2, "offset": 2})
    monkeypatch.setattr(
        chembl,
        "api_get_json",
        lambda *_args: {"activities": [{"activity_id": 3}], "page_meta": {"next": None}},
    )
    rows, state = chembl.fetch_activity_rows(
        start_url=chembl.build_url("activity.json", {"limit": 2}),
        timeout=1,
        max_retries=0,
        sleep_seconds=0,
        checkpoint_path=str(tmp_path / "resume.json"),
        checkpoint_state={
            "next_url": next_url,
            "pages_fetched": 1,
            "records_fetched": 2,
            "activities": [{"activity_id": 1}, {"activity_id": 2}],
        },
        max_pages=0,
        max_records=0,
    )
    assert [row["activity_id"] for row in rows] == [1, 2, 3]
    assert state["pages_fetched"] == 2


def test_fetch_activity_rows_rejects_lossy_legacy_resume(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="legacy checkpoint"):
        chembl.fetch_activity_rows(
            start_url=chembl.build_url("activity.json", {"limit": 2}),
            timeout=1,
            max_retries=0,
            sleep_seconds=0,
            checkpoint_path=str(tmp_path / "resume.json"),
            checkpoint_state={"next_url": "next", "pages_fetched": 1, "records_fetched": 2},
            max_pages=0,
            max_records=0,
        )


def test_fetch_activity_rows_marks_mid_page_record_caps_as_non_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        chembl,
        "api_get_json",
        lambda *_args: {
            "activities": [{"activity_id": 1}, {"activity_id": 2}],
            "page_meta": {"next": "/activity.json?offset=2"},
        },
    )
    rows, state = chembl.fetch_activity_rows(
        start_url=chembl.build_url("activity.json", {"limit": 2}),
        timeout=1,
        max_retries=0,
        sleep_seconds=0,
        checkpoint_path=str(tmp_path / "capped.json"),
        checkpoint_state=None,
        max_pages=0,
        max_records=1,
    )
    assert [row["activity_id"] for row in rows] == [1]
    assert state["truncated_mid_page"] is True

    with pytest.raises(ValueError, match="partway through an API page"):
        chembl.fetch_activity_rows(
            start_url=chembl.build_url("activity.json", {"limit": 2}),
            timeout=1,
            max_retries=0,
            sleep_seconds=0,
            checkpoint_path=str(tmp_path / "capped.json"),
            checkpoint_state=state,
            max_pages=0,
            max_records=0,
        )


def test_fetch_entity_map_deduplicates_ids_and_ignores_malformed_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested: list[str] = []

    def fake_get(url: str, *_args: object) -> dict[str, object]:
        requested.append(url)
        return {
            "molecules": [
                {"molecule_chembl_id": "CHEMBL1", "name": "one"},
                "bad",
                {"name": "missing id"},
            ]
        }

    monkeypatch.setattr(chembl, "api_get_json", fake_get)
    result = chembl.fetch_entity_map(
        "molecule", "molecule_chembl_id", ["CHEMBL1", " CHEMBL1 ", ""],
        timeout=1, max_retries=0, sleep_seconds=0, chunk_size=10,
    )
    assert result == {"CHEMBL1": {"molecule_chembl_id": "CHEMBL1", "name": "one"}}
    assert len(requested) == 1
    assert "CHEMBL1" in requested[0]


def test_build_raw_dataframe_preserves_source_and_metadata_provenance() -> None:
    activities = [{
        "activity_id": 7,
        "record_id": 70,
        "assay_chembl_id": "A1",
        "molecule_chembl_id": "M1",
        "standard_type": "pKa",
        "standard_relation": "=",
        "standard_value": "9.2",
        "document_chembl_id": "D1",
        "assay_type": "fallback",
    }]
    frame = chembl.build_raw_dataframe(
        activities,
        assay_map={"A1": {"target_chembl_id": "T1", "assay_type": "B", "confidence_score": 9}},
        target_map={"T1": {"pref_name": "target", "organism": "human"}},
        molecule_map={"M1": {
            "molecule_structures": {"canonical_smiles": "C[NH3+]", "standard_inchi_key": "KEY"},
            "molecule_hierarchy": {"parent_chembl_id": "P1"},
        }},
        chembl_release="CHEMBL_99",
    )
    row = frame.iloc[0]
    assert row["activity_id"] == 7
    assert row["document_chembl_id"] == "D1"
    assert row["target_chembl_id"] == "T1"
    assert row["parent_molecule_chembl_id"] == "P1"
    assert row["canonical_smiles"] == "C[NH3+]"
    assert row["assay_type"] == "B"
    assert row["chembl_release"] == "CHEMBL_99"


def test_build_raw_dataframe_tolerates_malformed_optional_metadata() -> None:
    frame = chembl.build_raw_dataframe(
        [{"activity_id": 1, "assay_chembl_id": "A", "molecule_chembl_id": "M"}, "bad"],
        assay_map={"A": None},  # type: ignore[dict-item]
        target_map={},
        molecule_map={"M": {"molecule_structures": "bad", "molecule_hierarchy": []}},  # type: ignore[dict-item]
        chembl_release=None,
    )
    assert len(frame) == 1
    assert frame.iloc[0]["molecule_chembl_id"] == "M"
    assert pd.isna(frame.iloc[0]["canonical_smiles"])


def test_clean_observations_assigns_bounds_quality_and_stable_drop_reasons() -> None:
    raw = pd.DataFrame([
        _raw_observation(activity_id=1, standard_relation=None, relation=None),
        _raw_observation(activity_id=2, standard_value="5", standard_relation=">"),
        _raw_observation(activity_id=3, standard_value="bad", standard_relation="~"),
        _raw_observation(activity_id=4, standard_relation="~"),
        _raw_observation(activity_id=5, standard_value="21"),
        _raw_observation(activity_id=6, canonical_smiles=" "),
        _raw_observation(activity_id=7, parent_molecule_chembl_id=""),
        _raw_observation(
            activity_id=8,
            potential_duplicate=1,
            standard_flag=0,
            data_validity_comment="questionable",
            assay_confidence_score=0,
            canonical_smiles="CC.[Na+]",
        ),
    ])
    clean = chembl.clean_observations(raw).set_index("activity_id")

    assert clean.loc[1, "relation_normalized"] == "="
    assert clean.loc[1, "lower_bound"] == pytest.approx(7.0)
    assert clean.loc[1, "upper_bound"] == pytest.approx(7.0)
    assert clean.loc[2, "lower_bound"] == pytest.approx(5.0)
    assert pd.isna(clean.loc[2, "upper_bound"])
    assert clean.loc[3, "drop_reason"] == "missing_numeric_value"
    assert clean.loc[4, "drop_reason"] == "unsupported_relation"
    assert clean.loc[5, "drop_reason"] == "outside_reasonable_range"
    assert clean.loc[6, "drop_reason"] == "missing_canonical_smiles"
    assert clean.loc[7, "drop_reason"] == "missing_parent_molecule"
    assert clean.loc[8, "quality_score"] == pytest.approx(0.2)
    assert bool(clean.loc[8, "has_formal_charge_token"])
    assert bool(clean.loc[8, "is_multicomponent_smiles"])
    assert clean["obs_id"].is_unique


def test_summarize_molecules_combines_exact_censored_and_provenance() -> None:
    raw = pd.DataFrame([
        _raw_observation(activity_id=2, standard_value=5.0, canonical_smiles="CCN", document_chembl_id="D2"),
        _raw_observation(activity_id=1, standard_value=7.0, canonical_smiles="CC[NH3+]", document_chembl_id="D1"),
        _raw_observation(
            activity_id=3,
            standard_value=8.0,
            standard_relation=">",
            document_chembl_id="D2",
        ),
    ])
    summary = chembl.summarize_molecules(chembl.clean_observations(raw)).iloc[0]

    assert summary["prevalent_method"] == "exact_weighted_median"
    assert summary["prevalent_pka"] == pytest.approx(5.0)
    assert bool(summary["conflict_flag"])
    assert summary["conflict_note"] == "exact_vs_censored_disagreement"
    assert bool(summary["mixed_charge_forms"])
    assert summary["provenance_activity_ids"] == "1|2|3"
    assert summary["provenance_document_ids"] == "D1|D2"


def test_summarize_molecules_detects_conflicting_censored_bounds() -> None:
    raw = pd.DataFrame([
        _raw_observation(activity_id=1, standard_value=8.0, standard_relation=">"),
        _raw_observation(activity_id=2, standard_value=5.0, standard_relation="<"),
    ])
    summary = chembl.summarize_molecules(chembl.clean_observations(raw)).iloc[0]
    assert summary["prevalent_method"] == "censored_conflict"
    assert pd.isna(summary["prevalent_pka"])
    assert summary["conflict_note"] == "inconsistent_censored_bounds"


def test_summarize_molecules_empty_input_has_stable_schema() -> None:
    clean = chembl.clean_observations(pd.DataFrame([
        _raw_observation(standard_value="bad")
    ]))
    summary = chembl.summarize_molecules(clean)
    assert summary.empty
    assert {"molecule_key", "prevalent_pka", "conflict_flag", "provenance_activity_ids"} <= set(summary.columns)


@pytest.mark.parametrize(
    ("file_name", "role", "recommended_use", "included"),
    [
        ("chembl25.sdf", "source_training", "use_with_caution", True),
        ("novartis_cleaned_mono_unique_notraindata.sdf", "duplicate_local_curated", "do_not_use", False),
        ("new_notraindata.sdf", "external_test", "external_eval_candidate", True),
        ("combined_training.sdf", "training_bundle", "audit_only", False),
        ("mystery.sdf", "other", "review_first", False),
    ],
)
def test_czodrowski_source_policy_is_explicit(
    file_name: str,
    role: str,
    recommended_use: str,
    included: bool,
) -> None:
    metadata = external._czodrowski_usage_metadata(file_name)
    assert metadata["dataset_role"] == role
    assert metadata["recommended_use"] == recommended_use
    assert metadata["include_in_combined_recommended"] is included


def test_external_strict_summary_excludes_marvin_and_epik(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, pd.DataFrame] = {}
    assignments = pd.DataFrame({
        "single_group_strict": [True, True, True, False],
        "final_group_summary_label": ["primary_amine"] * 4,
        "training_group_label": ["primary_amine"] * 4,
        "distribution_ok": [True] * 4,
        "pka_source_method": ["experimental", "marvin", "epik", "experimental"],
        "pka_value": [10.0, 99.0, 88.0, 7.0],
    })

    def fake_summary(frame: pd.DataFrame) -> pd.DataFrame:
        captured["frame"] = frame.copy()
        return pd.DataFrame({"final_group_label": ["primary_amine"]})

    monkeypatch.setattr(external, "build_summary_table", fake_summary)
    external._strict_summary_frame(assignments)
    assert captured["frame"]["pka_source_method"].tolist() == ["experimental"]
    assert captured["frame"]["pka_value"].tolist() == [10.0]


def test_sampl7_reader_normalizes_filters_and_drops_unmeasured_rows(tmp_path: Path) -> None:
    path = tmp_path / "physical_property/pKa/analysis/macrostate_analysis"
    path.mkdir(parents=True)
    pd.DataFrame({
        "Molecule ID": ["SM1", "SM1", "SM2"],
        "pKa mean": [4.0, "N/A", 8.0],
        "pKa SEM": [0.1, 0.2, 0.3],
        "Isomeric SMILES": ["CC(=O)O", "CC(=O)O", "CN"],
    }).to_csv(path / "pKa_experimental_values.csv", index=False)

    result = external._read_sampl7_experimental(str(tmp_path), include_compounds=["SM1"])
    assert result["compound_id"].tolist() == ["SM1"]
    assert result["pka_value"].tolist() == [4.0]
    assert result["observation_index"].tolist() == [0]
    assert set(result["dataset_name"]) == {"SAMPL7"}


def test_sampl8_reader_strips_headers_and_numbers_observations(tmp_path: Path) -> None:
    path = tmp_path / "physical_properties/pKa"
    path.mkdir(parents=True)
    uncertainty = (
        "uncertainty (every pKa is followed by an uncertainty; multiple pairs "
        "indicate multiple pKa values; N/A if not measured)"
    )
    pd.DataFrame({
        " compound ID ": ["S8-1", "S8-1", "S8-2"],
        " pKa ": [2.5, 5.0, "N/A"],
        f" {uncertainty} ": [0.1, 0.2, "N/A"],
    }).to_csv(path / "experimental_pKas.csv", index=False)

    result = external._read_sampl8_experimental(str(tmp_path))
    assert result["compound_id"].tolist() == ["S8-1", "S8-1"]
    assert result["observation_index"].tolist() == [0, 1]
    assert result["pka_value"].tolist() == [2.5, 5.0]
    assert result["reference_smiles"].tolist() == ["", ""]


def test_sampl_reader_rejects_schema_drift(tmp_path: Path) -> None:
    path = tmp_path / "physical_properties/pKa"
    path.mkdir(parents=True)
    pd.DataFrame({"compound ID": ["S8-1"], "pKa": [2.5]}).to_csv(
        path / "experimental_pKas.csv", index=False
    )
    with pytest.raises(ValueError, match="missing columns"):
        external._read_sampl8_experimental(str(tmp_path))


def test_microstate_loader_validates_canonicalizes_and_computes_charge(tmp_path: Path) -> None:
    pd.DataFrame({
        " microstate ID ": ["CMP1_micro000", "CMP1_micro001"],
        " canonical isomeric SMILES ": ["OC(C)=O", "CC(=O)[O-]"],
    }).to_csv(tmp_path / "CMP1_microstates.csv", index=False)

    result = external._load_microstates(str(tmp_path), "TEST")
    assert result["microstate_id"].tolist() == ["CMP1_micro000", "CMP1_micro001"]
    assert result["formal_charge"].tolist() == [0, -1]
    assert result["is_reference"].tolist() == [True, False]
    assert result.loc[0, "smiles"] == "CC(=O)O"


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ({"wrong": ["x"]}, "missing columns"),
        ({"microstate ID": ["CMP_micro000"], "canonical isomeric SMILES": ["not smiles"]}, "invalid SMILES"),
        ({"microstate ID": ["CMP_micro001"], "canonical isomeric SMILES": ["CC"]}, "exactly one reference"),
        ({"microstate ID": ["CMP_micro000", "CMP_micro000"], "canonical isomeric SMILES": ["CC", "CC"]}, "Duplicate microstate IDs"),
    ],
)
def test_microstate_loader_rejects_malformed_truth_inputs(
    tmp_path: Path,
    rows: dict[str, list[str]],
    match: str,
) -> None:
    pd.DataFrame(rows).to_csv(tmp_path / "CMP_microstates.csv", index=False)
    with pytest.raises(ValueError, match=match):
        external._load_microstates(str(tmp_path), "TEST")


@pytest.mark.parametrize(
    ("reference", "other", "other_charge", "label", "family", "pka_type"),
    [
        ("CC(=O)O", "CC(=O)[O-]", -1, "carboxylic_acid", "carboxyl", "acidic"),
        ("CN", "C[NH3+]", 1, "primary_amine", "amine", "basic"),
    ],
)
def test_reference_microstate_transition_recovers_exact_site_and_direction(
    reference: str,
    other: str,
    other_charge: int,
    label: str,
    family: str,
    pka_type: str,
) -> None:
    records = external._reference_transition_records(
        _microstate_frame(reference, other, other_charge), overlap_threshold=0.75
    )
    assert len(records) == 1
    assert records[0]["truth_status"] == "unambiguous"
    assert records[0]["true_label_on_reference"] == label
    assert records[0]["true_family_on_reference"] == family
    assert records[0]["pka_type_canonical"] == pka_type
    assert records[0]["true_atom_index"] is not None


def test_reference_transition_ignores_nonadjacent_charge_states_and_missing_reference() -> None:
    nonadjacent = _microstate_frame("CC(=O)O", "CC(=O)[O-]", -2)
    assert external._reference_transition_records(nonadjacent, 0.75) == []
    no_reference = nonadjacent.assign(is_reference=False)
    assert external._reference_transition_records(no_reference, 0.75) == []


def test_czodrowski_preparation_excludes_duplicate_and_bundle_from_recommended_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "datasets"
    output_dir = tmp_path / "out"
    dataset_dir.mkdir()
    files = [
        "chembl25.sdf",
        "novartis_cleaned_mono_unique_notraindata.sdf",
        "combined_training.sdf",
    ]
    for file_name in files:
        (dataset_dir / file_name).touch()

    calls: list[dict[str, object]] = []

    def fake_build(**kwargs: object) -> pd.DataFrame:
        calls.append(dict(kwargs))
        marker = str(kwargs["include_files"][0])  # type: ignore[index]
        return pd.DataFrame({
            "smiles": [marker],
            "assignment_status": ["single_group"],
            "distribution_ok": [True],
            "atom_site_match": [True],
            "group_mode": ["individual"],
            "final_group_label": ["test"],
            "pka_source_method": ["experimental"],
        })

    monkeypatch.setattr(external, "build_functional_group_table", fake_build)
    monkeypatch.setattr(
        external,
        "_strict_summary_frame",
        lambda _frame: pd.DataFrame(columns=external.EMPTY_SUMMARY_COLUMNS),
    )
    monkeypatch.setattr(
        external,
        "_write_candidate_tables",
        lambda _a, _s, out_dir: (
            pd.DataFrame(columns=external.EMPTY_CANDIDATE_COLUMNS),
            pd.DataFrame(columns=external.EMPTY_CANDIDATE_COLUMNS),
            str(Path(out_dir) / "summary_curated_priors.csv"),
        ),
    )

    manifest = external.prepare_czodrowski_datasets(
        dataset_dir=str(dataset_dir),
        out_root=str(output_dir),
        include_files=files,
    )

    assert all(call["allow_epik"] is False for call in calls)
    recommended = pd.read_csv(output_dir / "combined_recommended/assignments.csv")
    assert recommended["smiles"].tolist() == ["chembl25.sdf"]
    duplicate = manifest[manifest["dataset_name"] == "novartis_cleaned_mono_unique_notraindata"].iloc[0]
    assert duplicate["recommended_use"] == "do_not_use"
    assert not bool(duplicate["include_in_combined_recommended"])


def test_curated_training_resolver_writes_experimental_rows_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for file_name in training_data_resolver.TRUSTED_TRAINING_SDF_FILES:
        (raw_dir / file_name).touch()
    assignments_path = tmp_path / "processed/assignments.csv"
    summary_path = tmp_path / "processed/summary.csv"
    source = pd.DataFrame({
        "single_group_strict": [True, True, True],
        "final_group_summary_label": ["primary_amine"] * 3,
        "training_group_label": ["primary_amine"] * 3,
        "distribution_ok": [True, True, True],
        "pka_source_method": ["experimental", "marvin", "epik"],
        "pka_value": [10.1, 99.0, 88.0],
    })
    captured: dict[str, pd.DataFrame] = {}

    monkeypatch.setattr(training_data_resolver, "_dependency_paths", lambda **_kwargs: [])
    monkeypatch.setattr(group_analysis, "build_functional_group_table", lambda **_kwargs: source.copy())

    def fake_summary(frame: pd.DataFrame) -> pd.DataFrame:
        captured["summary_input"] = frame.copy()
        return pd.DataFrame({"final_group_label": ["primary_amine"], "mean": [10.1]})

    monkeypatch.setattr(group_analysis, "build_summary_table", fake_summary)
    training_data_resolver.ensure_curated_training_data(
        assignments_path=str(assignments_path),
        summary_path=str(summary_path),
        raw_dir=str(raw_dir),
        site_overrides_path=str(tmp_path / "overrides.csv"),
    )

    written = pd.read_csv(assignments_path)
    assert written["pka_source_method"].tolist() == ["experimental"]
    assert written["pka_value"].tolist() == [10.1]
    assert captured["summary_input"]["pka_source_method"].tolist() == ["experimental"]


def test_fresh_but_nonexperimental_curated_assignments_are_rebuilt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for file_name in training_data_resolver.TRUSTED_TRAINING_SDF_FILES:
        (raw_dir / file_name).touch()
    assignments_path = tmp_path / "processed/assignments.csv"
    summary_path = tmp_path / "processed/summary.csv"
    assignments_path.parent.mkdir()
    pd.DataFrame({"pka_source_method": ["experimental", "marvin"]}).to_csv(
        assignments_path, index=False
    )
    summary_path.write_text("summary\nready\n")
    future = max(path.stat().st_mtime for path in raw_dir.iterdir()) + 100.0
    assignments_path.touch()
    summary_path.touch()
    import os
    os.utime(assignments_path, (future, future))
    os.utime(summary_path, (future, future))

    rebuilt = {"called": False}
    source = pd.DataFrame({
        "single_group_strict": [True],
        "final_group_summary_label": ["primary_amine"],
        "training_group_label": ["primary_amine"],
        "distribution_ok": [True],
        "pka_source_method": ["experimental"],
        "pka_value": [10.1],
    })

    def fake_build(**_kwargs: object) -> pd.DataFrame:
        rebuilt["called"] = True
        return source.copy()

    monkeypatch.setattr(training_data_resolver, "_dependency_paths", lambda **_kwargs: [])
    monkeypatch.setattr(group_analysis, "build_functional_group_table", fake_build)
    monkeypatch.setattr(
        group_analysis,
        "build_summary_table",
        lambda _frame: pd.DataFrame({"final_group_label": ["primary_amine"]}),
    )
    training_data_resolver.ensure_curated_training_data(
        assignments_path=str(assignments_path),
        summary_path=str(summary_path),
        raw_dir=str(raw_dir),
        site_overrides_path=str(tmp_path / "overrides.csv"),
    )
    assert rebuilt["called"]
    assert pd.read_csv(assignments_path)["pka_source_method"].tolist() == ["experimental"]


def test_measurement_extraction_does_not_use_epik_unless_explicitly_enabled() -> None:
    mol = Chem.MolFromSmiles("CN")
    assert mol is not None
    mol.SetProp("pKa", "10.2")
    mol.SetProp("ionization_type", "basic")
    mol.SetProp("marvin_pKa", "99")
    mol.SetProp("r_epik_pKa_1", "88")
    mol.SetProp("i_epik_pKa_atom_1", "2")
    mol.SetProp("s_epik_pKa_identifier_1", "basic")

    default_records = group_analysis.extract_measurements(mol, allow_epik=False)
    assert {row["pka_source_method"] for row in default_records} == {"experimental", "marvin"}
    epik_records = group_analysis.extract_measurements(mol, allow_epik=True)
    assert {row["pka_source_method"] for row in epik_records} == {"epik"}

