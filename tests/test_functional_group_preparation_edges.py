"""Boundary and orchestration tests for functional-group data preparation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

import functional_group_pka_analysis as groups
import prepare_external_pka_datasets as external


def _site(label: str, site_type: str = "basic", atoms: set[int] | None = None, **extra):
    return {
        "label": label,
        "type": site_type,
        "atom_set": atoms or {0},
        "specificity": 10,
        "priority": 10,
        **extra,
    }


def test_training_labels_and_canonical_family_helpers_cover_missing_and_fallbacks():
    assert groups.training_group_label(None) is None
    assert groups.training_group_reason(None) == ""
    assert groups.training_group_label("primary_alcohol") != "primary_alcohol"
    assert "sparse" in groups.training_group_reason("primary_alcohol")
    assert groups.training_group_label("primary_amine") == "primary_amine"
    assert groups.training_group_reason("primary_amine") == "kept as standalone training label"
    assert groups.normalize_conjugate_family(None) is None
    assert groups.normalize_conjugate_family("phenolate") == "phenol_phenolate"
    assert groups.canonical_family_label(None, None) is None
    assert groups.canonical_family_label("phenolate") == "phenol"
    assert groups.canonical_family_label("unknown", "unknown_family") == "unknown"


def test_standard_measurement_extraction_preserves_both_experimental_and_marvin_metadata():
    molecule = Chem.MolFromSmiles("CN")
    assert molecule is not None
    molecule.SetProp("pKa", "10.1")
    molecule.SetProp("pKa_type", "base")
    molecule.SetProp("marvin_pKa", "9.9")
    molecule.SetProp("marvin_atom", "1")
    molecule.SetProp("marvin_pKa_type", "basic")

    records = groups.extract_measurements(molecule)
    assert [record["pka_source_method"] for record in records] == ["experimental", "marvin"]
    assert records[0]["pka_type_raw"] == "base"
    assert records[1]["atom_index_raw"] == 1
    assert records[1]["pka_type_raw"] == "basic"


def test_epik_parser_skips_unrelated_and_invalid_properties_and_infers_acid_type():
    molecule = Chem.MolFromSmiles("CC(=O)O")
    assert molecule is not None
    molecule.SetProp("unrelated", "4.2")
    molecule.SetProp("r_epik_pKa_1", "nan")
    molecule.SetProp("r_epik_pKa_2", "4.7")
    molecule.SetProp("i_epik_pKa_atom_2", "bad")
    molecule.SetProp("s_epik_pKa_identifier_2", "acid centre")

    records = groups.extract_measurements(molecule, allow_epik=True)
    assert records == [{
        "pka_value": 4.7,
        "pka_source_method": "epik",
        "pka_type_raw": "acidic",
        "atom_index_raw": None,
        "site_identifier_raw": "acid centre",
    }]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", None), ("Acid dissociation", "acidic"), ("BASE", "basic"), ("other", "other")],
)
def test_pka_type_canonicalization(raw, expected):
    assert groups.canonicalize_pka_type(raw) == expected


def test_site_selection_reports_empty_fallback_and_rank_ties():
    selected, metadata = groups.select_pair_first_site([], "basic")
    assert selected is None
    assert metadata["pair_selection_reason"] == "no_resolved_sites"

    unpaired = _site("ketone", "neutral")
    selected, metadata = groups.select_pair_first_site([unpaired], "basic")
    assert selected == unpaired
    assert metadata["pair_selection_reason"] == "no_pair_family_match"

    tied = [_site("primary_amine"), _site("secondary_amine")]
    selected, metadata = groups.select_pair_first_site(tied, "basic")
    assert selected is not None
    assert metadata["pair_assignment_confidence"] == "ambiguous"
    assert metadata["pair_candidate_count"] == 2


def test_atom_index_matching_reports_none_and_unmatched_indices():
    sites = [_site("primary_amine", atoms={1})]
    matched, metadata = groups._sites_matching_atom_index(sites, None)
    assert matched == [] and metadata["atom_index_mode"] == "none"
    matched, metadata = groups._sites_matching_atom_index(sites, 99)
    assert matched == [] and metadata["atom_index_mode"] == "unmatched"


def test_pair_classification_and_expansion_reject_nonpairs_and_wrong_reported_type():
    assert groups.classify_pair_type(None, "amine") == ("individual", None)
    assert groups.classify_pair_type("ketone", "ketone") == ("individual", None)
    assert groups.classify_pair_type("unknown", "amine") == ("pair_type", "unknown_form")
    assert groups.expand_pair_site_transitions(_site("ketone", "neutral")) == []
    assert groups.expand_pair_site_transitions(_site("carboxylic_acid"), "basic") == []
    transition = groups.expand_pair_site_transitions(_site("carboxylic_acid"), "acidic")
    assert transition[0]["pair_family"] == "carboxyl"
    assert transition[0]["pair_member_form"] == "acid_form"


def test_serial_azole_selection_rejects_absent_invalid_and_mixed_coordinates():
    candidate = {
        **_site("imidazole_basicity", atoms={0, 1}),
        "pair_family": "imidazole_basicity",
        "coupling_group": "ring-a",
    }
    assert groups.select_serial_azole_transition_by_reference([], 7.0)[1].startswith("serial_azole")
    assert groups.select_serial_azole_transition_by_reference([candidate], "bad")[0] is None
    mixed = [candidate, {**candidate, "coupling_group": "ring-b"}]
    assert groups.select_serial_azole_transition_by_reference(mixed, 7.0)[1] == "not_one_serial_azole_coordinate"


@pytest.mark.parametrize(
    ("mode", "form", "label", "pka", "expected"),
    [
        ("individual", None, None, 7.0, (False, "missing_group")),
        ("pair_type", "acid_form", "carboxylic_acid", 99.0, (True, "pair_type")),
        ("pair_type", "unknown_form", "carboxylic_acid", 4.0, (False, "pair_unknown_form")),
        ("individual", None, "unwindowed", 99.0, (True, "individual_no_window")),
        ("individual", None, "ketone", 20.0, (True, "individual_in_window")),
        ("individual", None, "ketone", -20.0, (False, "individual_out_of_window")),
    ],
)
def test_distribution_sanity_contract(mode, form, label, pka, expected):
    assert groups.distribution_sanity_check(mode, form, label, pka) == expected


@pytest.mark.parametrize(
    ("family", "pka_type", "pka", "flag"),
    [
        (None, None, -1.0, "unknown_type_extreme"),
        (None, None, 7.0, "ok"),
        ("carboxyl", "basic", 5.0, "acidic_family_vs_basic_type"),
        ("carboxyl", "acidic", 14.0, "acidic_too_high_hard"),
        ("carboxyl", "acidic", 12.0, "acidic_high_soft"),
        ("amine", "acidic", 5.0, "basic_family_vs_acidic_type"),
        ("amine", "basic", 0.5, "basic_too_low_hard"),
        ("amine", "basic", 2.0, "basic_low_soft"),
        ("other", "basic", 2.0, "basic_low_soft"),
        ("other", None, 7.0, "ok"),
    ],
)
def test_conjugate_sanity_flags(family, pka_type, pka, flag):
    assert groups.conjugate_check_flag(family, pka_type, pka) == flag


def test_two_faced_type_reinterpretation_only_flips_family_mismatches():
    assert groups.reinterpret_pka_type_for_two_faced(None, "amine") == (None, False)
    assert groups.reinterpret_pka_type_for_two_faced("acidic", "amine") == ("basic", True)
    assert groups.reinterpret_pka_type_for_two_faced("basic", "carboxyl") == ("acidic", True)
    assert groups.reinterpret_pka_type_for_two_faced("basic", "amine") == ("basic", False)


def test_refined_labels_distinguish_aromatic_context_and_preserve_other_groups():
    aromatic = Chem.MolFromSmiles("NC(=O)c1ccccc1")
    aliphatic = Chem.MolFromSmiles("NC(=O)C")
    assert aromatic is not None and aliphatic is not None
    # The carbonyl carbon directly touches the aromatic ring in benzamide.
    aromatic_carbonyl = next(atom.GetIdx() for atom in aromatic.GetAtoms() if atom.GetSymbol() == "C" and any(b.GetBondTypeAsDouble() == 2 for b in atom.GetBonds()))
    aliphatic_carbonyl = next(atom.GetIdx() for atom in aliphatic.GetAtoms() if atom.GetSymbol() == "C" and any(b.GetBondTypeAsDouble() == 2 for b in atom.GetBonds()))
    assert groups.refine_group_label(aromatic, _site("amide", atoms={aromatic_carbonyl}), "amide") == "amide_aromatic"
    assert groups.refine_group_label(aliphatic, _site("amide", atoms={aliphatic_carbonyl}), "amide") == "amide_aliphatic"
    assert groups.refine_group_label(aliphatic, None, "ketone") == "ketone"
    assert groups.refine_group_label(aliphatic, _site("ketone"), None) is None


def test_sdf_file_resolution_filters_incorrect_files_and_validates_requested_names(tmp_path: Path):
    (tmp_path / "b.sdf").touch()
    (tmp_path / "a.sdf").touch()
    (tmp_path / "INCORRECT_bad.sdf").touch()
    assert [Path(path).name for path in groups._resolve_sdf_files(str(tmp_path))] == ["a.sdf", "b.sdf"]
    assert groups._resolve_sdf_files(str(tmp_path), ["INCORRECT_bad.sdf"]) == []
    with pytest.raises(FileNotFoundError, match="missing.sdf"):
        groups._resolve_sdf_files(str(tmp_path), ["missing.sdf"])


def test_override_schema_validation_and_empty_override_file(tmp_path: Path):
    frame = pd.DataFrame({"selected_group_label": ["primary_amine"], "final_group_label": ["primary_amine"]})
    malformed = tmp_path / "malformed.csv"
    pd.DataFrame({"override_group_label": ["primary_amine"]}).to_csv(malformed, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        groups.apply_site_resolution_overrides(frame, str(malformed))

    empty = tmp_path / "empty.csv"
    pd.DataFrame(columns=[*groups.SITE_OVERRIDE_KEY_COLUMNS, "override_group_label"]).to_csv(empty, index=False)
    result = groups.apply_site_resolution_overrides(frame, str(empty))
    assert result["site_override_applied"].tolist() == [False]


def _summary_contract_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "source_file": "a.sdf", "record_index": 0, "smiles": "CN", "pka_value": 10.0,
            "pka_source_method": "experimental", "pka_type_canonical": "basic", "pka_type_effective": "basic",
            "assignment_status": "single_group", "resolved_group_count": 1, "resolved_groups": "primary_amine",
            "resolved_group_instances": "primary_amine", "final_group_label": "primary_amine",
            "final_group_summary_label": "primary_amine", "final_group_family": "amine",
            "group_mode": "pair_type", "pair_family": "amine", "pair_member_form": "base_form",
            "pair_assignment_confidence": "high", "pair_candidate_count": 1,
            "pair_candidate_labels": "primary_amine", "neutral_input_risk": True,
            "molecule_formal_charge": 0, "distribution_ok": True, "conjugate_check_flag": "ok",
            "two_faced_type_flip": False, "single_group_strict": True,
        },
        {
            "source_file": "b.sdf", "record_index": 1, "smiles": "CC(=O)O", "pka_value": 4.0,
            "pka_source_method": "experimental", "pka_type_canonical": "acidic", "pka_type_effective": "acidic",
            "assignment_status": "multi_group", "resolved_group_count": 2,
            "resolved_groups": "carboxylic_acid|primary_amine",
            "resolved_group_instances": "carboxylic_acid|primary_amine", "final_group_label": "carboxylic_acid",
            "final_group_summary_label": "carboxylic_acid", "final_group_family": "carboxyl",
            "group_mode": "pair_type", "pair_family": "carboxyl", "pair_member_form": "acid_form",
            "pair_assignment_confidence": "ambiguous", "pair_candidate_count": 2,
            "pair_candidate_labels": "carboxylic_acid|primary_amine", "neutral_input_risk": False,
            "molecule_formal_charge": 0, "distribution_ok": True, "conjugate_check_flag": "ok",
            "two_faced_type_flip": False, "single_group_strict": False,
        },
    ])


def test_all_assignment_summary_tables_and_pair_training_filter_have_stable_outputs():
    frame = _summary_contract_frame()
    summary = groups.build_summary_table(frame)
    assert set(["q1", "q3", "iqr"]) <= set(summary.columns)
    assert len(groups.build_conjugate_check_summary(frame)) == 2
    assert len(groups.build_two_faced_resolution_summary(frame)) == 2
    assert len(groups.build_pair_type_summary(frame)) == 2
    training = groups.build_pair_training_labels(frame)
    assert training["target_label"].tolist() == ["base_form"]
    assert groups.build_pair_training_labels(frame.assign(distribution_ok=False)).empty
    assert list(groups.build_multi_group_group_summary(frame.iloc[0:0]).columns) == ["group_label", "count", "fraction_in_multi"]
    assert list(groups.build_multi_group_combo_summary(frame.iloc[0:0]).columns) == ["group_combo", "count", "fraction_in_multi_rows"]


def test_distribution_plot_fallback_writes_png_and_rejects_empty_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    real_import = groups.importlib.import_module

    def without_seaborn(name: str):
        if name == "seaborn":
            raise ImportError("test fallback")
        return real_import(name)

    monkeypatch.setattr(groups.importlib, "import_module", without_seaborn)
    output = tmp_path / "groups.png"
    groups.plot_group_distributions(_summary_contract_frame(), str(output), min_group_count=1)
    assert output.exists() and output.stat().st_size > 0
    with pytest.raises(ValueError, match="No groups"):
        groups.plot_group_distributions(_summary_contract_frame(), str(output), min_group_count=99)


def test_functional_group_cli_writes_the_complete_output_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    frame = _summary_contract_frame()
    monkeypatch.setattr(groups, "build_functional_group_table", lambda **_kwargs: frame.copy())
    monkeypatch.setattr(groups, "plot_group_distributions", lambda _df, path, **_kwargs: Path(path).write_bytes(b"png"))
    monkeypatch.setattr(sys, "argv", ["functional_group_pka_analysis.py", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path / "out"), "--min-group-count", "1"])
    groups.main()
    expected = {
        "functional_group_assignments.csv", "functional_group_single_group_subset.csv",
        "functional_group_pka_summary.csv", "functional_group_pka_distributions.png",
        "functional_group_strictness_checks.csv", "functional_group_multi_group_subset.csv",
        "functional_group_multi_group_identified_groups.csv", "functional_group_multi_group_combinations.csv",
        "functional_group_pka_summary_multi_group.csv", "functional_group_conjugate_check.csv",
        "functional_group_two_faced_resolution.csv", "functional_group_pair_type_summary.csv",
        "functional_group_pair_training_labels.csv",
    }
    assert expected <= {path.name for path in (tmp_path / "out").iterdir()}


def test_external_helper_boundaries_and_manifest_provenance(tmp_path: Path):
    assert external._formal_charge(None) == 0
    assert external._formal_charge(Chem.MolFromSmiles("C[NH3+]")) == 1
    assert external._top_label_counts(pd.Series(dtype=object)) == "{}"
    assert json.loads(external._top_label_counts(pd.Series(["b", "a", "b"]))) == {"a": 1, "b": 2}
    assert external._safe_rate(pd.Series(dtype=float)) == 0.0
    assert external._safe_rate(pd.Series([True, False, None])) == pytest.approx(1 / 3)

    assignments = _summary_contract_frame()
    row = external._dataset_manifest_row("test", "external", assignments, assignments.iloc[:1], assignments.iloc[:0], {"extra": "kept"})
    assert row["molecules"] == 2
    assert row["single_group_rows"] == 1
    assert row["multi_group_rows"] == 1
    assert row["extra"] == "kept"


def test_external_curated_copy_strict_empty_frames_and_candidate_table_schemas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert list(external._strict_summary_frame(pd.DataFrame()).columns) == external.EMPTY_SUMMARY_COLUMNS
    rejected = pd.DataFrame({
        "single_group_strict": [True], "final_group_summary_label": ["amine"],
        "distribution_ok": [True], "pka_source_method": ["marvin"],
        "training_group_label": ["amine"], "pka_value": [10.0],
    })
    assert external._strict_summary_frame(rejected).empty

    curated = tmp_path / "curated.csv"
    curated.write_text("final_group_label,count\namine,1\n")
    monkeypatch.setattr(external, "ensure_curated_training_data", lambda: ("assignments.csv", str(curated)))
    copied = external._copy_curated_summary(str(tmp_path / "nested/copied.csv"))
    assert Path(copied).read_text() == curated.read_text()

    assignments = tmp_path / "empty-assignments.csv"
    pd.DataFrame(columns=["smiles"]).to_csv(assignments, index=False)
    (tmp_path / "candidate-out").mkdir()
    self_candidates, curated_candidates, curated_copy = external._write_candidate_tables(
        str(assignments), str(tmp_path / "self.csv"), str(tmp_path / "candidate-out")
    )
    assert list(self_candidates.columns) == external.EMPTY_CANDIDATE_COLUMNS
    assert list(curated_candidates.columns) == external.EMPTY_CANDIDATE_COLUMNS
    assert Path(curated_copy).exists()


def test_external_candidate_tables_build_both_self_and_curated_prior_views(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "out"
    out.mkdir()
    assignments = tmp_path / "assignments.csv"
    pd.DataFrame({"smiles": ["CN"]}).to_csv(assignments, index=False)
    summary = tmp_path / "summary.csv"
    summary.write_text("final_group_label,count\namine,1\n")
    curated = tmp_path / "curated.csv"
    curated.write_text(summary.read_text())
    monkeypatch.setattr(external, "ensure_curated_training_data", lambda: ("ignored.csv", str(curated)))
    seen_summaries: list[str] = []

    def fake_candidates(config):
        seen_summaries.append(str(config.summary_path))
        return pd.DataFrame({"candidate": [Path(config.summary_path).name]})

    monkeypatch.setattr(external, "build_candidate_dataset", fake_candidates)
    self_view, curated_view, curated_copy = external._write_candidate_tables(
        str(assignments), str(summary), str(out)
    )
    assert self_view.iloc[0]["candidate"] == "summary.csv"
    assert curated_view.iloc[0]["candidate"] == "summary_curated_priors.csv"
    assert seen_summaries == [str(summary), curated_copy]


def test_external_smiles_analysis_caches_valid_and_invalid_results():
    external._SMILES_ANALYSIS_CACHE.clear()
    invalid = external._analyze_smiles("not smiles", 0.75)
    assert invalid["mol"] is None and invalid["status"] == "no_group"
    assert external._analyze_smiles("not smiles", 0.75) is invalid
    valid = external._analyze_smiles("CC(=O)O", 0.75)
    assert valid["smiles"] == "CC(=O)O"
    assert valid["resolved_sites"]
    forms = external._pair_family_form_map(valid["resolved_sites"])
    assert forms["carboxyl"]["acid_form"] >= 1
    assert any(site["pair_family"] == "carboxyl" for site in external._expanded_pair_sites(valid["resolved_sites"]))


def test_reference_transition_becomes_a_fully_provenanced_benchmark_assignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    microstates = pd.DataFrame([
        {
            "dataset_name": "SYNTHETIC", "compound_id": "ACID", "microstate_id": "ACID_micro000",
            "smiles": "CC(=O)O", "formal_charge": 0, "is_reference": True,
            "microstate_csv": "ACID_microstates.csv",
        },
        {
            "dataset_name": "SYNTHETIC", "compound_id": "ACID", "microstate_id": "ACID_micro001",
            "smiles": "CC(=O)[O-]", "formal_charge": -1, "is_reference": False,
            "microstate_csv": "ACID_microstates.csv",
        },
    ])
    transitions = external._reference_transition_records(microstates, 0.75)
    assert len(transitions) == 1 and transitions[0]["truth_status"] == "unambiguous"
    observation = pd.Series({
        "dataset_name": "SYNTHETIC", "compound_id": "ACID", "observation_index": 0,
        "pka_value": 4.76, "pka_sem": 0.02, "reference_smiles": "CC(=O)O",
    })
    assignment = external._benchmark_assignment_row(observation, pd.Series(transitions[0]), 0.75)
    assert assignment["final_group_label"] == "carboxylic_acid"
    assert assignment["final_group_family"] == "carboxyl"
    assert assignment["pair_member_form"] == "acid_form"
    assert assignment["benchmark_truth_status"] == "unambiguous"
    assert assignment["atom_site_match"] is True

    monkeypatch.setattr(external, "_write_candidate_tables", lambda *_args: (
        pd.DataFrame({"self": [1]}), pd.DataFrame({"curated": [1]}), "curated.csv"
    ))
    out = tmp_path / "benchmark"
    self_candidates, curated_candidates = external._write_benchmark_outputs(
        str(out), pd.DataFrame([observation]), microstates,
        pd.DataFrame(transitions), pd.DataFrame([assignment]),
    )
    assert len(self_candidates) == len(curated_candidates) == 1
    assert {"macro_observations.csv", "microstates.csv", "transition_catalog.csv", "unambiguous_assignments.csv", "summary_self_priors.csv"} <= {path.name for path in out.iterdir()}


def test_sampl_preparation_empty_filters_write_each_dataset_and_combined_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sampl7 = tmp_path / "sampl7"
    sampl8 = tmp_path / "sampl8"
    sampl7.mkdir()
    sampl8.mkdir()
    calls: list[str] = []

    def fake_write(out_dir, observations_df, microstates_df, transitions_df, assignments_df):
        calls.append(Path(out_dir).name)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return pd.DataFrame(columns=["x"]), pd.DataFrame(columns=["x"])

    monkeypatch.setattr(external, "_write_benchmark_outputs", fake_write)
    manifest = external.prepare_sampl_benchmarks(
        str(sampl7), str(sampl8), str(tmp_path / "out"),
        sampl7_compounds=[], sampl8_compounds=[],
    )
    assert calls == ["SAMPL7", "SAMPL8", "combined"]
    assert set(manifest["dataset_name"]) == {"SAMPL7", "SAMPL8", "combined"}
    assert manifest["macro_observations"].sum() == 0
    assert (tmp_path / "out/manifest.csv").exists()


def test_sampl_preparation_attaches_only_single_observation_single_label_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sampl7 = tmp_path / "sampl7"
    sampl8 = tmp_path / "sampl8"
    sampl7.mkdir()
    sampl8.mkdir()

    def observations(dataset_name: str) -> pd.DataFrame:
        return pd.DataFrame([{
            "dataset_name": dataset_name, "compound_id": "CMP1", "observation_index": 0,
            "pka_value": 4.2, "pka_sem": 0.1, "reference_smiles": "CC(=O)O",
        }])

    monkeypatch.setattr(external, "_read_sampl7_experimental", lambda *_args, **_kwargs: observations("SAMPL7"))
    monkeypatch.setattr(external, "_read_sampl8_experimental", lambda *_args, **_kwargs: observations("SAMPL8"))
    monkeypatch.setattr(external, "_load_microstates", lambda microstate_dir, dataset_name, **_kwargs: pd.DataFrame([{
        "dataset_name": dataset_name, "compound_id": "CMP1", "microstate_id": "CMP1_micro000",
        "smiles": "CC(=O)O", "formal_charge": 0, "is_reference": True, "microstate_csv": "CMP1.csv",
    }]))
    transition = {
        "truth_status": "unambiguous", "true_label_on_reference": "carboxylic_acid",
        "pka_type_canonical": "acidic", "reference_smiles": "CC(=O)O",
    }
    monkeypatch.setattr(external, "_reference_transition_records", lambda *_args, **_kwargs: [transition.copy()])
    monkeypatch.setattr(external, "_benchmark_assignment_row", lambda observation, _transition, **_kwargs: {
        "smiles": observation["reference_smiles"], "assignment_status": "single_group",
    })
    written: dict[str, pd.DataFrame] = {}

    def fake_write(out_dir, observations_df, microstates_df, transitions_df, assignments_df):
        written[Path(out_dir).name] = observations_df.copy()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return pd.DataFrame([{"candidate": 1}]), pd.DataFrame([{"candidate": 1}])

    monkeypatch.setattr(external, "_write_benchmark_outputs", fake_write)
    manifest = external.prepare_sampl_benchmarks(str(sampl7), str(sampl8), str(tmp_path / "out"))
    assert written["SAMPL7"].iloc[0]["usable_for_unambiguous_site_scoring"]
    assert written["SAMPL8"].iloc[0]["assigned_true_label"] == "carboxylic_acid"
    assert manifest.loc[manifest["dataset_name"] == "combined", "unambiguous_assignments"].iloc[0] == 2

