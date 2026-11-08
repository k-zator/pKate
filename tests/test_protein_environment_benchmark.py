from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_protein_environment as benchmark
import acquire_binding_shift_structures as acquire_structures
from protein_environment_correction import StructureAtom


def _pkadr_row(**updates):
    row = {
        "PDB": "1AAA",
        "Chain": "A",
        "ResName": "ASP",
        "ResID_in_PDB": "10",
        "Expt_pKa": "4.2",
        "pKa_Classification": "Main",
        "Warning": "",
        "Reference": "https://doi.org/10.1000/training",
        "Sequence_Identity_>_30%": "2BBB.B",
    }
    row.update(updates)
    return row


def _pkad2_table():
    return pd.DataFrame([
        {
            "PDB ID": "1AAA", "Res Name": "ASP", "Chain": "A",
            "Res ID": 10, "pKa": 4.2, "Reference": "10.1000/other",
        },
        {
            "PDB ID": "2BBB", "Res Name": "GLU", "Chain": "B",
            "Res ID": 20, "pKa": 5.1, "Reference": "10.1000/other",
        },
        {
            "PDB ID": "3CCC", "Res Name": "HIS", "Chain": "A",
            "Res ID": 30, "pKa": 6.4,
            "Reference": "https://doi.org/10.1000/training",
        },
        {
            "PDB ID": "4DDD", "Res Name": "HIS", "Chain": "A",
            "Res ID": 31, "pKa": 6.4, "Reference": "10.1000/new",
        },
        {
            "PDB ID": "5EEE", "Res Name": "HIS", "Chain": "A",
            "Res ID": 31, "pKa": 6.4, "Reference": "10.1000/new",
        },
        {
            "PDB ID": "6FFF", "Res Name": "ARG", "Chain": "A",
            "Res ID": 2, "pKa": ">12", "Reference": "10.1000/newer",
        },
    ])


def test_numeric_and_reference_normalization_reject_censored_labels():
    assert benchmark.exact_float("7.2") == 7.2
    assert benchmark.exact_float(">7.2") is None
    assert benchmark.exact_float("~7.2") is None
    assert benchmark.normalize_reference(
        "https://doi.org/10.1021/ABC.123?x=1"
    ) == "10.1021/abc.123"
    assert benchmark.normalize_reference(
        "https://pubmed.ncbi.nlm.nih.gov/12345/"
    ) == "pmid:12345"


def test_pkad2_ledger_excludes_every_declared_leakage_route_and_deduplicates():
    ledger = benchmark.build_pkad2_overlap_ledger(
        _pkad2_table(), [_pkadr_row()]
    )
    assert bool(ledger.loc[0, "overlap_exact_training_site"])
    assert bool(ledger.loc[1, "overlap_pkadr_30pct_link"])
    assert bool(ledger.loc[2, "overlap_training_reference"])
    selected = ledger[ledger["portability_selected"]]
    assert selected["pdb"].tolist() == ["4DDD"]
    assert bool(ledger.loc[4, "duplicate_measurement_row"])
    assert not bool(ledger.loc[5, "exact_numeric_pka"])
    benchmark.validate_evaluation_only_labels(ledger, name="synthetic")


def test_evaluation_provenance_fails_closed_for_computational_teacher():
    table = pd.DataFrame([{
        "label_origin": "computational",
        "computational_teacher": "PROPKA",
        "evaluation_only": True,
        "eligible_for_training": False,
    }])
    with pytest.raises(ValueError, match="teacher-derived"):
        benchmark.validate_evaluation_only_labels(table, name="bad panel")


def _binding_panel():
    return pd.DataFrame([
        {
            "case_id": "exact", "experimental_delta_pka": 2.0,
            "qualifier": "exact", "label_semantics": "individual_site_shift",
            "label_origin": "experimental", "computational_teacher": "",
            "evaluation_only": True, "eligible_for_training": False,
        },
        {
            "case_id": "lower", "experimental_delta_pka": 1.0,
            "qualifier": "lower_bound", "label_semantics": "individual_site_shift",
            "label_origin": "experimental", "computational_teacher": "",
            "evaluation_only": True, "eligible_for_training": False,
        },
        {
            "case_id": "upper", "experimental_delta_pka": -1.0,
            "qualifier": "upper_bound", "label_semantics": "combined_titration_midpoint_shift",
            "label_origin": "experimental", "computational_teacher": "",
            "evaluation_only": True, "eligible_for_training": False,
        },
    ])


def test_binding_scorer_keeps_exact_bounds_and_coupled_semantics_separate():
    submitted = pd.DataFrame([
        {"case_id": "exact", "predicted_delta_pka": 2.5,
         "prediction_semantics": "individual_site_shift"},
        {"case_id": "lower", "predicted_delta_pka": 1.5,
         "prediction_semantics": "individual_site_shift"},
        {"case_id": "upper", "predicted_delta_pka": 0.0,
         "prediction_semantics": "combined_titration_midpoint_shift"},
    ])
    scored, metrics = benchmark.score_binding_predictions(_binding_panel(), submitted)
    assert metrics["exact_mae"] == pytest.approx(0.5)
    assert metrics["censored_bound_satisfaction"] == pytest.approx(0.5)
    assert bool(scored.loc[scored.case_id == "lower", "bound_satisfied"].iloc[0])
    assert scored.loc[scored.case_id == "upper", "bound_violation"].iloc[0] == 1.0


def test_binding_scorer_rejects_wrong_prediction_semantics():
    submitted = pd.DataFrame([{
        "case_id": "upper", "predicted_delta_pka": -2.0,
        "prediction_semantics": "individual_site_shift",
    }])
    with pytest.raises(ValueError, match="semantics"):
        benchmark.score_binding_predictions(_binding_panel(), submitted)


def test_metric_block_reports_interval_coverage_and_baseline_gain():
    table = pd.DataFrame({
        "experimental_pka": [4.0, 6.0],
        "reference_pka_prediction": [3.0, 7.0],
        "raw_environment_prediction": [3.8, 6.2],
        "gated_environment_prediction": [3.9, 6.1],
        "gated_ci90_contains_experiment": [True, False],
    })
    metrics = benchmark._metric_block(table)
    assert metrics["reference_mae"] == 1.0
    assert metrics["gated_environment_mae"] == pytest.approx(0.1)
    assert metrics["gated_mae_gain_over_reference"] == pytest.approx(0.9)
    assert metrics["gated_interval_empirical_coverage"] == 0.5


def test_repository_binding_panel_is_locked_and_experimental():
    panel = benchmark.read_binding_panel(
        PROJECT_ROOT / "data/benchmarks/binding_induced_pka_shifts.csv"
    )
    assert len(panel) == 13
    assert int(panel["qualifier"].eq("exact").sum()) == 7
    assert int(
        panel["label_semantics"].eq("combined_titration_midpoint_shift").sum()
    ) == 5
    assert panel["computational_teacher"].eq("").all()


def test_repository_binding_mapping_exactly_covers_locked_panel():
    panel = benchmark.read_binding_panel(
        PROJECT_ROOT / "data/benchmarks/binding_induced_pka_shifts.csv"
    )
    mapping = benchmark.read_binding_mapping(
        PROJECT_ROOT
        / "data/benchmarks/binding_induced_pka_structure_mapping.json",
        panel,
    )
    assert len(mapping["cases"]) == 13
    assert set(mapping["structure_sha256"]) == {
        "1BVV", "1C5I", "1HHP", "1HPX", "1LYB", "1LYW", "1PFZ",
        "1QBS", "1SME", "2YAS", "3HVP", "6GCH",
    }
    assert all(case["mapping_status"].startswith("resolved")
               for case in mapping["cases"])


def test_coupled_midpoint_handles_individual_and_symmetric_pair():
    parameters = {
        "dielectric": 20.0,
        "debye_length_angstrom": 10.0,
        "maximum_absolute_coupling_log10": 0.75,
        "centering_ph": 7.0,
    }
    individual, detail = benchmark._coupled_midpoint(
        [{"predicted_pka": 6.25, "center": [0.0, 0.0, 0.0]}],
        parameters,
    )
    assert individual == 6.25
    assert detail["method"] == "individual_site_midpoint"
    pair, pair_detail = benchmark._coupled_midpoint([
        {"predicted_pka": 6.0, "center": [0.0, 0.0, 0.0]},
        {"predicted_pka": 6.0, "center": [4.0, 0.0, 0.0]},
    ], parameters)
    assert np.isfinite(pair)
    assert pair_detail["distance_angstrom"] == 4.0
    assert -0.75 <= pair_detail["coupling_log10_units"] <= 0.75


def test_csv_reader_digest_and_identifier_helpers(tmp_path: Path):
    source = tmp_path / "pkad2.csv"
    _pkad2_table().to_csv(source, index=False)
    loaded = benchmark.read_pkad2_wild_type(source)
    assert len(loaded) == 6
    assert benchmark.sha256(source) == benchmark.sha256(source)
    assert benchmark.clean_text(np.nan) == ""
    assert benchmark.residue_id(12.0) == "12"
    assert benchmark.residue_id("12A") == "12A"
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"PDB ID": ["1AAA"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        benchmark.read_pkad2_wild_type(bad)


def test_binding_structure_acquisition_downloads_and_verifies_checksum(
    tmp_path: Path, monkeypatch
):
    payload = b"data_test\n#\n"
    source = tmp_path / "source.cif"
    source.write_bytes(payload)
    digest = benchmark.sha256(source)
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({
        "download_url_template": "https://example.test/{pdb}.cif",
        "structure_sha256": {"1ABC": digest},
    }), encoding="utf-8")

    def copy_download(url, destination):
        assert url == "https://example.test/1ABC.cif"
        Path(destination).write_bytes(payload)
        return str(destination), None

    monkeypatch.setattr(
        acquire_structures.urllib.request, "urlretrieve", copy_download
    )
    output = tmp_path / "structures"
    first = acquire_structures.acquire(mapping, output)
    second = acquire_structures.acquire(mapping, output)
    assert first["structures"][0]["status"] == "downloaded_and_verified"
    assert second["structures"][0]["status"] == "verified_existing"
    assert (output / "1ABC.cif").read_bytes() == payload


def _atoms(sequence: str, chain: str = "A"):
    names = {"A": "ALA", "G": "GLY", "D": "ASP"}
    return [
        StructureAtom(
            group="ATOM", atom_name="CA", element="C",
            residue_name=names[letter], chain=chain, residue_id=str(index),
            insertion_code="", x=float(index), y=0.0, z=0.0,
        )
        for index, letter in enumerate(sequence, start=1)
    ]


def test_near_duplicate_audit_marks_similar_disjoint_and_short_chains(monkeypatch):
    ledger = pd.DataFrame([
        {"pdb": "2SIM", "chain": "A", "portability_selected": True},
        {"pdb": "3NEW", "chain": "A", "portability_selected": True},
        {"pdb": "4SHORT", "chain": "A", "portability_selected": True},
        {"pdb": "5SKIP", "chain": "A", "portability_selected": False},
    ])
    sequences = {
        "1AAA.cif.gz": _atoms("A" * 40),
        "2sim.pdb": _atoms("A" * 40),
        "3new.pdb": _atoms("G" * 40),
        "4short.pdb": _atoms("A" * 5),
    }
    monkeypatch.setattr(
        benchmark,
        "read_structure_atoms",
        lambda path: sequences[Path(path).name],
    )
    audited = benchmark.add_near_duplicate_audit(
        ledger,
        [_pkadr_row(Sequence_Identity_30="")],
        pkad2_structures="candidate",
        pkadr_structures="training",
    )
    status = dict(zip(audited.pdb, audited.sequence_audit_status))
    assert status["2SIM"] == "near_duplicate_training_chain"
    assert status["3NEW"] == "disjoint_at_proxy_threshold"
    assert status["4SHORT"] == "target_chain_sequence_too_short"
    assert status["5SKIP"] == "not_selected"


class _ConstantModel:
    def predict(self, frame):
        return np.full(len(frame), 0.5)


def test_portability_prediction_maps_structure_and_records_exclusion(
    tmp_path: Path, monkeypatch
):
    pdb = tmp_path / "1abc.pdb"
    pdb.write_text(
        "ATOM      1  OD1 ASP A  10       0.000   0.000   0.000  1.00 20.00           O\n"
        "ATOM      2  OD2 ASP A  10       1.000   0.000   0.000  1.00 20.00           O\n"
        "ATOM      3  CA  ALA A  11       3.000   0.000   0.000  1.00 20.00           C\n"
        "END\n",
        encoding="utf-8",
    )
    bundle = {
        "model": _ConstantModel(),
        "prediction_clip": [-5.0, 5.0],
        "calibration": {"chemistry": {"carboxyl": {
            "correction_scale": 1.0,
            "correction_evidence_tier": "test",
            "absolute_error_radius_90": 1.0,
        }}},
        "chemistry_p90_abs_error": {"carboxyl": 1.0},
        "oof_p90_abs_error": 1.0,
    }
    monkeypatch.setattr(benchmark, "_load_bundle", lambda _path: bundle)
    common = {
        "experimental_pka": 4.6, "portability_selected": True,
        "near_duplicate_disjoint_eligible": True, "label_origin": "experimental",
        "computational_teacher": "", "evaluation_only": True,
        "eligible_for_training": False,
    }
    ledger = pd.DataFrame([
        {**common, "pdb": "1ABC", "chain": "A", "residue_name": "ASP",
         "residue_id": "10"},
        {**common, "pdb": "1ABC", "chain": "A", "residue_name": "ASP",
         "residue_id": "99"},
    ])
    predictions, exclusions = benchmark.predict_pkad2_portability(
        ledger, structures_dir=tmp_path, model_path=tmp_path / "model.pkl"
    )
    assert len(predictions) == 1
    assert predictions.iloc[0]["gated_environment_prediction"] == pytest.approx(4.4)
    assert bool(predictions.iloc[0]["gated_ci90_contains_experiment"])
    assert len(exclusions) == 1
    assert "Missing target residue" in exclusions[0]["reason"]


def test_pkad2_metric_summary_separates_strict_subset():
    predictions = pd.DataFrame({
        "experimental_pka": [4.0, 6.0],
        "reference_pka_prediction": [3.5, 6.5],
        "raw_environment_prediction": [3.8, 6.2],
        "gated_environment_prediction": [3.9, 6.1],
        "gated_ci90_contains_experiment": [True, True],
        "chemistry": ["carboxyl", "imidazole"],
        "near_duplicate_disjoint_eligible": [True, False],
    })
    ledger = pd.DataFrame({
        "supported_residue": [True, True], "exact_numeric_pka": [True, True],
        "portability_eligible": [True, True], "portability_selected": [True, True],
    })
    metrics = benchmark.pkad2_metrics(predictions, ledger)
    assert metrics["near_duplicate_disjoint_selected_rows"] == 1
    assert metrics["near_duplicate_disjoint_metrics"]["rows"] == 1
    assert set(metrics["portability_metrics_by_chemistry"]) == {
        "carboxyl", "imidazole"
    }


def test_run_writes_auditable_outputs_and_scores_optional_binding_file(
    tmp_path: Path, monkeypatch
):
    pkadr = tmp_path / "pkadr.json"
    pkadr.write_text("[]\n", encoding="utf-8")
    pkad2 = tmp_path / "pkad2.csv"
    _pkad2_table().to_csv(pkad2, index=False)
    model = tmp_path / "model.pkl"
    model.write_bytes(b"placeholder")
    panel = tmp_path / "panel.csv"
    _binding_panel().to_csv(panel, index=False)
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({
        "schema_version": "1.0.0",
        "structure_sha256": {},
        "cases": [
            {
                "case_id": case_id,
                "apo": {"pdb": "1AAA", "target_groups": [[{
                    "chain": "A", "residue_name": "ASP", "residue_id": "10",
                }]]},
                "holo": {"pdb": "1AAA", "target_groups": [[{
                    "chain": "A", "residue_name": "ASP", "residue_id": "10",
                }]]},
            }
            for case_id in ("exact", "lower", "upper")
        ],
    }), encoding="utf-8")
    submitted = tmp_path / "submitted.csv"
    pd.DataFrame([
        {"case_id": "exact", "predicted_delta_pka": 2.0,
         "prediction_semantics": "individual_site_shift"},
        {"case_id": "lower", "predicted_delta_pka": 2.0,
         "prediction_semantics": "individual_site_shift"},
        {"case_id": "upper", "predicted_delta_pka": -2.0,
         "prediction_semantics": "combined_titration_midpoint_shift"},
    ]).to_csv(submitted, index=False)
    monkeypatch.setattr(
        benchmark, "_load_bundle",
        lambda _path: {"training_source_sha256": benchmark.sha256(pkadr)},
    )
    ledger = benchmark.build_pkad2_overlap_ledger(_pkad2_table(), [])
    ledger["near_duplicate_disjoint_eligible"] = False
    monkeypatch.setattr(benchmark, "build_pkad2_overlap_ledger", lambda *_: ledger)
    monkeypatch.setattr(benchmark, "add_near_duplicate_audit", lambda value, *_args, **_kwargs: value)
    predicted = pd.DataFrame({
        "experimental_pka": [4.0], "reference_pka_prediction": [3.9],
        "raw_environment_prediction": [4.0], "gated_environment_prediction": [4.0],
        "gated_ci90_contains_experiment": [True], "chemistry": ["carboxyl"],
        "near_duplicate_disjoint_eligible": [False],
    })
    monkeypatch.setattr(
        benchmark, "predict_pkad2_portability", lambda *_args, **_kwargs: (predicted, [])
    )
    output = tmp_path / "out"
    summary = benchmark.run(
        pkad2_path=pkad2, pkad2_structures=tmp_path, pkadr_path=pkadr,
        pkadr_structures=tmp_path, model_path=model,
        binding_panel_path=panel, binding_mapping_path=mapping,
        binding_structures=tmp_path / "missing-structures",
        binding_predictions_path=submitted,
        output_dir=output,
    )
    assert summary["submitted_binding_shift_predictions"]["exact_mae"] == 0.0
    assert summary["model_sha256"] == benchmark.sha256(model)
    assert (output / "benchmark_summary.json").is_file()
    assert (output / "binding_shift_submitted_scored_predictions.csv").is_file()


def test_main_forwards_cli_arguments(monkeypatch, capsys):
    called = {}
    monkeypatch.setattr(benchmark, "run", lambda **kwargs: called.update(kwargs) or {"ok": True})
    assert benchmark.main(["--pkad2", "input.csv", "--output", "out"]) == 0
    assert called["pkad2_path"] == "input.csv"
    assert called["output_dir"] == "out"
    assert '"ok": true' in capsys.readouterr().out.lower()
