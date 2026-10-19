import hashlib
import json
import sys

import pandas as pd
import pytest

import run_canonical_protonation_pipeline as runner


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runner_hands_exact_stage2_artifact_to_stage3_and_writes_manifest(
    tmp_path, monkeypatch
):
    network = tmp_path / "network.csv"
    model = tmp_path / "stage2.pkl"
    stage2_output = tmp_path / "stage2.csv"
    stage3_output = tmp_path / "stage3.csv"
    stage3_dir = tmp_path / "stage3_artifacts"
    manifest_path = tmp_path / "manifest.json"
    network.write_text("molecule_id\nmol_1\n", encoding="utf-8")
    model.write_bytes(b"stage2-model")

    def fake_stage2(**kwargs):
        assert kwargs["network_dataset_path"] == str(network)
        stage2_output.write_text("stage2\nvalue\n", encoding="utf-8")
        return None, {
            "model_sha256": _sha256(model),
            "stage2_schema_version": "2.2.0",
            "stage2_free_energy_schema_version": "1.0.0",
        }

    def fake_stage3(**kwargs):
        assert kwargs["input_path"] == str(stage2_output)
        assert kwargs["max_tautomers_per_configuration"] == 16
        stage3_dir.mkdir(parents=True)
        stage3_output.write_text("stage3\nvalue\n", encoding="utf-8")
        (stage3_dir / "stage3_site_predictions.csv").write_text(
            "site_id\nsite_1\n", encoding="utf-8"
        )
        (stage3_dir / "stage3_ranked_tautomers.csv").write_text(
            "molecule_id\nmol_1\n", encoding="utf-8"
        )
        (stage3_dir / "stage3_empirical_call_calibration.json").write_text(
            json.dumps({"schema_version": "1.0.0"}), encoding="utf-8"
        )
        (stage3_dir / "metrics.json").write_text(
            json.dumps({"stage3_schema_version": "1.6.0"}), encoding="utf-8"
        )
        (stage3_dir / "stage3_output_schema.json").write_text(
            json.dumps({"schema_version": "1.6.0"}), encoding="utf-8"
        )
        return None, {
            "input_sha256": _sha256(stage2_output),
            "stage2_model_sha256": _sha256(model),
            "molecule_rows": 1,
            "complete_population_molecules": 1,
            "unavailable_incomplete_network_molecules": 0,
            "stage3_schema_version": "1.6.0",
            "tautomer_ranking_schema_version": "1.1.0",
            "max_tautomers_per_configuration": 16,
            "ranked_tautomer_rows": 3,
            "tautomer_truncated_configurations": 1,
            "empirical_pka_side_calibration": {
                "schema_version": "1.0.0",
                "deployment_confidence_summary": {"conflicts_below_half": 2},
            },
        }

    monkeypatch.setattr(runner, "apply_stage2", fake_stage2)
    monkeypatch.setattr(runner, "apply_stage3", fake_stage3)

    manifest = runner.run_canonical_pipeline(
        network_dataset_path=str(network),
        stage2_model_path=str(model),
        stage2_output_path=str(stage2_output),
        stage3_output_path=str(stage3_output),
        stage3_out_dir=str(stage3_dir),
        manifest_path=str(manifest_path),
    )

    assert manifest["stage2_output_sha256"] == _sha256(stage2_output)
    assert manifest["stage3_output_sha256"] == _sha256(stage3_output)
    assert manifest["stage2_application_reused"] is False
    assert manifest["stage3_schema_version"] == "1.6.0"
    assert manifest["tautomer_ranking_schema_version"] == "1.1.0"
    assert manifest["max_tautomers_per_configuration"] == 16
    assert manifest["empirical_calibration_schema_version"] == "1.0.0"
    assert manifest["ranked_tautomer_rows"] == 3
    assert manifest["empirical_pka_side_conflicts_below_half"] == 2
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest


def test_runner_reuses_only_prior_manifest_matched_stage2_artifact(tmp_path, monkeypatch):
    network = tmp_path / "network.csv"
    model = tmp_path / "stage2.pkl"
    stage2_output = tmp_path / "stage2.csv"
    stage3_output = tmp_path / "stage3.csv"
    stage3_dir = tmp_path / "stage3_artifacts"
    manifest_path = tmp_path / "manifest.json"
    network.write_text("molecule_id\nmol_1\n", encoding="utf-8")
    model.write_bytes(b"stage2-model")
    stage2_output.write_text(
        "stage2_schema_version,stage2_free_energy_schema_version\n2.2.0,1.0.0\n",
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({
        "network_dataset_sha256": _sha256(network),
        "stage2_model_sha256": _sha256(model),
        "stage2_output_sha256": _sha256(stage2_output),
    }), encoding="utf-8")

    def fail_stage2(**kwargs):
        raise AssertionError("Stage 2 must not run in reuse mode")

    def fake_stage3(**kwargs):
        stage3_dir.mkdir(parents=True)
        stage3_output.write_text("stage3\nvalue\n", encoding="utf-8")
        (stage3_dir / "stage3_site_predictions.csv").write_text("site_id\ns\n")
        (stage3_dir / "stage3_ranked_tautomers.csv").write_text("molecule_id\nmol_1\n")
        (stage3_dir / "stage3_empirical_call_calibration.json").write_text("{}")
        (stage3_dir / "stage3_output_schema.json").write_text("{}")
        (stage3_dir / "metrics.json").write_text("{}")
        return None, {
            "input_sha256": _sha256(stage2_output),
            "stage2_model_sha256": _sha256(model),
            "molecule_rows": 1,
            "complete_population_molecules": 1,
            "unavailable_incomplete_network_molecules": 0,
            "stage3_schema_version": "1.6.0",
            "tautomer_ranking_schema_version": "1.1.0",
            "max_tautomers_per_configuration": 16,
            "empirical_pka_side_calibration": {"schema_version": "1.0.0"},
        }

    monkeypatch.setattr(runner, "apply_stage2", fail_stage2)
    monkeypatch.setattr(runner, "apply_stage3", fake_stage3)
    manifest = runner.run_canonical_pipeline(
        network_dataset_path=str(network),
        stage2_model_path=str(model),
        stage2_output_path=str(stage2_output),
        stage3_output_path=str(stage3_output),
        stage3_out_dir=str(stage3_dir),
        manifest_path=str(manifest_path),
        reuse_stage2_output=True,
    )
    assert manifest["stage2_application_reused"] is True


def _canonical_fixture(tmp_path, monkeypatch, *, report_updates=None, omit=None):
    """Create a complete fake canonical run with configurable Stage 3 faults."""
    network = tmp_path / "network.csv"
    model = tmp_path / "stage2.pkl"
    stage2_output = tmp_path / "stage2.csv"
    stage3_output = tmp_path / "stage3.csv"
    stage3_dir = tmp_path / "stage3"
    manifest = tmp_path / "manifest.json"
    network.write_text("molecule_id\nmol\n", encoding="utf-8")
    model.write_bytes(b"model")

    def fake_stage2(**_kwargs):
        stage2_output.write_text("value\n1\n", encoding="utf-8")
        return None, {
            "model_sha256": _sha256(model),
            "stage2_schema_version": "2.2.0",
            "stage2_free_energy_schema_version": "1.0.0",
        }

    def fake_stage3(**_kwargs):
        stage3_dir.mkdir(exist_ok=True)
        artifacts = {
            stage3_output: "value\n1\n",
            stage3_dir / "stage3_site_predictions.csv": "site\n1\n",
            stage3_dir / "stage3_ranked_tautomers.csv": "tautomer\n1\n",
            stage3_dir / "stage3_empirical_call_calibration.json": "{}",
            stage3_dir / "metrics.json": "{}",
            stage3_dir / "stage3_output_schema.json": "{}",
        }
        for path, text in artifacts.items():
            if path.name != omit:
                path.write_text(text, encoding="utf-8")
        report = {
            "input_sha256": _sha256(stage2_output),
            "stage2_model_sha256": _sha256(model),
            "molecule_rows": 1,
            "complete_population_molecules": 1,
            "unavailable_incomplete_network_molecules": 0,
        }
        report.update(report_updates or {})
        return None, report

    monkeypatch.setattr(runner, "apply_stage2", fake_stage2)
    monkeypatch.setattr(runner, "apply_stage3", fake_stage3)
    kwargs = {
        "network_dataset_path": str(network),
        "stage2_model_path": str(model),
        "stage2_output_path": str(stage2_output),
        "stage3_output_path": str(stage3_output),
        "stage3_out_dir": str(stage3_dir),
        "manifest_path": str(manifest),
    }
    return kwargs


@pytest.mark.parametrize(
    ("report_updates", "message"),
    [
        ({"input_sha256": "wrong"}, "did not consume"),
        ({"stage2_model_sha256": "wrong"}, "model provenance differs"),
        ({"molecule_rows": 2}, "account for every"),
    ],
)
def test_runner_rejects_cross_stage_provenance_and_accounting_faults(
    tmp_path, monkeypatch, report_updates, message
):
    kwargs = _canonical_fixture(tmp_path, monkeypatch, report_updates=report_updates)
    with pytest.raises(RuntimeError, match=message):
        runner.run_canonical_pipeline(**kwargs)


def test_runner_rejects_missing_stage3_artifact(tmp_path, monkeypatch):
    kwargs = _canonical_fixture(
        tmp_path, monkeypatch, omit="stage3_site_predictions.csv"
    )
    with pytest.raises(RuntimeError, match="output was not written"):
        runner.run_canonical_pipeline(**kwargs)


def test_reuse_mode_rejects_missing_empty_unmanifested_and_stale_outputs(
    tmp_path, monkeypatch
):
    kwargs = _canonical_fixture(tmp_path, monkeypatch)
    stage2 = kwargs["stage2_output_path"]
    with pytest.raises(FileNotFoundError, match="missing Stage 2"):
        runner.run_canonical_pipeline(**kwargs, reuse_stage2_output=True)

    open(stage2, "w", encoding="utf-8").close()
    with pytest.raises(ValueError, match="empty Stage 2"):
        runner.run_canonical_pipeline(**kwargs, reuse_stage2_output=True)

    pd.DataFrame([{
        "stage2_schema_version": "2.2.0",
        "stage2_free_energy_schema_version": "1.0.0",
    }]).to_csv(stage2, index=False)
    with pytest.raises(FileNotFoundError, match="prior canonical manifest"):
        runner.run_canonical_pipeline(**kwargs, reuse_stage2_output=True)

    with open(kwargs["manifest_path"], "w", encoding="utf-8") as handle:
        json.dump({
            "network_dataset_sha256": "stale",
            "stage2_model_sha256": "stale",
            "stage2_output_sha256": "stale",
        }, handle)
    with pytest.raises(ValueError, match="provenance differs"):
        runner.run_canonical_pipeline(**kwargs, reuse_stage2_output=True)


def test_runner_main_forwards_cli_options_and_prints_manifest(
    tmp_path, monkeypatch, capsys
):
    received = {}

    def fake_run(**kwargs):
        received.update(kwargs)
        return {"pipeline": "synthetic", "population_ph": kwargs["ph"]}

    monkeypatch.setattr(runner, "run_canonical_pipeline", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "run_canonical_protonation_pipeline.py",
        "--network-dataset", "network.csv",
        "--stage2-model", "model.pkl",
        "--stage2-output", "stage2.csv",
        "--stage3-output", "stage3.csv",
        "--stage3-out-dir", "stage3-dir",
        "--manifest", str(tmp_path / "manifest.json"),
        "--ph", "6.8",
        "--stage1-oof", "oof.csv",
        "--stage2-eval", "eval.csv",
        "--stage2-quarantine", "quarantine.csv",
        "--tautomer-score-temperature", "0.75",
        "--max-tautomers-per-configuration", "9",
        "--allow-compatible-network",
        "--reuse-stage2-output",
    ])
    runner.main()
    assert received["ph"] == pytest.approx(6.8)
    assert received["allow_compatible_network"] is True
    assert received["reuse_stage2_output"] is True
    assert received["max_tautomers_per_configuration"] == 9
    assert json.loads(capsys.readouterr().out)["pipeline"] == "synthetic"
