import hashlib
import json

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
