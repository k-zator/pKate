"""Fast contract tests for the public unseen-molecule API."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
import sklearn

import pkate


def test_public_version_matches_release_manifest():
    project_root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (project_root / "models" / "MODEL_ARTIFACT_MANIFEST.json").read_text()
    )

    assert pkate.__version__ == "0.2.0"
    assert manifest["release_version"] == pkate.__version__


def test_release_runtime_matches_serialized_model_contract():
    """Keep clean-clone installs compatible with the fitted release pickles."""
    project_root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (project_root / "models" / "MODEL_ARTIFACT_MANIFEST.json").read_text()
    )
    required_version = str(manifest["runtime"]["scikit_learn"])

    requirements = (project_root / "requirements.txt").read_text().splitlines()
    environment = (project_root / "environment.yml").read_text().splitlines()

    assert f"scikit-learn=={required_version}" in requirements
    assert f"  - scikit-learn={required_version}" in environment
    assert sklearn.__version__ == required_version


def _artifacts(tmp_path: Path) -> pkate.ModelArtifacts:
    """Create the six deployment paths with tiny deterministic contents."""
    artifacts = pkate.ModelArtifacts(
        stage1_model=tmp_path / "stage1.pkl",
        stage1_metrics=tmp_path / "stage1.json",
        stage1_oof=tmp_path / "stage1_oof.csv",
        stage2_model=tmp_path / "stage2.pkl",
        stage2_eval=tmp_path / "stage2_eval.csv",
        stage2_quarantine=tmp_path / "stage2_quarantine.csv",
    )
    artifacts.stage1_model.write_bytes(b"stage1-model")
    artifacts.stage2_model.write_bytes(b"stage2-model")
    artifacts.stage1_metrics.write_text(json.dumps({"scaffold": {"mae": 1.25}}))
    artifacts.stage1_oof.write_text("actual,predicted\n7,7\n")
    artifacts.stage2_eval.write_text("actual,predicted\n7,7\n")
    artifacts.stage2_quarantine.write_text("molecule_id\n")
    return artifacts


def _synthetic_network(*, truncated: bool = False) -> dict:
    return {
        "sites": [{
            "site_id": "site_1",
            "family": "amine",
            "label": "primary_amine",
            "center_maps": [2],
        }],
        "nodes": [{"node_id": "node_0"}, {"node_id": "node_1"}],
        "edges": [{"site_id": "site_1", "from_node_id": "node_0", "to_node_id": "node_1"}],
        "protonation_state_enumeration_truncated": truncated,
        "incomplete_site_ids": ["site_1"] if truncated else [],
        "max_protonation_states": 64,
    }


def _install_fake_public_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict,
    *,
    empty_tautomers: bool = False,
) -> None:
    """Replace expensive stages while preserving the public filesystem contract."""
    molecule_id = "query_fake"

    def fake_build(**kwargs):
        captured["build"] = kwargs
        return pd.DataFrame([{"molecule_id": molecule_id, "network": "stage1"}])

    def fake_stage2(**kwargs):
        captured["stage2"] = kwargs
        frame = pd.read_csv(kwargs["network_dataset_path"])
        frame["network"] = "stage2"
        frame.to_csv(kwargs["output_path"], index=False)
        Path(kwargs["report_path"]).write_text('{"status": "ok"}')
        return frame, {"status": "ok"}

    def fake_stage3(**kwargs):
        captured["stage3"] = kwargs
        assert pd.read_csv(kwargs["input_path"]).loc[0, "network"] == "stage2"
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        molecule_table = pd.DataFrame([{
            "molecule_id": molecule_id,
            "stage3_population_status": "complete_pairwise_coupled_population",
            "stage3_dominant_atom_mapped_smiles": "[CH3:1][NH3+:2]",
            "stage3_dominant_configuration_node_id": "node_1",
            "stage3_dominant_formal_charge": 1,
            "stage3_dominant_population": 0.9,
            "stage3_dominant_complete_microstate_heuristic_population": 0.8,
        }])
        molecule_table.to_csv(kwargs["output_path"], index=False)
        pd.DataFrame([{
            "molecule_id": molecule_id,
            "site_id": "site_1",
            "site_label": "primary_amine",
            "site_family": "amine",
            "site_atom_maps_json": "[2]",
            "site_center_maps_json": "[2]",
            "stage3_predicted_site_form_at_ph": "acid_form",
            "stage3_protonated_probability_at_ph": 0.9,
            "stage3_protonated_probability_ci_low": 0.7,
            "stage3_protonated_probability_ci_high": 0.98,
            "stage2_predicted_macro_pka": 9.6,
            "stage3_empirical_macro_pka_interval_90_low": 8.7,
            "stage3_empirical_macro_pka_interval_90_high": 10.4,
            "stage3_contextual_edge_pka_at_dominant_background": 9.4,
            "stage3_overall_site_state_confidence": 0.85,
            "stage3_overall_site_state_confidence_tier": "high",
            "stage1_applicability_domain": "interpolation_high_similarity",
            "stage1_exact_label_training_rows": 50,
            "stage2_applied": "False",
            "stage2_prediction_confidence": "stage1_fallback",
        }]).to_csv(out_dir / "stage3_site_predictions.csv", index=False)
        tautomer_path = out_dir / "stage3_ranked_tautomers.csv"
        if empty_tautomers:
            tautomer_path.write_text("")
        else:
            pd.DataFrame([
                {
                    "molecule_id": molecule_id,
                    "origin_node_id": "node_1",
                    "atom_mapped_smiles": "[CH3:1][NH3+:2]",
                    "tautomer_rank_within_configuration": 2,
                    "tautomer_heuristic_conditional_weight": 0.2,
                    "complete_microstate_heuristic_population_at_ph": 0.18,
                    "rdkit_tautomer_score": 1.0,
                    "ranking_semantics": "heuristic",
                },
                {
                    "molecule_id": molecule_id,
                    "origin_node_id": "node_1",
                    "atom_mapped_smiles": "[CH3:1][NH3+:2]",
                    "tautomer_rank_within_configuration": 1,
                    "tautomer_heuristic_conditional_weight": 0.8,
                    "complete_microstate_heuristic_population_at_ph": 0.72,
                    "rdkit_tautomer_score": 2.0,
                    "ranking_semantics": "heuristic",
                },
                {
                    "molecule_id": molecule_id,
                    "origin_node_id": "other_node",
                    "atom_mapped_smiles": "[CH3:1][NH2:2]",
                    "tautomer_rank_within_configuration": 1,
                },
            ]).to_csv(tautomer_path, index=False)
        return molecule_table, pd.DataFrame()

    monkeypatch.setattr(pkate, "_build_query_network", fake_build)
    monkeypatch.setattr(pkate, "apply_stage2", fake_stage2)
    monkeypatch.setattr(pkate, "apply_stage3", fake_stage3)


def test_nonionizable_molecule_returns_unchanged_without_models(monkeypatch):
    def fail_if_models_are_requested(*args, **kwargs):
        raise AssertionError("model artifacts should not be loaded for propane")

    monkeypatch.setattr(pkate.ModelArtifacts, "defaults", fail_if_models_are_requested)
    result = pkate.predict("CCC", ph=7.4)

    assert result.status == "no_supported_ionizable_sites_detected"
    assert result.dominant_microstate_smiles == "CCC"
    assert result.dominant_formal_charge == 0
    assert result.dominant_configuration_probability == pytest.approx(1.0)
    assert result.site_predictions == ()
    assert result.provenance["stage1_applied"] is False


def test_invalid_smiles_is_rejected():
    with pytest.raises(ValueError, match="RDKit could not parse"):
        pkate.predict("this is not smiles")


@pytest.mark.parametrize("smiles", [None, "", "   ", 42])
def test_empty_or_nonstring_smiles_is_rejected(smiles):
    with pytest.raises(ValueError, match="non-empty string"):
        pkate.predict(smiles)  # type: ignore[arg-type]


@pytest.mark.parametrize("ph", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_ph_is_rejected(ph):
    with pytest.raises(ValueError, match="ph must be finite"):
        pkate.predict("CCC", ph=ph)


def test_nonnumeric_ph_has_stable_public_error():
    with pytest.raises(ValueError, match="ph must be a finite number"):
        pkate.predict("CCC", ph=None)  # type: ignore[arg-type]


@pytest.mark.parametrize("overlap", [-0.01, 1.01, float("nan"), "bad"])
def test_overlap_threshold_is_validated(overlap):
    with pytest.raises(ValueError, match="overlap_threshold"):
        pkate.predict("CCC", overlap_threshold=overlap)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("max_protonation_states", 0),
        ("max_protonation_states", 1.5),
        ("max_tautomers_per_state", 0),
        ("max_tautomers_per_configuration", False),
        ("top_tautomers", -1),
        ("top_tautomers", 2.5),
    ],
)
def test_integer_caps_reject_invalid_or_fractional_values(argument, value):
    with pytest.raises(ValueError, match=argument):
        pkate.predict("CCC", **{argument: value})


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), "bad"])
def test_tautomer_temperature_must_be_positive_and_finite(temperature):
    with pytest.raises(ValueError, match="tautomer_score_temperature"):
        pkate.predict("CCC", tautomer_score_temperature=temperature)


def test_ionizable_query_reports_all_missing_artifacts(tmp_path):
    artifacts = pkate.ModelArtifacts(
        stage1_model=tmp_path / "stage1.pkl",
        stage1_metrics=tmp_path / "stage1.json",
        stage1_oof=tmp_path / "stage1_oof.csv",
        stage2_model=tmp_path / "stage2.pkl",
        stage2_eval=tmp_path / "stage2_eval.csv",
        stage2_quarantine=tmp_path / "stage2_quarantine.csv",
    )
    with pytest.raises(FileNotFoundError, match="Missing") as error:
        pkate.predict("CC(=O)O", artifacts=artifacts)

    message = str(error.value)
    for path in artifacts.__dict__.values():
        assert str(path) in message


def test_result_dictionary_is_json_ready():
    result = pkate.predict("CCC")
    payload = result.to_dict()

    assert payload["canonical_input_smiles"] == "CCC"
    assert payload["site_predictions"] == ()
    assert payload["provenance"]["marvin_values_used"] is False


def test_unmapped_smiles_removes_atom_map_numbers():
    smiles = pkate._unmapped_smiles("[NH3+:1][CH2:2][C:3](=[O:4])[O-:5]")
    molecule = Chem.MolFromSmiles(smiles)

    assert molecule is not None
    assert all(atom.GetAtomMapNum() == 0 for atom in molecule.GetAtoms())


def test_default_artifacts_are_root_relative(tmp_path):
    artifacts = pkate.ModelArtifacts.defaults(tmp_path)

    assert artifacts.stage1_model == (
        Path(tmp_path)
        / "data/processed/ml_models_experimental_only/stage1_intrinsic/"
        "stage1_intrinsic_model.pkl"
    )
    assert artifacts.stage2_model == (
        Path(tmp_path)
        / "data/processed/ml_models_experimental_only/stage2_network_context/"
        "stage2_network_context_model.pkl"
    )


def test_complete_artifact_set_validates(tmp_path):
    _artifacts(tmp_path).validate()


def test_model_hash_is_content_based(tmp_path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"known model")
    assert pkate._sha256(path) == hashlib.sha256(b"known model").hexdigest()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("bad", None),
        (float("nan"), None),
        ("3.5", 3.5),
    ],
)
def test_optional_float_normalizes_nonfinite_and_malformed_values(value, expected):
    assert pkate._optional_float(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, True), (False, False), ("true", True), ("False", False), (1, True), (0, False), (None, False), ("unknown", False)],
)
def test_optional_bool_handles_csv_values(value, expected):
    assert pkate._optional_bool(value) is expected


def test_json_list_parser_rejects_nonlist_and_malformed_values():
    assert pkate._parse_json_list("[1, 2]") == [1, 2]
    assert pkate._parse_json_list('{"x": 1}') == []
    assert pkate._parse_json_list("bad") == []


def test_molecule_group_key_falls_back_when_inchi_fails(monkeypatch):
    molecule = Chem.MolFromSmiles("CC")
    assert molecule is not None
    monkeypatch.setattr(pkate.inchi, "MolToInchiKey", lambda _mol: (_ for _ in ()).throw(RuntimeError("no inchi")))
    assert pkate._molecule_group_key(molecule, "CC").startswith("smiles_")


def test_site_detection_expands_an_acid_to_a_transition():
    molecule = Chem.MolFromSmiles("CC(=O)O")
    assert molecule is not None
    resolved, transitions = pkate._detect_sites(molecule, overlap_threshold=0.5)
    assert resolved
    assert any(site["family"] == "carboxyl" for site in transitions)


@pytest.mark.parametrize(
    ("form", "expected_state"),
    [
        ("acid_form", "protonated"),
        ("base_form", "deprotonated"),
        ("unavailable", "unavailable"),
    ],
)
def test_public_site_payload_preserves_state_call_and_confidence(form, expected_state):
    payload = pkate._site_payload(pd.Series({
        "site_id": "site_1",
        "site_label": "primary_amine",
        "site_family": "amine",
        "site_atom_maps_json": "[2]",
        "site_center_maps_json": "[2]",
        "stage3_predicted_site_form_at_ph": form,
        "stage3_protonated_probability_at_ph": 0.5,
        "stage3_protonated_probability_ci_low": 0.2,
        "stage3_protonated_probability_ci_high": 0.8,
        "stage2_predicted_macro_pka": 7.4,
        "stage3_empirical_macro_pka_interval_90_low": 6.7,
        "stage3_empirical_macro_pka_interval_90_high": 8.1,
        "stage3_contextual_edge_pka_at_dominant_background": 7.4,
        "stage3_overall_site_state_confidence": 0.0,
        "stage3_overall_site_state_confidence_tier": "very_low",
        "stage1_applicability_domain": "interpolation_high_similarity",
        "stage1_exact_label_training_rows": 42,
        "stage2_applied": False,
        "stage2_prediction_confidence": "stage1_fallback",
    }))

    assert payload["predicted_form"] == form
    assert payload["predicted_protonation_state"] == expected_state
    assert payload["protonated_probability"] == pytest.approx(0.5)
    assert payload["protonated_probability_interval"] == [0.2, 0.8]
    assert payload["overall_confidence"] == 0.0
    assert payload["confidence_tier"] == "very_low"


def test_public_site_payload_normalizes_missing_strings_and_false_csv_boolean():
    payload = pkate._site_payload(pd.Series({
        "site_id": "site_1",
        "site_label": "amine",
        "site_family": "amine",
        "stage3_predicted_site_form_at_ph": np.nan,
        "stage3_overall_site_state_confidence_tier": np.nan,
        "stage2_applied": "False",
    }))
    assert payload["predicted_form"] == "unavailable"
    assert payload["predicted_protonation_state"] == "unavailable"
    assert payload["confidence_tier"] == "unavailable"
    assert payload["stage2_applied"] is False


def test_public_site_payload_exposes_explicit_protein_residue_identity():
    payload = pkate._site_payload(pd.Series({
        "site_id": "protein_A_HIS_57",
        "component_type": "protein_residue",
        "site_label": "protein_his",
        "site_family": "protein_residue",
        "protein_chain": "A",
        "protein_residue_name": "HIS",
        "protein_residue_id": "57",
        "protein_insertion_code": "B",
        "protein_nearest_ligand_site_distance_angstrom": 3.2,
        "stage3_predicted_site_form_at_ph": "acid_form",
        "stage2_applied": True,
    }))
    assert payload["component_type"] == "protein_residue"
    assert payload["protein_residue"] == {
        "chain": "A",
        "name": "HIS",
        "number": "57",
        "insertion_code": "B",
        "nearest_ligand_site_distance_angstrom": 3.2,
    }


def test_public_result_warns_about_low_confidence_boundary_call(tmp_path):
    stage1_model = tmp_path / "stage1.pkl"
    stage2_model = tmp_path / "stage2.pkl"
    stage1_model.write_bytes(b"stage1")
    stage2_model.write_bytes(b"stage2")
    artifacts = pkate.ModelArtifacts(
        stage1_model=stage1_model,
        stage1_metrics=tmp_path / "stage1.json",
        stage1_oof=tmp_path / "stage1_oof.csv",
        stage2_model=stage2_model,
        stage2_eval=tmp_path / "stage2_eval.csv",
        stage2_quarantine=tmp_path / "stage2_quarantine.csv",
    )
    molecule_table = pd.DataFrame([{
        "molecule_id": "query_boundary",
        "stage3_population_status": "complete_pairwise_coupled_population",
        "stage3_dominant_atom_mapped_smiles": "[CH3:1][NH3+:2]",
        "stage3_dominant_configuration_node_id": "acid",
        "stage3_dominant_formal_charge": 1,
        "stage3_dominant_protein_site_forms_json": json.dumps({
            "protein_A_ASP_42": "base_form"
        }),
        "stage3_dominant_population": 0.5,
        "stage3_dominant_complete_microstate_heuristic_population": 0.5,
    }])
    site_table = pd.DataFrame([{
        "molecule_id": "query_boundary",
        "site_id": "site_1",
        "site_label": "primary_amine",
        "site_family": "amine",
        "site_atom_maps_json": "[2]",
        "site_center_maps_json": "[2]",
        "stage3_predicted_site_form_at_ph": "acid_form",
        "stage3_protonated_probability_at_ph": 0.5,
        "stage3_overall_site_state_confidence": 0.0,
        "stage3_overall_site_state_confidence_tier": "very_low",
    }])
    tautomer_table = pd.DataFrame(columns=[
        "molecule_id",
        "origin_node_id",
        "tautomer_rank_within_configuration",
    ])

    result = pkate._result_from_outputs(
        input_smiles="CN",
        canonical_smiles="CN",
        ph=7.4,
        molecule_table=molecule_table,
        site_table=site_table,
        tautomer_table=tautomer_table,
        artifacts=artifacts,
        persistent_output_dir=None,
        unresolved_contexts=[],
        top_tautomers=3,
    )

    assert result.site_predictions[0]["predicted_protonation_state"] == "protonated"
    assert result.site_predictions[0]["confidence_tier"] == "very_low"
    assert result.dominant_protein_site_forms == {
        "protein_A_ASP_42": "base_form"
    }
    assert "Low-confidence site calls: site_1" in result.warnings


def test_result_filters_and_ranks_only_dominant_configuration_tautomers(tmp_path):
    artifacts = _artifacts(tmp_path)
    molecule_table = pd.DataFrame([{
        "molecule_id": "query_1",
        "stage3_population_status": "incomplete_network",
        "stage3_dominant_atom_mapped_smiles": None,
        "stage3_dominant_configuration_node_id": "node_a",
    }])
    site_table = pd.DataFrame([{
        "molecule_id": "query_1",
        "site_id": "site_2",
        "site_label": "carboxylic_acid",
        "site_family": "carboxyl",
        "stage3_overall_site_state_confidence_tier": "medium",
    }])
    tautomer_table = pd.DataFrame([
        {"molecule_id": "query_1", "origin_node_id": "node_a", "atom_mapped_smiles": "[CH3:1][OH:2]", "tautomer_rank_within_configuration": 2},
        {"molecule_id": "query_1", "origin_node_id": "node_a", "atom_mapped_smiles": "[CH3:1][OH:2]", "tautomer_rank_within_configuration": 1},
        {"molecule_id": "query_1", "origin_node_id": "node_b", "atom_mapped_smiles": "[CH3:1][O-:2]", "tautomer_rank_within_configuration": 1},
        {"molecule_id": "other", "origin_node_id": "node_a", "atom_mapped_smiles": "[CH3:1][OH:2]", "tautomer_rank_within_configuration": 1},
    ])
    result = pkate._result_from_outputs(
        input_smiles="CO",
        canonical_smiles="CO",
        ph=7.4,
        molecule_table=molecule_table,
        site_table=site_table,
        tautomer_table=tautomer_table,
        artifacts=artifacts,
        persistent_output_dir=tmp_path / "output",
        unresolved_contexts=[{"label": "unknown"}],
        top_tautomers=1,
    )
    assert result.dominant_microstate_smiles is None
    assert [tautomer["rank"] for tautomer in result.ranked_tautomers] == [1]
    assert len(result.warnings) == 2
    assert "output_directory" in result.provenance


@pytest.mark.parametrize(
    ("molecule_table", "site_table", "tautomer_table", "message"),
    [
        (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "no molecule-level"),
        (pd.DataFrame([{"wrong": 1}]), pd.DataFrame(), pd.DataFrame(), "molecule_id"),
        (pd.DataFrame([{"molecule_id": "a"}, {"molecule_id": "b"}]), pd.DataFrame(), pd.DataFrame(), "exactly one"),
        (pd.DataFrame([{"molecule_id": "a"}]), pd.DataFrame([{"molecule_id": "a"}]), pd.DataFrame(), "site output"),
        (
            pd.DataFrame([{"molecule_id": "a"}]),
            pd.DataFrame([{"molecule_id": "a", "site_id": "s", "site_label": "x", "site_family": "x"}]),
            pd.DataFrame([{"molecule_id": "a"}]),
            "tautomer output",
        ),
    ],
)
def test_result_rejects_malformed_stage3_tables(
    tmp_path,
    molecule_table,
    site_table,
    tautomer_table,
    message,
):
    with pytest.raises(RuntimeError, match=message):
        pkate._result_from_outputs(
            input_smiles="CN",
            canonical_smiles="CN",
            ph=7.4,
            molecule_table=molecule_table,
            site_table=site_table,
            tautomer_table=tautomer_table,
            artifacts=_artifacts(tmp_path),
            persistent_output_dir=None,
            unresolved_contexts=[],
            top_tautomers=1,
        )


@pytest.mark.parametrize(
    ("domain", "p90", "p95", "expected_confidence", "expected_half_width"),
    [
        ("interpolation_high_similarity", 1.5, 1.8, "high_empirical_support", 1.8),
        ("interpolation_limited_support", 2.5, 2.8, "medium_empirical_support", 2.8),
        ("zero_shot_label", 3.0, 4.0, "low_empirical_support", 8.0),
    ],
)
def test_query_network_builds_stage1_tasks_confidence_and_macro_steps(
    tmp_path,
    monkeypatch,
    domain,
    p90,
    p95,
    expected_confidence,
    expected_half_width,
):
    artifacts = _artifacts(tmp_path)
    captured = {}
    bundle = {
        "functional_group_detector_schema_versions": [pkate.FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION],
        "feature_columns": ["group_primary_amine"],
        "error_calibration_global": {"rows": 12, "mae": 1.0, "p90_abs_error": p90, "p95_abs_error": p95},
        "stage1_schema_version": "test-stage1",
        "target_definition": "intrinsic transition pKa",
        "training_measurement_method": "experimental",
        "training_rows": 100,
        "training_measurement_count": 80,
        "training_dataset_sha256": "dataset-hash",
        "trained_on_all_eligible_rows": True,
    }

    monkeypatch.setattr(pkate, "load_stage1_bundle", lambda *_args, **_kwargs: bundle)
    monkeypatch.setattr(pkate, "enumerate_joint_protonation_network", lambda *_args, **_kwargs: _synthetic_network())

    def fake_predict(tasks, *_args, **_kwargs):
        captured["tasks"] = tasks
        return {tasks[0]["task_id"]: 9.5}

    monkeypatch.setattr(pkate, "predict_stage1_tasks", fake_predict)
    monkeypatch.setattr(
        pkate,
        "stage1_applicability",
        lambda *_args: {
            "stage1_applicability_domain": domain,
            "stage1_exact_label_training_rows": 42,
        },
    )
    monkeypatch.setattr(
        pkate,
        "stage1_local_prediction_for_site",
        lambda _site, prediction, _domain: (prediction, "stage1_model", "", None),
    )
    monkeypatch.setattr(
        pkate,
        "coupled_network_thermodynamics",
        lambda *_args: (
            [{"protonation_step": 1, "predicted_macro_pka": 9.2}],
            {"site_1": {"protonation_step": 1, "predicted_macro_pka": 9.2}},
        ),
    )
    monkeypatch.setattr(pkate, "unresolved_ionizable_contexts", lambda *_args: [])
    molecule = Chem.MolFromSmiles("CN")
    assert molecule is not None
    frame = pkate._build_query_network(
        input_smiles="CN",
        canonical_smiles="CN",
        molecule=molecule,
        resolved_sites=[],
        transition_sites=[{"label": "primary_amine"}],
        ph=7.4,
        artifacts=artifacts,
        overlap_threshold=0.5,
        max_protonation_states=64,
        max_tautomers_per_state=4,
    )
    assert captured["tasks"][0]["candidate_label"] == "primary_amine"
    assert captured["tasks"][0]["pka_type_canonical"] == "basic"
    row = frame.iloc[0]
    site = json.loads(row["sites_json"])[0]
    edge = json.loads(row["microstate_edges_json"])[0]
    assert site["local_pka_confidence"] == expected_confidence
    assert site["stage1_predicted_pka_ci_low"] == pytest.approx(9.5 - expected_half_width)
    assert site["rank_associated_macro_pka"] == pytest.approx(9.2)
    assert edge["local_pka_used"] == pytest.approx(9.5)
    assert row["network_confidence"] == "stage1_only_unseen_molecule"
    assert not row["marvin_values_used"] and not row["epik_values_used"]


def test_query_network_rejects_detector_model_schema_mismatch(tmp_path, monkeypatch):
    artifacts = _artifacts(tmp_path)
    monkeypatch.setattr(
        pkate,
        "load_stage1_bundle",
        lambda *_args, **_kwargs: {
            "functional_group_detector_schema_versions": ["incompatible"],
        },
    )
    molecule = Chem.MolFromSmiles("CN")
    assert molecule is not None
    with pytest.raises(ValueError, match="incompatible"):
        pkate._build_query_network(
            input_smiles="CN",
            canonical_smiles="CN",
            molecule=molecule,
            resolved_sites=[],
            transition_sites=[],
            ph=7.4,
            artifacts=artifacts,
            overlap_threshold=0.5,
            max_protonation_states=64,
            max_tautomers_per_state=4,
        )


def test_query_network_rejects_empty_enumeration(tmp_path, monkeypatch):
    artifacts = _artifacts(tmp_path)
    monkeypatch.setattr(
        pkate,
        "load_stage1_bundle",
        lambda *_args, **_kwargs: {"feature_columns": []},
    )
    monkeypatch.setattr(pkate, "enumerate_joint_protonation_network", lambda *_args, **_kwargs: {"sites": [], "nodes": []})
    molecule = Chem.MolFromSmiles("CN")
    assert molecule is not None
    with pytest.raises(RuntimeError, match="could not be enumerated"):
        pkate._build_query_network(
            input_smiles="CN", canonical_smiles="CN", molecule=molecule,
            resolved_sites=[], transition_sites=[], ph=7.4, artifacts=artifacts,
            overlap_threshold=0.5, max_protonation_states=64,
            max_tautomers_per_state=4,
        )


def test_predict_runs_public_stage_handoff_and_writes_audit_outputs(
    tmp_path,
    monkeypatch,
):
    artifacts = _artifacts(tmp_path)
    output_dir = tmp_path / "prediction"
    captured = {}
    _install_fake_public_pipeline(monkeypatch, captured)

    result = pkate.predict(
        "CN",
        ph=6.5,
        artifacts=artifacts,
        output_dir=output_dir,
        max_protonation_states="8",  # type: ignore[arg-type]
        max_tautomers_per_state=3,
        max_tautomers_per_configuration=2,
        tautomer_score_temperature=0.75,
        top_tautomers=1,
    )

    assert result.status == "complete_pairwise_coupled_population"
    assert result.ph == pytest.approx(6.5)
    assert result.dominant_formal_charge == 1
    assert result.site_predictions[0]["predicted_protonation_state"] == "protonated"
    assert result.site_predictions[0]["stage2_applied"] is False
    assert [row["rank"] for row in result.ranked_tautomers] == [1]
    assert captured["build"]["max_protonation_states"] == 8
    assert captured["stage2"]["allow_compatible_network"] is True
    assert captured["stage3"]["ph"] == pytest.approx(6.5)
    assert captured["stage3"]["tautomer_score_temperature"] == pytest.approx(0.75)
    assert captured["stage3"]["max_tautomers_per_configuration"] == 2
    assert result.provenance["stage1_model_sha256"] == pkate._sha256(artifacts.stage1_model)
    assert (output_dir / "query_stage1_network.csv").is_file()
    assert (output_dir / "query_stage2_network.csv").is_file()
    assert (output_dir / "stage3/stage3_site_predictions.csv").is_file()
    saved = json.loads((output_dir / "prediction.json").read_text())
    assert saved["molecule_id"] == "query_fake"
    assert saved["ranked_tautomers"][0]["rank"] == 1


def test_predict_handles_empty_tautomer_output_and_cleans_temporary_directory(
    tmp_path,
    monkeypatch,
):
    artifacts = _artifacts(tmp_path)
    captured = {}
    _install_fake_public_pipeline(monkeypatch, captured, empty_tautomers=True)
    result = pkate.predict("CN", artifacts=artifacts)
    temporary_work_dir = Path(captured["stage2"]["output_path"]).parent
    assert result.ranked_tautomers == ()
    assert not temporary_work_dir.exists()


def test_predict_reports_missing_site_output_as_pipeline_error(tmp_path, monkeypatch):
    artifacts = _artifacts(tmp_path)
    captured = {}
    _install_fake_public_pipeline(monkeypatch, captured)

    original_stage3 = pkate.apply_stage3

    def stage3_without_site_output(**kwargs):
        result = original_stage3(**kwargs)
        (Path(kwargs["out_dir"]) / "stage3_site_predictions.csv").unlink()
        return result

    monkeypatch.setattr(pkate, "apply_stage3", stage3_without_site_output)
    with pytest.raises(RuntimeError, match="readable site-prediction"):
        pkate.predict("CN", artifacts=artifacts)


def test_complex_coordinates_are_required_as_a_pair():
    with pytest.raises(ValueError, match="must be supplied together"):
        pkate.predict("CN", protein_structure="protein.cif")


def test_predict_inserts_environment_layer_between_stage2_and_stage3(
    tmp_path, monkeypatch
):
    artifacts = _artifacts(tmp_path)
    captured = {}
    _install_fake_public_pipeline(monkeypatch, captured)

    def fake_environment(input_path, output_path, **kwargs):
        captured["environment"] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            **kwargs,
        }
        frame = pd.read_csv(input_path)
        frame.to_csv(output_path, index=False)
        return frame, {
            "method": "synthetic_environment",
            "applied_site_count": 1,
        }

    monkeypatch.setattr(pkate, "apply_protein_environment_file", fake_environment)
    result = pkate.predict(
        "CN",
        artifacts=artifacts,
        protein_structure=tmp_path / "complex.cif",
        ligand_structure=tmp_path / "ligand.sdf",
        protein_environment_model=tmp_path / "environment.pkl",
        ligand_chain="L",
        ligand_resname="LIG",
        ligand_resid="7",
    )

    assert Path(captured["environment"]["input_path"]).name == "query_stage2_network.csv"
    assert Path(captured["stage3"]["input_path"]).name == (
        "query_stage2_protein_environment.csv"
    )
    selector = captured["environment"]["ligand_selector"]
    assert (selector.chain, selector.residue_name, selector.residue_id) == (
        "L", "LIG", "7"
    )
    assert captured["environment"]["joint_protein_sites"] is True
    assert captured["environment"]["joint_protein_radius"] == pytest.approx(8.0)
    assert captured["environment"]["maximum_joint_protein_sites"] == 6
    assert captured["environment"]["maximum_joint_nodes"] == 4096
    assert result.provenance["protein_environment"]["applied_site_count"] == 1
    assert any("protein-residue" in warning for warning in result.warnings)


def test_cli_forwards_user_arguments_and_emits_strict_json(monkeypatch, capsys):
    captured = {}
    molecule = Chem.MolFromSmiles("CCC")
    assert molecule is not None
    result = pkate._no_site_result("CCC", "CCC", molecule, 6.2)

    def fake_predict(smiles, **kwargs):
        captured["smiles"] = smiles
        captured.update(kwargs)
        return result

    monkeypatch.setattr(pkate, "predict", fake_predict)
    assert pkate.main(["CCC", "--ph", "6.2", "--top-tautomers", "0", "--compact"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert captured == {"smiles": "CCC", "ph": 6.2, "output_dir": None, "top_tautomers": 0}
    assert payload["ph"] == pytest.approx(6.2)
    assert payload["status"] == "no_supported_ionizable_sites_detected"


def test_cli_forwards_joint_protein_network_caps(monkeypatch, capsys):
    captured = {}
    molecule = Chem.MolFromSmiles("CCC")
    assert molecule is not None
    result = pkate._no_site_result("CCC", "CCC", molecule, 7.4)

    def fake_predict(smiles, **kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(pkate, "predict", fake_predict)
    assert pkate.main([
        "CCC",
        "--protein", "complex.cif",
        "--ligand", "ligand.sdf",
        "--no-joint-protein-sites",
        "--joint-protein-radius", "6.5",
        "--max-joint-protein-sites", "4",
        "--max-joint-nodes", "512",
        "--compact",
    ]) == 0
    json.loads(capsys.readouterr().out)
    assert captured["joint_protein_sites"] is False
    assert captured["joint_protein_radius"] == pytest.approx(6.5)
    assert captured["max_joint_protein_sites"] == 4
    assert captured["max_joint_nodes"] == 512


def test_calculate_pkas_is_public_predict_alias():
    assert pkate.calculate_pkas is pkate.predict


def test_cli_reports_release_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        pkate.main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip().endswith(" 0.2.0")
