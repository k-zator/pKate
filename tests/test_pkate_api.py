"""Fast contract tests for the public unseen-molecule API."""

from pathlib import Path

import pytest
from rdkit import Chem

import pkate


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


@pytest.mark.parametrize("ph", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_ph_is_rejected(ph):
    with pytest.raises(ValueError, match="ph must be finite"):
        pkate.predict("CCC", ph=ph)


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
