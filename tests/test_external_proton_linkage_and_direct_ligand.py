"""Contracts for PLBD linkage evidence and direct bound-ligand calibration."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import acquire_direct_bound_ligand_structures as acquire_direct
import acquire_plbd_data
import prepare_direct_bound_ligand_calibration as direct
import prepare_plbd_proton_linkage as plbd
import train_protein_environment_model as environment_trainer
from protein_environment_correction import FEATURE_COLUMNS, _prediction_for_site
from functional_group_pka_analysis import expand_pair_site_transitions
from pka_evidence_policy import ionization_policy_role, unresolved_ionizable_contexts
from substructure_match import find_sites_with_metadata, resolve_overlapping_sites
from rdkit import Chem


class _ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


def _write_plbd_archive(path: Path) -> None:
    tables = {
        "compound": [{"attributes": {
            "id": "c1", "name": "ethanol", "label": "EtOH", "smiles": "CCO",
        }}],
        "protein": [{"attributes": {
            "id": "p1", "name": "protein", "label": "P", "uniprot": "P00001",
        }}],
        "best_compound_protonation_pka": [{"attributes": {
            "id": "cp1", "compound_id": {"id": "c1"}, "compound_label": "EtOH",
            "name": "ethanol", "pka": 15.9, "sigpka": 0.1,
            "temperature": 25.0, "pka_sources": "experiment", "reliability": 3,
        }}],
        "best_protein_protonation_pka": [{"attributes": {
            "id": "pp1", "protein_id": {"id": "p1"}, "protein_label": "P",
            "protein_name": "protein", "primary_group_label": "metal-A",
            "pka_zn": 7.1, "sigpka_zn": 0.2, "temperature": 25.0,
            "pka_source": "experiment", "reliability": 3,
        }}],
        "intrinsic_parameters": [{"attributes": {
            "id": "i1", "compound_id": {"id": "c1"},
            "protein_id": {"id": "p1"}, "primary_group_label": "metal-A",
            "temperature": 25.0, "kb_int": 100.0, "dbg_int": -11.4,
            "dbh_int": -8.0, "sigdbg_int": 0.5, "min_q": 2, "n": 3,
        }}],
        "crystal_structure": [{"attributes": {
            "pdb_id": "1abc", "resolution": 1.8, "reliability": 3,
        }}],
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, records in tables.items():
            archive.writestr(
                f"outputs/main/{name}.json", json.dumps({"data": records})
            )


def _pdb_atom(
    record: str,
    serial: int,
    name: str,
    residue: str,
    chain: str,
    residue_id: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"{record:<6}{serial:>5} {name:>4} {residue:>3} {chain}{residue_id:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{20.0:>6.2f}          "
        f"{element:>2}\n"
    )


def _write_direct_registry(path: Path, **updates: object) -> None:
    row = {
        "case_id": "case-1",
        "protein_group": "group-1",
        "protein_name": "test protein",
        "ligand_name": "test ligand",
        "free_pka": 6.0,
        "free_pka_error": "",
        "bound_pka": 7.0,
        "bound_pka_error": 0.1,
        "bound_qualifier": "exact",
        "delta_pka": 1.0,
        "site_family": "heteroaromatic_base",
        "environment_chemistry": "imidazole",
        "transition_semantics": "neutral_to_protonated_N",
        "source_doi": "10.1000/test",
        "source_pmid": "",
        "label_origin": "experimental",
        "computational_teacher": "",
        "coordinate_pdb": "ABCD",
        "ligand_chain": "L",
        "ligand_resname": "LIG",
        "ligand_residue_id": "1",
        "site_atom_name": "N1",
        "structure_relation": "exact_protein_and_ligand",
        "site_mapping_status": "resolved",
        "candidate_for_calibration": True,
        "exclusion_reason": "",
    }
    row.update(updates)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def test_plbd_intrinsic_binding_remains_nontraining_linkage_evidence(tmp_path):
    archive = tmp_path / "plbd.zip"
    output = tmp_path / "prepared"
    _write_plbd_archive(archive)
    report = plbd.prepare(archive, output)
    assert report["proton_linkage_rows"] == 1
    assert report["direct_bound_ligand_calibration_rows"] == 0
    linkage = pd.read_csv(output / "proton_linkage_constraints.csv")
    assert not linkage["direct_bound_ligand_label"].astype(bool).any()
    assert not linkage["eligible_for_direct_environment_fitting"].astype(bool).any()
    assert linkage.loc[0, "evidence_use"] == (
        "proton_linkage_consistency_or_locked_evaluation"
    )


def test_plbd_acquisition_checks_both_pinned_digests(tmp_path, monkeypatch):
    payload = b"synthetic-plbd"
    monkeypatch.setattr(acquire_plbd_data, "EXPECTED_MD5", hashlib.md5(payload).hexdigest())
    monkeypatch.setattr(
        acquire_plbd_data, "EXPECTED_SHA256", hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(
        acquire_plbd_data.urllib.request,
        "urlretrieve",
        lambda _url, destination: Path(destination).write_bytes(payload),
    )
    report = acquire_plbd_data.acquire(tmp_path)
    assert report["sha256"] == hashlib.sha256(payload).hexdigest()
    assert "not direct bound-ligand" in report["evidence_boundary"]


def test_repository_direct_ligand_registry_is_experimental_and_gate_limited():
    table = direct.read_registry()
    assert len(table) == 15
    assert table["label_origin"].eq("experimental").all()
    assert table["computational_teacher"].eq("").all()
    assert int(table["candidate_for_calibration"].map(direct._truthy).sum()) == 7
    assert set(table.loc[
        table["candidate_for_calibration"].map(direct._truthy), "coordinate_pdb"
    ]) == {"3DFR", "2HM9", "1PKW", "1XW5", "5GSS", "1TDI", "1XW6"}
    assert set(table.loc[
        table["candidate_for_calibration"].map(direct._truthy),
        "environment_chemistry",
    ]) == {"imidazole", "thiol"}
    assert table.loc[table["case_id"].eq("DBL015"), "bound_qualifier"].item() == (
        "lower_bound"
    )


def test_direct_ligand_preparation_maps_center_but_keeps_small_gate_closed(tmp_path):
    registry = tmp_path / "registry.csv"
    structures = tmp_path / "structures"
    output = tmp_path / "output"
    structures.mkdir()
    _write_direct_registry(registry)
    (structures / "ABCD.pdb").write_text(
        _pdb_atom("HETATM", 1, "N1", "LIG", "L", 1, 0, 0, 0, "N")
        + _pdb_atom("HETATM", 2, "C2", "LIG", "L", 1, 1, 0, 0, "C")
        + _pdb_atom("ATOM", 3, "OD1", "ASP", "A", 8, 0, 0, 3, "O")
        + "END\n",
        encoding="utf-8",
    )
    report = direct.prepare_and_train(registry, structures, output)
    assert report["strict_structure_mapped_candidate_rows"] == 1
    assert report["protein_groups"] == 1
    assert report["count_and_diversity_gate_passed"] is False
    assert report["deployed_training_rows"] == 0
    assert not (output / "direct_bound_ligand_model.pkl").exists()
    features = pd.read_csv(output / "direct_bound_ligand_features.csv")
    assert features.loc[0, "negative_atoms_6a"] == 1.0


def test_direct_ligand_registry_rejects_teacher_and_inconsistent_delta(tmp_path):
    registry = tmp_path / "registry.csv"
    _write_direct_registry(registry, computational_teacher="QM")
    with pytest.raises(ValueError, match="teacher"):
        direct.read_registry(registry)
    _write_direct_registry(registry, delta_pka=2.0)
    with pytest.raises(ValueError, match="must equal"):
        direct.read_registry(registry)


def test_direct_model_takes_precedence_over_residue_analogue_transfer():
    artifact = {
        "feature_columns": list(FEATURE_COLUMNS),
        "model": _ConstantModel(1.25),
        "oof_p90_abs_error": 0.4,
        "prediction_clip": [-2.0, 2.0],
    }
    prediction = _prediction_for_site(
        {"direct_bound_ligand_model": artifact},
        {column: 0.0 for column in FEATURE_COLUMNS},
        chemistry="imidazole",
        transfer_factor=0.2,
    )
    assert prediction["delta_pka"] == pytest.approx(1.25)
    assert prediction["transfer_shrinkage"] == 1.0
    assert prediction["uses_direct_bound_ligand_calibration"] is True
    assert prediction["prediction_basis"] == "direct_bound_ligand_calibration"


def test_environment_trainer_records_a_closed_direct_ligand_gate(tmp_path):
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({
        "schema_version": environment_trainer.DIRECT_LIGAND_SCHEMA_VERSION,
        "registry_rows": 7,
        "strict_structure_mapped_candidate_rows": 2,
        "deployed": False,
        "deployed_training_rows": 0,
    }), encoding="utf-8")
    gate, artifact = environment_trainer._direct_ligand_calibration(
        gate_path, tmp_path / "model.pkl"
    )
    assert gate["strict_structure_mapped_candidate_rows"] == 2
    assert gate["deployed_training_rows"] == 0
    assert artifact is None


def test_environment_trainer_fails_if_open_gate_model_is_missing(tmp_path):
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({
        "schema_version": environment_trainer.DIRECT_LIGAND_SCHEMA_VERSION,
        "registry_rows": 12,
        "strict_structure_mapped_candidate_rows": 12,
        "deployed": True,
        "deployed_training_rows": 12,
    }), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="model artifact is missing"):
        environment_trainer._direct_ligand_calibration(
            gate_path, tmp_path / "missing.pkl"
        )


def test_environment_trainer_rejects_direct_model_feature_drift(tmp_path):
    gate_path = tmp_path / "gate.json"
    model_path = tmp_path / "model.pkl"
    gate_path.write_text(json.dumps({
        "schema_version": environment_trainer.DIRECT_LIGAND_SCHEMA_VERSION,
        "registry_rows": 12,
        "strict_structure_mapped_candidate_rows": 12,
        "deployed": True,
        "deployed_training_rows": 12,
    }), encoding="utf-8")
    with model_path.open("wb") as handle:
        pickle.dump({
            "schema_version": environment_trainer.DIRECT_LIGAND_SCHEMA_VERSION,
            "feature_columns": ["stale_feature"],
            "training_rows": 12,
        }, handle)
    with pytest.raises(ValueError, match="feature columns are incompatible"):
        environment_trainer._direct_ligand_calibration(gate_path, model_path)


def test_direct_structure_acquisition_fails_closed_on_changed_bytes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(acquire_direct, "STRUCTURES", {
        "ABCD": {"sha256": "0" * 64, "bytes": 3, "role": "test"}
    })
    monkeypatch.setattr(
        acquire_direct.urllib.request,
        "urlretrieve",
        lambda _url, destination: Path(destination).write_bytes(b"bad"),
    )
    with pytest.raises(ValueError, match="checksum/size mismatch"):
        acquire_direct.acquire(tmp_path)


@pytest.mark.parametrize("smiles", [
    "c1ccc2[nH]ccc2c1",
    "Cn1c2ccccc2cc1",
    "O=C(O)Cn1c(C(=O)O)c(Cc2cccc3ccccc23)c2ccccc21",
])
def test_plain_indole_is_context_not_an_unresolved_active_range_site(smiles):
    mol = Chem.MolFromSmiles(smiles)
    resolved, _ = resolve_overlapping_sites(find_sites_with_metadata(mol), 0.75)
    pairs = [
        transition
        for site in resolved
        for transition in expand_pair_site_transitions(site)
    ]
    unresolved = unresolved_ionizable_contexts(resolved, pairs)
    assert ionization_policy_role("indole") == "extreme_range_disabled"
    assert "indole" not in {item["label"] for item in unresolved}
