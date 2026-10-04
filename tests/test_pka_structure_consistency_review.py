import pandas as pd
import pytest
from rdkit import Chem

from build_release_microstate_dataset import build_dataset
from pka_structure_consistency_review import (
    _load_decisions,
    _save_decision,
    validate_replacement_smiles,
)


def _assignment_row():
    return {
        "source_file": "toy.sdf",
        "record_index": 0,
        "smiles": "CC(=O)O",
        "pka_value": 4.76,
        "pka_source_method": "experimental",
        "pka_type_raw": "acidic",
        "pka_type_canonical": "acidic",
        "atom_index_raw": None,
        "source_exact_duplicate_record_indices": "0",
    }


def _builder_kwargs(tmp_path, decisions_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    writer = Chem.SDWriter(str(raw_dir / "toy.sdf"))
    writer.write(Chem.MolFromSmiles("CC(=O)O"))
    writer.close()
    assignments = tmp_path / "assignments.csv"
    pd.DataFrame([_assignment_row()]).to_csv(assignments, index=False)
    return {
        "raw_dir": str(raw_dir),
        "assignments_path": str(assignments),
        "override_paths": [],
        "output_path": str(tmp_path / "dataset.csv"),
        "quarantine_path": str(tmp_path / "quarantine.csv"),
        "clean_overrides_path": str(tmp_path / "clean.csv"),
        "override_quarantine_path": str(tmp_path / "override_q.csv"),
        "outlier_review_decisions_path": None,
        "structure_review_decisions_path": str(decisions_path),
        "max_tautomers": 4,
    }


def test_replacement_validation_allows_protonation_but_not_new_skeleton():
    assert validate_replacement_smiles("CC(=O)O", "CC(=O)[O-]") == "CC(=O)[O-]"
    with pytest.raises(ValueError, match="heavy-atom"):
        validate_replacement_smiles("CC(=O)O", "CCC(=O)O")


def test_decision_is_persistent_and_replacement_overlay_is_applied(tmp_path):
    decisions_path = tmp_path / "decisions.csv"
    review_row = pd.Series({
        "review_id": "review_1",
        "source_file": "toy.sdf",
        "record_index": 0,
        "pka_value": 4.76,
        "molecule_id": "mol_1",
        "site_id": "site_1",
        "original_smiles": "CC(=O)O",
    })
    _save_decision(
        str(decisions_path), review_row, "replace_input_smiles", "use conjugate base",
        "CC(=O)[O-]", "",
    )
    assert _load_decisions(str(decisions_path)).iloc[0]["review_decision"] == "replace_input_smiles"

    dataset, quarantine, _, _ = build_dataset(
        **_builder_kwargs(tmp_path, decisions_path)
    )
    assert quarantine.empty
    assert len(dataset) == 1
    row = dataset.iloc[0]
    assert row["raw_structure_smiles"] == "CC(=O)O"
    assert row["input_smiles"] == "CC(=O)[O-]"
    assert row["structure_provenance"] == "manual_pka_structure_review_replacement_overlay"
    assert row["structure_review_decision"] == "replace_input_smiles"


def test_transition_correction_is_quarantined_until_supported(tmp_path):
    decisions_path = tmp_path / "decisions.csv"
    pd.DataFrame([{
        **_assignment_row(),
        "original_smiles": "CC(=O)O",
        "review_decision": "correct_transition_definition",
        "replacement_smiles": "",
        "replacement_transition_mode": "acidic_deprotonation",
        "review_note": "wrong direction",
        "reviewed_at": "2026-09-08T00:00:00Z",
    }]).to_csv(decisions_path, index=False)

    dataset, quarantine, _, _ = build_dataset(
        **_builder_kwargs(tmp_path, decisions_path)
    )
    assert dataset.empty
    assert len(quarantine) == 1
    assert quarantine.iloc[0]["quarantine_reason"] == (
        "manual_pka_structure_review:transition_definition_requires_rebuild_support"
    )
