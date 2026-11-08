"""Contracts for leakage-safe pKahub training supplements."""

import json

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from prepare_pkahub_training_data import prepare
from stage1_train_intrinsic_pka import (
    _load_supplemental_sites,
    _supplemental_gate,
)
from stage2_train_network_context import _cap_weak_supervision_weights


def _pkahub_row(molid, smiles, pka, source="IUPAC digitized pKa"):
    return {
        "pka_value": pka,
        "data_source": source,
        "charge_state_pre": 0,
        "charge_state_post": -1,
        "microspecies_pre": "teacher-state-a",
        "microspecies_post": "teacher-state-b",
        "molid": molid,
        "smiles": smiles,
        "rawdataID": f"raw-{molid}-{pka}",
        "temperature": 25.0,
        "ionic_strength": None,
        "assignment_method": "epikx.nearest_reference_match",
        "assignment_error": 0.1,
        "primary_source": "fixture",
    }


def test_prepare_separates_single_transition_and_macro_set_without_teacher(tmp_path):
    source = tmp_path / "pkahub.tsv"
    pd.DataFrame([
        _pkahub_row("acid", "CC(=O)O", 4.6),
        _pkahub_row("acid", "CC(=O)O", 4.8),
        _pkahub_row("gly", "NCC(=O)O", 2.3),
        _pkahub_row("gly", "NCC(=O)O", 9.6),
        _pkahub_row("heldout", "CCC(=O)O", 4.9, source="SAMPL8"),
    ]).to_csv(source, sep="\t", index=False)

    summary = prepare(
        source,
        tmp_path / "prepared",
        tmp_path / "missing-stage1.csv",
        tmp_path / "missing-network.csv",
    )
    stage1 = pd.read_csv(tmp_path / "prepared/stage1_supplemental_sites.csv")
    macro = pd.read_csv(tmp_path / "prepared/macro_set_candidates.csv")

    assert summary["stage1_accepted_rows"] == 1
    assert stage1.iloc[0]["pka_value"] == pytest.approx(4.7)
    assert bool(stage1.iloc[0]["assignment_teacher_used"]) is False
    assert bool(stage1.iloc[0]["pkahub_assignment_columns_ignored"]) is True
    assert json.loads(macro.iloc[0]["experimental_pka_values_json"]) == [2.3, 9.6]
    assert summary["macro_set_accepted_rows"] == 1


def test_supplement_loader_rejects_teacher_and_caps_total_weight(tmp_path):
    canonical = pd.DataFrame({
        "smiles": ["CC(=O)O", "CCC(=O)O"],
        "sample_weight": [1.0, 1.0],
    })
    path = tmp_path / "supplement.csv"
    supplemental = pd.DataFrame({
        "smiles": ["CN", "CCN"],
        "candidate_label": ["primary_amine", "primary_amine"],
        "final_group_label": ["primary_amine", "primary_amine"],
        "site_family": ["amine", "amine"],
        "pka_type_canonical": ["basic", "basic"],
        "pka_value": [10.5, 10.7],
        "sample_weight_raw": [10.0, 10.0],
        "measurement_method": ["experimental", "experimental"],
        "assignment_teacher_used": [False, False],
        "pkahub_assignment_columns_ignored": [True, True],
        "training_cohort": [
            "pkahub_clean_single_transition", "pkahub_clean_single_transition"
        ],
        "molecule_id": ["one", "two"],
    })
    supplemental.to_csv(path, index=False)

    loaded, report = _load_supplemental_sites(path, canonical)
    assert loaded["sample_weight"].sum() == pytest.approx(1.0)
    assert report["maximum_weight_fraction_of_canonical"] == 0.5

    supplemental["assignment_teacher_used"] = True
    supplemental.to_csv(path, index=False)
    with pytest.raises(ValueError, match="prohibited assignment teachers"):
        _load_supplemental_sites(path, canonical)


def test_supplement_gate_uses_only_canonical_rows_for_acceptance():
    training = pd.DataFrame({
        "pka_value": [1.0, 2.0, 3.0, 4.0, 9.0, 10.0],
        "sample_weight": [1.0] * 6,
        "scaffold_group": ["a", "b", "c", "d", "e", "f"],
        "training_cohort": ["canonical_network"] * 4
        + ["pkahub_clean_single_transition"] * 2,
    })
    features = pd.DataFrame({"x": [0.0] * 6})
    result = _supplemental_gate(
        DummyRegressor(strategy="mean"), features, training, folds=3
    )
    assert result["canonical_rows"] == 4
    assert set(result) >= {"baseline", "augmented", "deployed"}


def test_stage2_weight_budget_preserves_legacy_before_pkahub_macro():
    exact = pd.DataFrame({"sample_weight": [2.0, 2.0, 2.0, 2.0]})
    weak = pd.DataFrame({
        "supervision_scope": [
            "weak_molecule_macro_marginalized",
            "weak_molecule_macro_set_jointly_matched",
            "weak_molecule_macro_set_jointly_matched",
        ],
        "sample_weight": [0.5, 10.0, 10.0],
        "sample_weight_raw": [0.5, 10.0, 10.0],
    })

    legacy, pkahub, report = _cap_weak_supervision_weights(exact, weak)

    assert legacy["sample_weight"].sum() == pytest.approx(0.5)
    assert pkahub["sample_weight"].sum() == pytest.approx(1.5)
    assert report["maximum_total_weight"] == pytest.approx(2.0)
    assert set(legacy["training_cohort"]) == {"legacy_weak"}
    assert set(pkahub["training_cohort"]) == {"pkahub_macro_weak"}
