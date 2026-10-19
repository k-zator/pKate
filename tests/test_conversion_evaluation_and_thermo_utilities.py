"""Tests for data conversion, evaluation, and thermodynamic form helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

import build_epik_charged_smiles_from_sdf as epik
import build_intrinsic_dataset as intrinsic
import eval_site_state_pka as site_eval
import thermo_pair_form_model as thermo


def _write_sdf(path: Path, records: list[tuple[str, dict[str, str]]]) -> None:
    writer = Chem.SDWriter(str(path))
    try:
        for smiles, props in records:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            for key, value in props.items():
                mol.SetProp(key, value)
            writer.write(mol)
    finally:
        writer.close()


def test_epik_scalar_and_site_state_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert epik.safe_float(None) is None
    assert epik.safe_float(" ") is None
    assert epik.safe_float("bad") is None
    assert epik.safe_float("4.2") == pytest.approx(4.2)
    assert epik.has_formal_charge_token("C[NH3+]")
    assert not epik.has_formal_charge_token(None)
    assert not epik.has_formal_charge_token("CCN")
    assert epik.collect_site_indices({"r_epik_pKa_2": 3, "x": 1, "r_epik_pKa_1": 4}) == [1, 2]

    amine = Chem.MolFromSmiles("CN")
    assert epik.get_mol_formal_charge(amine) == 0
    smiles, charge, error = epik.generate_site_state_smiles(amine, 1, +1)
    assert smiles is not None and charge == 1 and error is None
    assert epik.has_formal_charge_token(smiles)

    monkeypatch.setattr(epik, "_mol_to_smiles_quick", lambda _mol: None)
    smiles, charge, error = epik.generate_site_state_smiles(amine, 1, +1)
    assert smiles is not None and charge == 1 and error is None

    monkeypatch.setattr(epik, "_apply_site_delta", lambda *_args, **_kwargs: (None, "failed"))
    assert epik.generate_site_state_smiles(amine, 1, +1) == (None, None, "failed")


def test_epik_cli_builds_assigned_and_both_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "epik.sdf"
    output = tmp_path / "nested" / "states.csv"
    _write_sdf(
        source,
        [
            (
                "CN",
                {
                    "chembl_id": "CHEMBL1",
                    "i_epik_Tot_Q": "0",
                    "r_epik_pKa_1": "9.5",
                    "i_epik_pKa_atom_1": "2",
                    "s_epik_pKa_identifier_1": "basic site",
                    "r_epik_pKa_2": "4.0",
                    "i_epik_pKa_atom_2": "99",
                },
            ),
            ("CCO", {}),
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_epik_charged_smiles_from_sdf.py",
            "--input", str(source),
            "--output", str(output),
            "--emit-both-states",
            "--max-mols", "1",
        ],
    )
    epik.main()
    frame = pd.read_csv(output)
    assert len(frame) == 1
    assert frame.iloc[0]["assigned_state_at_ph"] == "protonated"
    assert bool(frame.iloc[0]["assigned_generation_ok"])
    assert bool(frame.iloc[0]["protonated_generation_ok"])
    assert frame.iloc[0]["epik_atom_symbol"] == "N"


def test_intrinsic_parser_and_main_clean_filter_and_deduplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    good = raw / "good.sdf"
    ignored = raw / "INCORRECT_bad.sdf"
    _write_sdf(
        good,
        [
            ("CCN", {"pKa": "9.5"}),
            ("CCN", {"pKa": "9.5"}),
            ("CCO", {"pKa": "not-numeric"}),
            ("CCO", {}),
            ("O", {"pKa": "25"}),
        ],
    )
    _write_sdf(ignored, [("CC(=O)O", {"pKa": "4.7"})])
    parsed = intrinsic.parse_sdf(str(good))
    assert len(parsed) == 3

    output = tmp_path / "processed" / "master.csv"
    monkeypatch.setattr(intrinsic, "RAW_DIR", str(raw))
    monkeypatch.setattr(intrinsic, "OUT_PATH", str(output))
    monkeypatch.chdir(tmp_path)
    intrinsic.main()
    result = pd.read_csv(output)
    assert len(result) == 1
    assert result.iloc[0]["pKa"] == pytest.approx(9.5)
    assert result.iloc[0]["source_file"] == "good.sdf"


def _evaluation_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"molecule_key": "a", "pred_proba": 0.9, "is_true_site": 1, "candidate_prior_mean_pka": 5.0, "pka_value": 6.0},
        {"molecule_key": "a", "pred_proba": 0.1, "is_true_site": 0, "candidate_prior_mean_pka": 7.0, "pka_value": 6.0},
        {"molecule_key": "b", "pred_proba": 0.8, "is_true_site": 0, "candidate_prior_mean_pka": 9.0, "pka_value": 7.0},
        {"molecule_key": "b", "pred_proba": 0.7, "is_true_site": 1, "candidate_prior_mean_pka": 8.0, "pka_value": 7.0},
    ])


def test_site_evaluation_metrics_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _evaluation_frame()
    assert site_eval.topk_site_accuracy(frame, 1) == pytest.approx(0.5)
    assert site_eval.topk_site_accuracy(frame, 3) == pytest.approx(1.0)
    assert site_eval.selected_candidate_mae(frame) == pytest.approx(1.5)
    metrics = site_eval.evaluate(frame)
    assert metrics["molecules"] == 2 and metrics["rows"] == 4
    assert np.isnan(site_eval.selected_candidate_mae(frame.drop(columns="pka_value")))

    csv = tmp_path / "predictions.csv"
    report = tmp_path / "metrics.json"
    frame.to_csv(csv, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_site_state_pka.py", "--predictions", str(csv), "--out-json", str(report)],
    )
    site_eval.main()
    assert json.loads(report.read_text())["top1_site_acc"] == pytest.approx(0.5)

    frame.drop(columns="is_true_site").to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Missing required"):
        site_eval.main()


class _ProbabilityModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.linspace(0.2, 0.8, len(frame))
        return np.column_stack([1.0 - values, values])


class _PredictionModel:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), 0.75)


def test_thermo_feature_modes_inference_and_application() -> None:
    frame = pd.DataFrame([
        {
            "ph_target": 7.4,
            "pred_effective_pka": 9.0,
            "intrinsic_pred_pka": 8.5,
            "pred_delta_pka": 0.5,
            "candidate_prior_mean_pka": "8.0",
            "candidate_family": "amine",
            "pka_type_canonical": "basic",
            "ion_ctx_neighbor_charge": 1,
            "candidate_member_form": "acid_form",
        },
        {
            "ph_target": None,
            "pred_effective_pka": 4.0,
            "candidate_family": None,
            "candidate_member_form": "base_form",
            "pred_member_form_rule_source": "baseline",
        },
        {"candidate_member_form": "individual"},
    ])
    full = thermo.build_thermo_pair_form_feature_frame(frame, "full")
    weak = thermo.build_thermo_pair_form_feature_frame(frame, "weak-prior")
    assert "candidate_prior_mean_pka" in full
    assert "candidate_prior_mean_pka" not in weak
    assert "family_amine" in full and "ion_ctx_neighbor_charge" in full
    assert full.isna().sum().sum() == 0
    with pytest.raises(ValueError, match="Unsupported"):
        thermo.build_thermo_pair_form_feature_frame(frame, "mystery")

    assert thermo.infer_thermo_pair_form_probability(frame, None).isna().all()
    assert thermo.infer_thermo_pair_form_probability(frame, {}).isna().all()
    columns = list(full.columns)
    bundle = {
        "model": _ProbabilityModel(),
        "feature_columns": columns,
        "feature_mode": "full",
        "decision_threshold": 0.5,
        "decision_margin": 0.1,
    }
    probability = thermo.infer_thermo_pair_form_probability(frame, bundle)
    assert probability.tolist() == pytest.approx([0.2, 0.5, 0.8])
    applied = thermo.apply_thermo_pair_form_model(frame, bundle)
    assert applied.loc[0, "pred_member_form"] == "base_form"
    assert applied.loc[0, "pred_member_form_rule_source"] == "thermo_pair_form_model"
    assert pd.isna(applied.loc[1, "pred_member_form"])
    assert thermo.apply_thermo_pair_form_model(frame, None) is frame

    direct = thermo.infer_thermo_pair_form_probability(
        frame,
        {"model": _PredictionModel(), "feature_columns": list(weak.columns), "feature_mode": "weak_prior"},
    )
    assert direct.tolist() == pytest.approx([0.75, 0.75, 0.75])
