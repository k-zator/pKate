"""PKAD-3 preparation, acquisition, and leakage-safe fitting contracts."""

from __future__ import annotations

import gzip
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

import acquire_pkad3_structures as acquisition
import prepare_pkad3_training_data as preparation
import train_protein_environment_model as training
from protein_environment_correction import FEATURE_COLUMNS


def _raw_row(**updates):
    row = {
        "PDB_ID": "1AAA", "Chain": "A", "Res_Name": "ASP", "Res_ID": 10,
        "Expt_pKa": 4.2, "Uniprot_ID": "P00001", "info": "1AAA-A-10",
        "infor": "P00001-10", "group_code": 1, "model_pka": 99.0,
        "com_min_d0_polar": -99.0,
    }
    row.update(updates)
    return row


def test_pkad3_preparation_removes_pkadr_overlap_and_shares_measurement_weight(
    tmp_path,
):
    raw = tmp_path / "split"
    raw.mkdir()
    pd.DataFrame([
        _raw_row(),
        _raw_row(PDB_ID="2BBB", info="2BBB-A-10"),
        _raw_row(PDB_ID="3CCC", Res_ID=20, info="3CCC-A-20", infor="P00001-20",
                 Expt_pKa=5.0),
    ]).to_csv(raw / "train0.csv", index=False)
    pd.DataFrame([
        _raw_row(PDB_ID="4DDD", Res_ID=20, info="4DDD-A-20", infor="P00001-20",
                 Expt_pKa=5.0),
        _raw_row(PDB_ID="BAD_IDENTIFIER", info="bad"),
    ]).to_csv(raw / "test0.csv", index=False)
    pkadr = tmp_path / "pkadr.json"
    pkadr.write_text(json.dumps([{
        "PDB": "1AAA", "Chain": "A", "ResName": "ASP", "ResID_in_PDB": "10"
    }]), encoding="utf-8")
    output = tmp_path / "prepared"
    report = preparation.prepare(raw, pkadr, output)

    candidates = pd.read_csv(output / "training_candidates.csv")
    exclusions = pd.read_csv(output / "preparation_exclusions.csv")
    assert set(candidates["pdb"]) == {"2BBB", "3CCC", "4DDD"}
    repeated = candidates[candidates["pdb"].isin(["3CCC", "4DDD"])]
    assert repeated["sample_weight_raw"].tolist() == pytest.approx([0.25, 0.25])
    assert report["raw_candidate_weight"] == pytest.approx(1.0)
    assert "model_pka" in report["ignored_computed_columns"]
    assert "exact_structure_site_overlap_with_pkadr" in set(exclusions["reason"])
    assert "no_experimental_four_character_pdb" in set(exclusions["reason"])


def test_pkad3_fit_recomputes_features_and_rejects_teacher_rows(tmp_path, monkeypatch):
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame([{
        "supplement_id": "s1", "pdb": "2BBB", "chain": "A",
        "residue_name": "ASP", "residue_id": "10", "experimental_pka": "5.0",
        "measurement_group_key": "m1", "protein_group_key": "P00001",
        "sample_weight_raw": "0.5", "label_origin": "experimental",
        "computational_teacher": "", "computed_pkad3_features_used": "False",
        "computed_model_pka_used": "False", "eligible_for_training": "True",
    }]).to_csv(candidates, index=False)
    pkadr = tmp_path / "pkadr.json"
    pkadr.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(training, "read_structure_atoms", lambda path: [path])
    monkeypatch.setattr(
        training, "residue_site_center",
        lambda atoms, **kwargs: (np.zeros(3), ("A", "10", "", "ASP")),
    )
    monkeypatch.setattr(
        training, "environment_descriptors",
        lambda *args, **kwargs: {name: 1.0 for name in FEATURE_COLUMNS},
    )
    table, exclusions = training.build_pkad3_supplement_table(
        candidates, tmp_path, pkadr
    )
    assert exclusions == []
    assert table.loc[0, "delta_pka"] == pytest.approx(1.1)
    assert table.loc[0, list(FEATURE_COLUMNS)].astype(float).eq(1.0).all()

    broken = pd.read_csv(candidates, dtype=str, keep_default_na=False)
    broken.loc[0, "computational_teacher"] = "KaML"
    broken.to_csv(candidates, index=False)
    with pytest.raises(ValueError, match="computationally derived"):
        training.build_pkad3_supplement_table(candidates, tmp_path, pkadr)


def test_pkad3_homology_extension_joins_shared_uniprot_contexts():
    canonical = pd.DataFrame([
        {"pdb": "1AAA", "chain": "A", "homology_group": "g1"},
        {"pdb": "2BBB", "chain": "B", "homology_group": "g2"},
    ])
    supplement = pd.DataFrame([
        {"pdb": "1AAA", "chain": "A", "protein_group_key": "P1"},
        {"pdb": "2BBB", "chain": "B", "protein_group_key": "P1"},
    ])
    joined_canonical, joined_supplement = training.extend_homology_groups(
        canonical, supplement
    )
    assert joined_canonical["homology_group"].nunique() == 1
    assert joined_supplement["homology_group"].nunique() == 1


def test_unaugmented_oof_paths_do_not_pass_synthetic_unit_weights(monkeypatch):
    """An inactive augmentation path must reproduce the baseline fit contract."""

    class NoImplicitWeightRegressor:
        def fit(self, x, y, sample_weight=None):
            assert sample_weight is None
            self.value = float(np.mean(y))
            return self

        def predict(self, x):
            return np.full(len(x), self.value, dtype=float)

    monkeypatch.setattr(training, "_model", NoImplicitWeightRegressor)
    rows = []
    for index, group in enumerate(["g1", "g1", "g2", "g2", "g3", "g3"]):
        row = {name: float(index) for name in FEATURE_COLUMNS}
        row.update({
            "delta_pka": float(index - 2),
            "homology_group": group,
            "chemistry": "carboxylate",
        })
        rows.append(row)
    canonical = pd.DataFrame(rows)

    perturbation_oof, pair_predictions = training.grouped_oof_with_perturbations(
        canonical,
        pd.DataFrame(),
        augment_training=False,
        folds=3,
    )
    pkad3_oof = training.grouped_oof_with_pkad3(
        canonical,
        pd.DataFrame(),
        pd.DataFrame(),
        augment_perturbations=False,
        augment_pkad3=False,
        folds=3,
    )

    assert np.isfinite(perturbation_oof).all()
    assert pair_predictions.empty
    assert pkad3_oof == pytest.approx(perturbation_oof)


def test_pkad3_structure_acquisition_writes_verified_manifest(tmp_path, monkeypatch):
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame({"pdb": ["1ABC", "1ABC"]}).to_csv(candidates, index=False)
    payload = (
        b"data_test\n" + b"# audit metadata\n" * 400
        + b"loop_\n_atom_site.group_PDB\n_atom_site.type_symbol\n"
        b"ATOM C\n#\n"
    )

    def download(url, destination):
        with gzip.open(destination, "wb") as handle:
            handle.write(payload)
        return str(destination), None

    monkeypatch.setattr(acquisition.urllib.request, "urlretrieve", download)
    output = tmp_path / "structures"
    first = acquisition.acquire(candidates, output, workers=1)
    second = acquisition.acquire(candidates, output, workers=1)
    assert first["requested_pdb_entries"] == 1
    assert first["structures"][0]["status"] == "downloaded_and_verified"
    assert second["structures"][0]["status"] == "verified_existing"
    assert first["structures"][0]["sha256"] == second["structures"][0]["sha256"]
