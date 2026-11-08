"""Contracts for the optional structure-dependent protein correction layer."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from sklearn.dummy import DummyRegressor

from protein_environment_correction import (
    FEATURE_COLUMNS,
    LigandSelector,
    PROTEIN_ENVIRONMENT_FEATURE_SCHEMA_VERSION,
    PROTEIN_ENVIRONMENT_METHOD,
    PROTEIN_ENVIRONMENT_SCHEMA_VERSION,
    apply_protein_environment,
    apply_protein_environment_file,
    environment_descriptors,
    ligand_analogue,
    map_ligand_coordinates,
    read_structure_atoms,
    residue_site_center,
)
import train_protein_environment_model as environment_training
from train_protein_environment_model import (
    build_perturbation_table,
    homology_groups,
    replicate_measurement_summary,
    train,
)


def _write_pose(path: Path, smiles: str = "CN") -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(index, (float(index), 0.0, 0.0))
    molecule.AddConformer(conformer)
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()
    return molecule


def _write_protein(path: Path, distance: float = 2.0) -> None:
    path.write_text(
        f"ATOM      1  CA  ALA A   1       1.000   0.000   {distance:5.3f}  1.00 20.00           C  \n"
        f"ATOM      2  OD1 ASP A   2       1.000   0.000   {distance + 1:5.3f}  1.00 20.00           O  \n"
        f"ATOM      3  OD2 ASP A   2       2.000   0.000   {distance + 1:5.3f}  1.00 20.00           O  \n"
        "END\n",
        encoding="utf-8",
    )


def _model(path: Path, delta: float = 2.0, calibration=None) -> None:
    x = pd.DataFrame(
        np.zeros((2, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    estimator = DummyRegressor(strategy="constant", constant=delta).fit(x, [0.0, 0.0])
    bundle = {
        "model": estimator,
        "feature_columns": list(FEATURE_COLUMNS),
        "schema_version": PROTEIN_ENVIRONMENT_SCHEMA_VERSION,
        "feature_schema_version": PROTEIN_ENVIRONMENT_FEATURE_SCHEMA_VERSION,
        "method": PROTEIN_ENVIRONMENT_METHOD,
        "training_rows": 10,
        "oof_mae": 1.0,
        "oof_p90_abs_error": 2.0,
        "chemistry_p90_abs_error": {"amine": 2.0},
        "training_source_sha256": "synthetic",
        "prediction_clip": [-5.0, 5.0],
    }
    if calibration is not None:
        bundle["calibration"] = calibration
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)


def _stage2_row() -> pd.DataFrame:
    site = {
        "site_id": "site_1",
        "family": "amine",
        "label": "primary_amine",
        "atom_maps": [2],
        "center_maps": [2],
        "stage1_intrinsic_pka": 10.0,
        "stage2_predicted_macro_pka": 10.0,
        "stage2_predicted_macro_pka_ci_low": 9.0,
        "stage2_predicted_macro_pka_ci_high": 11.0,
    }
    nodes = [
        {
            "node_id": "base",
            "site_forms": {"site_1": "base_form"},
            "protonated_site_count": 0,
        },
        {
            "node_id": "acid",
            "site_forms": {"site_1": "acid_form"},
            "protonated_site_count": 1,
        },
    ]
    edge = {
        "site_id": "site_1",
        "from_node_id": "base",
        "to_node_id": "acid",
        "stage2_contextual_edge_pka": 10.0,
    }
    step = {
        "protonation_step": 1,
        "rank_associated_site_id": "site_1",
        "stage2_predicted_macro_pka": 10.0,
    }
    free_energy = {
        "one_body_terms": [{"site_id": "site_1", "one_body_pka": 10.0}],
        "pair_couplings": [],
        "diagnostics": {},
    }
    return pd.DataFrame([{
        "sites_json": json.dumps([site]),
        "microstate_nodes_json": json.dumps(nodes),
        "microstate_edges_json": json.dumps([edge]),
        "stage2_macro_pka_steps_json": json.dumps([step]),
        "stage2_free_energy_model_json": json.dumps(free_energy),
        "protonation_state_enumeration_truncated": "False",
        "incomplete_site_ids_json": "[]",
    }])


def test_structure_descriptors_exclude_target_and_preserve_charge_proxy(tmp_path):
    protein = tmp_path / "protein.pdb"
    _write_protein(protein)
    atoms = read_structure_atoms(protein)
    descriptors = environment_descriptors(
        atoms,
        np.asarray([1.0, 0.0, 2.0]),
        chemistry="amine",
        excluded_residue_keys=[("A", "1", "", "ALA")],
    )

    assert descriptors["protein_heavy_atoms_4a"] == 2.0
    assert descriptors["negative_atoms_6a"] == 2.0
    assert descriptors["signed_charge_over_distance"] < 0.0
    assert list(descriptors) == list(FEATURE_COLUMNS)


def test_environment_shift_updates_one_body_edges_and_macro_ladder(tmp_path):
    ligand_path = tmp_path / "ligand.sdf"
    query = _write_pose(ligand_path)
    protein_path = tmp_path / "protein.pdb"
    _write_protein(protein_path)
    model_path = tmp_path / "environment.pkl"
    _model(model_path, delta=2.0)

    corrected, report = apply_protein_environment(
        _stage2_row(),
        query_molecule=query,
        protein_structure_path=protein_path,
        ligand_structure_path=ligand_path,
        model_path=model_path,
        ligand_selector=LigandSelector(),
    )
    row = corrected.iloc[0]
    site = json.loads(row["sites_json"])[0]
    edge = json.loads(row["microstate_edges_json"])[0]
    term = json.loads(row["stage2_free_energy_model_json"])["one_body_terms"][0]
    step = json.loads(row["stage2_macro_pka_steps_json"])[0]

    # Primary amines use the declared 0.65 lysine-to-ligand transfer shrinkage.
    assert site["protein_environment_delta_pka"] == pytest.approx(1.3)
    assert site["protein_environment_applied"] is True
    assert term["one_body_pka"] == pytest.approx(11.3)
    assert edge["stage2_contextual_edge_pka"] == pytest.approx(11.3)
    assert step["stage2_predicted_macro_pka"] == pytest.approx(11.3)
    assert row["protein_environment_applied_site_count"] == 1
    assert report["nearest_pose_contact_angstrom"] < 8.0


def test_joint_network_crosses_ligand_with_nearby_residue_states(tmp_path):
    ligand_path = tmp_path / "ligand.sdf"
    query = _write_pose(ligand_path)
    protein_path = tmp_path / "protein.pdb"
    _write_protein(protein_path)
    model_path = tmp_path / "environment.pkl"
    _model(model_path, delta=2.0)
    stage2 = _stage2_row()
    edges = json.loads(stage2.iloc[0]["microstate_edges_json"])
    edges[0].update({
        "edge_id": "edge_site_1",
        "acid_node_id": "acid",
        "base_node_id": "base",
    })
    stage2.loc[0, "microstate_edges_json"] = json.dumps(edges)

    corrected, report = apply_protein_environment(
        stage2,
        query_molecule=query,
        protein_structure_path=protein_path,
        ligand_structure_path=ligand_path,
        model_path=model_path,
        joint_protein_sites=True,
    )
    row = corrected.iloc[0]
    sites = json.loads(row["sites_json"])
    nodes = json.loads(row["microstate_nodes_json"])
    joint_edges = json.loads(row["microstate_edges_json"])
    free_energy = json.loads(row["stage2_free_energy_model_json"])
    protein_sites = [
        site for site in sites if site.get("component_type") == "protein_residue"
    ]

    assert bool(row["joint_protein_network_applied"]) is True
    assert len(protein_sites) == 1
    assert protein_sites[0]["protein_residue_name"] == "ASP"
    assert len(nodes) == 4
    assert len(joint_edges) == 4
    assert len(json.loads(row["stage2_macro_pka_steps_json"])) == 2
    assert len(free_energy["one_body_terms"]) == 2
    assert len(free_energy["pair_couplings"]) == 1
    assert free_energy["diagnostics"]["joint_max_cycle_closure_error"] < 1.0e-10
    assert report["joint_protein_network"]["selected_site_count"] == 1


def test_joint_network_respects_global_node_cap(tmp_path):
    ligand_path = tmp_path / "ligand.sdf"
    query = _write_pose(ligand_path)
    protein_path = tmp_path / "protein.pdb"
    _write_protein(protein_path)
    model_path = tmp_path / "environment.pkl"
    _model(model_path)

    corrected, report = apply_protein_environment(
        _stage2_row(),
        query_molecule=query,
        protein_structure_path=protein_path,
        ligand_structure_path=ligand_path,
        model_path=model_path,
        joint_protein_sites=True,
        maximum_joint_nodes=2,
    )

    assert bool(corrected.iloc[0]["joint_protein_network_applied"]) is False
    assert len(json.loads(corrected.iloc[0]["microstate_nodes_json"])) == 2
    assert report["joint_protein_network"]["status"] == (
        "no_nearby_tractable_protein_sites"
    )
    assert report["joint_protein_network"]["cap_excluded_site_count"] == 1


def test_environment_rejects_different_coordinate_frames(tmp_path):
    ligand_path = tmp_path / "ligand.sdf"
    query = _write_pose(ligand_path)
    protein_path = tmp_path / "protein.pdb"
    _write_protein(protein_path, distance=30.0)
    model_path = tmp_path / "environment.pkl"
    _model(model_path)

    with pytest.raises(ValueError, match="different frames"):
        apply_protein_environment(
            _stage2_row(),
            query_molecule=query,
            protein_structure_path=protein_path,
            ligand_structure_path=ligand_path,
            model_path=model_path,
        )


def test_calibration_gate_withholds_harmful_chemistry_without_widening(tmp_path):
    ligand_path = tmp_path / "ligand.sdf"
    query = _write_pose(ligand_path)
    protein_path = tmp_path / "protein.pdb"
    _write_protein(protein_path)
    model_path = tmp_path / "environment.pkl"
    _model(model_path, calibration={
        "schema_version": "1.0.0",
        "chemistry": {
            "amine": {
                "correction_scale": 0.0,
                "correction_evidence_tier": (
                    "no_grouped_oof_improvement_correction_withheld"
                ),
                "absolute_error_radius_90": 3.5,
            }
        },
    })

    corrected, report = apply_protein_environment(
        _stage2_row(),
        query_molecule=query,
        protein_structure_path=protein_path,
        ligand_structure_path=ligand_path,
        model_path=model_path,
    )
    site = json.loads(corrected.iloc[0]["sites_json"])[0]
    assert site["protein_environment_applied"] is False
    assert site["protein_environment_delta_pka"] == 0.0
    assert site["stage2_predicted_macro_pka"] == pytest.approx(10.0)
    assert site["stage2_predicted_macro_pka_ci_low"] == pytest.approx(9.0)
    assert report["calibration_withheld_site_count"] == 1


def test_homology_groups_are_transitive():
    rows = [
        {"PDB": "1AAA", "Chain": "A", "Sequence_Identity_>_30%": "2BBB.B"},
        {"PDB": "2BBB", "Chain": "B", "Sequence_Identity_>_30%": "3CCC.C"},
        {"PDB": "4DDD", "Chain": "A", "Sequence_Identity_>_30%": ""},
    ]
    groups = homology_groups(rows)
    assert groups["1AAA.A"] == groups["2BBB.B"] == groups["3CCC.C"]
    assert groups["4DDD.A"] != groups["1AAA.A"]


def test_mmcif_parser_and_residue_center(tmp_path):
    cif = tmp_path / "protein.cif"
    cif.write_text(
        "data_test\nloop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.type_symbol\n"
        "_atom_site.auth_atom_id\n"
        "_atom_site.auth_comp_id\n"
        "_atom_site.auth_asym_id\n"
        "_atom_site.auth_seq_id\n"
        "_atom_site.pdbx_PDB_ins_code\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.pdbx_PDB_model_num\n"
        "ATOM O OD1 ASP A 2 ? 0.0 0.0 0.0 1\n"
        "ATOM O OD2 ASP A 2 ? 2.0 0.0 0.0 1\n#\n",
        encoding="utf-8",
    )
    atoms = read_structure_atoms(cif)
    center, key = residue_site_center(
        atoms, chain="A", residue_id="2", residue_name="ASP"
    )
    assert len(atoms) == 2
    assert center.tolist() == [1.0, 0.0, 0.0]
    assert key == ("A", "2", "", "ASP")


@pytest.mark.parametrize(
    ("family", "label", "chemistry", "factor"),
    [
        ("carboxyl", "carboxylic_acid", "carboxyl", 0.90),
        ("imidazole_basicity", "imidazole", "imidazole", 0.80),
        ("amine", "tertiary_amine", "amine", 0.65),
        ("amine", "guanidine_like", "amine", 0.50),
        ("thiol_thiolate", "thiol", "thiol", 0.75),
        ("phenol_phenolate", "phenol", "phenol", 0.75),
        ("phosphate", "phosphate", None, 0.0),
    ],
)
def test_ligand_analogue_contract(family, label, chemistry, factor):
    result = ligand_analogue({"family": family, "label": label})
    assert result[0] == chemistry
    assert result[1] == pytest.approx(factor)


def test_full_graph_ligand_mapping_rejects_wrong_heavy_atom_count(tmp_path):
    ligand_path = tmp_path / "ligand.sdf"
    _write_pose(ligand_path, "CN")
    with pytest.raises(ValueError, match="same heavy atoms"):
        map_ligand_coordinates(Chem.MolFromSmiles("CCN"), ligand_path)


def test_file_wrapper_writes_corrected_table_and_report(tmp_path):
    ligand_path = tmp_path / "ligand.sdf"
    query = _write_pose(ligand_path)
    protein_path = tmp_path / "protein.pdb"
    _write_protein(protein_path)
    model_path = tmp_path / "environment.pkl"
    _model(model_path)
    input_path = tmp_path / "stage2.csv"
    output_path = tmp_path / "corrected.csv"
    report_path = tmp_path / "report.json"
    _stage2_row().to_csv(input_path, index=False)

    corrected, report = apply_protein_environment_file(
        input_path,
        output_path,
        query_molecule=query,
        protein_structure_path=protein_path,
        ligand_structure_path=ligand_path,
        model_path=model_path,
        report_path=report_path,
    )
    assert len(corrected) == 1
    assert output_path.is_file()
    assert report_path.is_file()
    assert json.loads(report_path.read_text())["output_sha256"] == report["output_sha256"]


def test_tiny_training_roundtrip_writes_provenance_artifacts(tmp_path):
    structures = tmp_path / "structures"
    structures.mkdir()
    rows = []
    for index, (pdb, pka) in enumerate((("1AAA", 4.0), ("2BBB", 5.0)), start=1):
        _write_protein(structures / f"{pdb}.cif.gz")
        # The parser is selected by extension, so store the PDB fixture using
        # the PDB path and reference it through an equivalent mmCIF name below.
        (structures / f"{pdb}.cif.gz").unlink()
        cif = structures / f"{pdb}.cif.gz"
        import gzip
        with gzip.open(cif, "wt", encoding="utf-8") as handle:
            handle.write(
                "data_test\nloop_\n"
                "_atom_site.group_PDB\n_atom_site.type_symbol\n"
                "_atom_site.auth_atom_id\n_atom_site.auth_comp_id\n"
                "_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n"
                "_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n"
                "_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
                "_atom_site.pdbx_PDB_model_num\n"
                "ATOM O OD1 ASP A 2 ? 0.0 0.0 0.0 1\n"
                "ATOM O OD2 ASP A 2 ? 2.0 0.0 0.0 1\n"
                "ATOM N NZ LYS A 3 ? 1.0 0.0 3.0 1\n#\n"
            )
        rows.append({
            "Index": str(index),
            "PDB": pdb,
            "Chain": "A",
            "ResID_in_PDB": "2",
            "ResName": "ASP",
            "Expt_pKa": str(pka),
            "pKa_Classification": "Main",
            "Warning": "",
            "Sequence_Identity_>_30%": "",
        })
    source = tmp_path / "pkadr.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "model" / "protein_environment_model.pkl"

    metrics = train(source, structures, output)
    assert metrics["training_rows"] == 2
    assert metrics["homology_groups"] == 2
    assert metrics["calibration"]["strict_structure_mapped_cohort_rows"] == 2
    assert output.is_file()
    assert (output.parent / "metrics.json").is_file()
    assert (output.parent / "calibration.json").is_file()
    assert (output.parent / "strict_648_site_calibration_cohort.csv").is_file()
    assert (output.parent / "training_features_and_oof.csv").is_file()
    assert (output.parent / "perturbation_pairs_and_oof.csv").is_file()
    assert (output.parent / "perturbation_excluded_rows.json").is_file()


def test_pkadr_alternatives_are_paired_without_inflating_repeats(
    tmp_path, monkeypatch
):
    rows = [
        {
            "Index": "1", "PDB": "1AAA", "Chain": "A",
            "ResID_in_PDB": "2", "ResName": "ASP", "Expt_pKa": "4.0",
            "pKa_Classification": "Main", "Warning": "",
            "Sequence_Identity_>_30%": "",
        },
        {
            "Index": "2", "PDB": "1AAA", "Chain": "A",
            "ResID_in_PDB": "2", "ResName": "ASP", "Expt_pKa": "4.2",
            "pKa_Classification": "Alt. pKa", "Warning": "",
            "Sequence_Identity_>_30%": "",
        },
        {
            "Index": "3", "PDB": "2BBB", "Chain": "A",
            "ResID_in_PDB": "2", "ResName": "ASP", "Expt_pKa": "5.0",
            "pKa_Classification": "Alt. pKa (mutant)", "Warning": "",
            "Sequence_Identity_>_30%": "", "Mut_Pos": "A10V",
        },
        {
            "Index": "4", "PDB": "3CCC", "Chain": "A",
            "ResID_in_PDB": "8", "ResName": "GLU", "Expt_pKa": "4.3",
            "pKa_Classification": "Main", "Warning": "",
            "Sequence_Identity_>_30%": "",
        },
        {
            "Index": "5", "PDB": "4DDD", "Chain": "A",
            "ResID_in_PDB": "8", "ResName": "GLU", "Expt_pKa": "3.8",
            "pKa_Classification": "Alt. pKa (state)", "Warning": "",
            "Sequence_Identity_>_30%": "",
        },
    ]
    source = tmp_path / "pkadr.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(
        environment_training, "read_structure_atoms", lambda path: str(path)
    )
    monkeypatch.setattr(
        environment_training,
        "residue_site_center",
        lambda atoms, **kwargs: (np.zeros(3), ("A", "1", "")),
    )
    monkeypatch.setattr(
        environment_training,
        "environment_descriptors",
        lambda *args, **kwargs: {name: 0.0 for name in FEATURE_COLUMNS},
    )

    pairs, exclusions = build_perturbation_table(source, tmp_path)
    repeats = replicate_measurement_summary(rows)
    groups = homology_groups(rows)

    assert len(pairs) == 2
    assert set(pairs["classification"]) == {
        "Alt. pKa (mutant)", "Alt. pKa (state)"
    }
    assert set(pairs["homology_group"]) == {"1AAA.A", "3CCC.A"}
    assert groups["1AAA.A"] == groups["2BBB.A"]
    assert groups["3CCC.A"] == groups["4DDD.A"]
    assert pairs["augmentation_weight"].tolist() == pytest.approx([0.5, 0.5])
    assert repeats["rows"] == 1
    assert repeats["median_absolute_difference"] == pytest.approx(0.2)
    assert exclusions == []
