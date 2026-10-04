import json
import pickle

import pandas as pd
import pytest
from rdkit import Chem

from build_release_microstate_dataset import (
    _review_explicitly_selects_transition,
    _resolved_pair_sites,
    _select_reviewed_site,
    _select_site,
    build_dataset,
)
from build_molecule_microstate_network_dataset import (
    _element_topology_query,
    _map_measurement_to_site,
    _molecule_group_key,
    coupled_network_thermodynamics,
    independent_site_macro_pkas,
)
from functional_group_pka_analysis import (
    classify_pair_type,
    expand_pair_site_transitions,
    normalize_conjugate_family,
    select_pair_first_site,
)
from microstate_enumerator import enumerate_joint_protonation_network, generate_conjugate_microstates
from substructure_match import find_sites_with_metadata, resolve_overlapping_sites


def _pair_result(smiles: str, label: str):
    mol = Chem.MolFromSmiles(smiles)
    resolved, _ = resolve_overlapping_sites(find_sites_with_metadata(mol), 0.5)
    site = next(site for site in resolved if site["label"] == label)
    family = normalize_conjugate_family(label)
    mode, member_form = classify_pair_type(label, family)
    assert mode == "pair_type"
    return generate_conjugate_microstates(mol, site, family, member_form, max_tautomers=8)


@pytest.mark.parametrize(
    "smiles,label,acid_charge,base_charge",
    [
        ("CC(=O)O", "carboxylic_acid", 0, -1),
        ("CC(=O)[O-]", "carboxylate", 0, -1),
        ("COP(=O)(O)O", "phosphoric_acid", 0, -1),
        ("COP(=O)([O-])O", "phosphate_anion", 0, -1),
        ("CCO", "primary_alcohol", 0, -1),
        ("CC[O-]", "alkoxide", 0, -1),
        ("C[OH2+]", "protonated_alcohol", 1, 0),
        ("C[SH2+]", "protonated_thiol", 1, 0),
        ("CNN", "hydrazine", 1, 0),
        ("C[NH2+]N", "hydrazinium", 1, 0),
        ("CON", "hydroxylamine", 1, 0),
        ("CO[NH3+]", "hydroxylammonium", 1, 0),
        ("CN", "primary_amine", 1, 0),
        ("C[NH3+]", "primary_ammonium", 1, 0),
        ("CC1(C)CN1", "aziridine", 1, 0),
        ("CC1(C)C[NH2+]1", "aziridinium", 1, 0),
        ("c1ccncc1", "pyridine", 1, 0),
        ("c1ccc2ncccc2c1", "quinoline", 1, 0),
        ("c1ccc2cnccc2c1", "isoquinoline", 1, 0),
        ("c1cc[nH]c1", "pyrrole", 1, 0),
        ("NC=N", "amidine", 1, 0),
        ("NC=[NH2+]", "amidinium", 1, 0),
        ("c1nnn[nH]1", "tetrazole", 0, -1),
        ("c1nnn[n-]1", "tetrazolate", 0, -1),
        ("[O-][n+]1ccccc1", "n-oxide", 1, 0),
        ("C[N+](C)(C)[O-]", "amine_n_oxide", 1, 0),
        ("c1ccnnc1", "pyridazine", 1, 0),
        ("[nH+]1ncccc1", "pyridazinium", 1, 0),
        ("c1nc2ccccc2[nH]1", "benzimidazole", 1, 0),
        ("c1ncco1", "oxazole", 1, 0),
        ("c1nocc1", "isoxazole", 1, 0),
        ("c1cnsc1", "thiazole", 1, 0),
        ("c1cncs1", "isothiazole", 1, 0),
    ],
)
def test_conjugate_microstates_are_mapped_and_have_unit_charge_difference(
    smiles, label, acid_charge, base_charge
):
    result = _pair_result(smiles, label)
    assert result.status == "ok", result.note
    assert result.acid_charge == acid_charge
    assert result.base_charge == base_charge
    assert result.acid_charge - result.base_charge == 1
    assert result.site_atom_maps
    assert result.protonation_center_maps
    for state in result.acid_microstates + result.base_microstates:
        mol = Chem.MolFromSmiles(state)
        assert mol is not None
        heavy_maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        assert sorted(heavy_maps) == list(range(1, len(heavy_maps) + 1))


def test_tetrazole_retains_all_neutral_prototropic_tautomers():
    result = _pair_result("c1nnn[n-]1", "tetrazolate")
    assert result.status == "ok"
    assert len(result.acid_microstates) >= 2
    assert len(result.base_microstates) == 1


def test_n_oxide_protonates_the_oxide_oxygen_not_nitrogen():
    result = _pair_result("[O-][n+]1ccccc1", "n-oxide")
    assert result.status == "ok"
    assert result.protonation_center_maps == (1,)
    acid = Chem.MolFromSmiles(result.acid_microstates[0])
    oxygen = next(atom for atom in acid.GetAtoms() if atom.GetAtomMapNum() == 1)
    assert oxygen.GetAtomicNum() == 8
    assert oxygen.GetTotalNumHs() == 1


def test_pyrrole_uses_c2_protonation_not_n_protonation():
    result = _pair_result("c1cc[nH]c1", "pyrrole")
    assert result.status == "ok", result.note
    assert len(result.acid_microstates) == 2
    assert "alpha-carbon (C2/C5)" in result.note
    for smiles in result.acid_microstates:
        acid = Chem.MolFromSmiles(smiles)
        assert acid is not None
        positive_n = next(atom for atom in acid.GetAtoms() if atom.GetAtomicNum() == 7)
        assert positive_n.GetFormalCharge() == 1
        assert positive_n.GetTotalNumHs() == 1
        assert any(
            atom.GetAtomicNum() == 6 and atom.GetTotalNumHs() == 2
            for atom in acid.GetAtoms()
        )


def test_release_override_maps_fused_7_4_site_to_quinoline(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    smiles = "COc1cccc2c1nc(C)c1ccn(-c3ccccc3C)c12"
    writer = Chem.SDWriter(str(raw_dir / "toy.sdf"))
    writer.write(Chem.MolFromSmiles(smiles))
    writer.close()

    row = {
        "source_file": "toy.sdf", "record_index": 0, "smiles": smiles,
        "pka_value": 7.4, "pka_source_method": "experimental",
        "pka_type_raw": None, "pka_type_canonical": None,
        "atom_index_raw": None, "source_exact_duplicate_record_indices": "0",
    }
    assignments_path = tmp_path / "assignments.csv"
    pd.DataFrame([row]).to_csv(assignments_path, index=False)
    override_path = tmp_path / "override.csv"
    pd.DataFrame([{**row, "override_group_label": "quinoline"}]).to_csv(override_path, index=False)

    dataset, quarantine, _, _ = build_dataset(
        raw_dir=str(raw_dir), assignments_path=str(assignments_path),
        override_paths=[str(override_path)], output_path=str(tmp_path / "dataset.csv"),
        quarantine_path=str(tmp_path / "quarantine.csv"),
        clean_overrides_path=str(tmp_path / "clean.csv"),
        override_quarantine_path=str(tmp_path / "override_q.csv"), max_tautomers=4,
    )
    assert quarantine.empty
    assert len(dataset) == 1
    assert dataset.iloc[0]["site_label"] == "quinoline"
    assert dataset.iloc[0]["site_family"] == "amine"
    assert dataset.iloc[0]["structural_site_assignment_confidence"] == "high"
    assert dataset.iloc[0]["source_site_evidence"] == "unannotated_molecule_level_pka"
    assert dataset.iloc[0]["source_site_evidence_confidence"] == "none"
    assert json.loads(dataset.iloc[0]["protonation_center_atom_maps_json"]) == [9]
    reference_acid = Chem.MolFromSmiles(dataset.iloc[0]["reference_acid_atom_mapped_smiles"])
    protonated_n = next(
        atom for atom in reference_acid.GetAtoms()
        if atom.GetAtomMapNum() == 9
    )
    assert protonated_n.GetFormalCharge() == 1
    assert protonated_n.GetTotalNumHs() == 1


def test_joint_network_enumerates_two_independent_sites():
    mol = Chem.MolFromSmiles("NCC(=O)O")
    resolved, _ = resolve_overlapping_sites(find_sites_with_metadata(mol), 0.5)
    pair_sites = []
    for site in resolved:
        family = normalize_conjugate_family(site["label"])
        mode, member_form = classify_pair_type(site["label"], family)
        if mode == "pair_type" and member_form in {"acid_form", "base_form"}:
            pair_sites.append({**site, "pair_family": family, "pair_member_form": member_form})

    network = enumerate_joint_protonation_network(
        mol,
        pair_sites,
        max_protonation_states=16,
        max_tautomers_per_state=4,
    )
    assert len(network["sites"]) == 2
    assert len(network["nodes"]) == 4
    assert len(network["edges"]) == 4
    assert not network["protonation_state_enumeration_truncated"]
    assert not network["incomplete_site_ids"]
    node_ids = {node["node_id"] for node in network["nodes"]}
    for edge in network["edges"]:
        assert edge["acid_node_id"] in node_ids
        assert edge["base_node_id"] in node_ids
        assert edge["charge_delta"] == 1


@pytest.mark.parametrize(
    "smiles,label,expected_charges",
    [
        ("c1nnn[nH]1", "tetrazole", {-1, 0}),
        ("c1nnn[n-]1", "tetrazolate", {-1, 0}),
        ("[O-][n+]1ccccc1", "n-oxide", {0, 1}),
        ("C[N+](C)(C)[O-]", "amine_n_oxide", {0, 1}),
    ],
)
def test_joint_network_retains_explicit_detector_protonation_centres(
    smiles, label, expected_charges
):
    mol = Chem.MolFromSmiles(smiles)
    _, sites = _resolved_pair_sites(mol, 0.5)
    assert any(site["label"] == label for site in sites)

    network = enumerate_joint_protonation_network(mol, sites, 16, 2)

    assert {node["formal_charge"] for node in network["nodes"]} == expected_charges
    assert len(network["edges"]) == 1
    assert not network["incomplete_site_ids"]


def test_n_substituted_imidazole_does_not_invent_n_h_acidity():
    mol = Chem.MolFromSmiles("Cn1ccnc1")
    _, sites = _resolved_pair_sites(mol, 0.5)
    network = enumerate_joint_protonation_network(mol, sites, 16, 2)

    assert {site["label"] for site in network["sites"]} == {"imidazole_basicity"}
    assert {node["formal_charge"] for node in network["nodes"]} == {0, 1}
    assert not network["incomplete_site_ids"]


def test_fused_pyrrole_c_protonation_preserves_the_fused_aromatic_valence():
    mol = Chem.MolFromSmiles(
        "Cc1c[nH]c2ncnc(O[C@H]3CC[C@H](N4CCOCC4)CC3)c12"
    )
    _, sites = _resolved_pair_sites(mol, 0.5)
    network = enumerate_joint_protonation_network(mol, sites, 64, 2)

    assert any(site["label"] == "pyrrole" for site in network["sites"])
    assert not network["protonation_state_enumeration_truncated"]
    assert not network["incomplete_site_ids"]


@pytest.mark.parametrize(
    "smiles,expected_state_count",
    [
        (
            "COc1cccc(C(=O)Cn2cnc3c(C#N)c(N4CC[NH2+]CC4)n(CC=C(C)C)c3c2=O)c1",
            16,
        ),
        ("Cn1c(N)nc2cc[nH]c2c1=O", 8),
    ],
)
def test_fused_pyrrole_c_protonation_respects_exocyclic_carbonyl_valence(
    smiles, expected_state_count
):
    mol = Chem.MolFromSmiles(smiles)
    _, sites = _resolved_pair_sites(mol, 0.5)
    network = enumerate_joint_protonation_network(mol, sites, 64, 2)

    assert any(site["label"] == "pyrrole" for site in network["sites"])
    assert len(network["nodes"]) == expected_state_count
    assert not network["protonation_state_enumeration_truncated"]
    assert not network["incomplete_site_ids"]


def test_default_state_cap_covers_seven_independent_binary_sites():
    mol = Chem.MolFromSmiles("NCCNCCNCCNCCNCCNCCN")
    _, sites = _resolved_pair_sites(mol, 0.5)
    network = enumerate_joint_protonation_network(
        mol, sites, max_tautomers_per_state=1
    )

    assert len(network["sites"]) == 7
    assert len(network["nodes"]) == 128
    assert network["max_protonation_states"] == 4096
    assert not network["protonation_state_enumeration_truncated"]
    assert not network["incomplete_site_ids"]


@pytest.mark.parametrize(
    "smiles,expected_labels",
    [
        ("c1nn[nH]c1", {"123triazole_acidity"}),
        ("c1nnc[nH]1", {"124triazole_basicity", "124triazole_acidity"}),
        ("c1ccc2[nH]ncc2c1", {"indazole_basicity", "indazole_acidity"}),
        ("c1cn[nH]c1", {"pyrazole_basicity", "pyrazole_acidity"}),
        ("c1ncc[nH]1", {"imidazole_basicity", "imidazole_acidity"}),
    ],
)
def test_supported_azole_detector_sites_expand_to_transition_specific_edges(
    smiles, expected_labels
):
    mol = Chem.MolFromSmiles(smiles)
    resolved, _ = resolve_overlapping_sites(find_sites_with_metadata(mol), 0.5)
    sites = [transition for site in resolved for transition in expand_pair_site_transitions(site)]
    assert {site["label"] for site in sites} == expected_labels


def test_release_site_selection_uses_reported_type_for_amphoteric_ring():
    mol = Chem.MolFromSmiles("c1nnc[nH]1")
    _, sites = _resolved_pair_sites(mol, 0.5)
    acidic, _, _ = _select_site(pd.Series({"pka_type_canonical": "acidic"}), sites, None)
    basic, _, _ = _select_site(pd.Series({"pka_type_canonical": "basic"}), sites, None)
    assert acidic["label"] == "124triazole_acidity"
    assert basic["label"] == "124triazole_basicity"


@pytest.mark.parametrize(
    "smiles,pka_value,expected_label",
    [
        ("c1nn[nH]c1", 9.26, "123triazole_acidity"),
        ("c1nn[nH]c1", 1.17, None),
        ("c1nnc[nH]1", 2.3, "124triazole_basicity"),
        ("c1nnc[nH]1", 9.8, "124triazole_acidity"),
        ("c1nnc[nH]1", 6.1, None),
        ("c1cn[nH]c1", 2.49, "pyrazole_basicity"),
        ("c1cn[nH]c1", 14.21, "pyrazole_acidity"),
        ("c1cn[nH]c1", 8.0, None),
        ("c1ncc[nH]1", 7.0, "imidazole_basicity"),
        ("c1ncc[nH]1", 14.2, "imidazole_acidity"),
        ("c1ncc[nH]1", 10.5, None),
    ],
)
def test_untyped_azole_measurement_requires_reference_separated_transition(
    smiles, pka_value, expected_label
):
    mol = Chem.MolFromSmiles(smiles)
    resolved, sites = _resolved_pair_sites(mol, 0.5)

    release_site, _, confidence = _select_site(
        pd.Series({"pka_type_canonical": None, "pka_value": pka_value}),
        sites,
        None,
    )
    stage1_site, stage1_meta = select_pair_first_site(
        resolved,
        preferred_type=None,
        pka_value=pka_value,
    )

    assert (release_site["label"] if release_site else None) == expected_label
    assert (stage1_site["label"] if stage1_site else None) == expected_label
    if expected_label is None:
        assert confidence == "none"
        assert stage1_meta["pair_assignment_confidence"] == "none"
    else:
        assert confidence == "medium"
        assert stage1_meta["pair_assignment_confidence"] == "medium"


def test_legacy_azole_review_requires_explicit_transition_identity():
    mol = Chem.MolFromSmiles("c1nn[nH]c1")
    _, sites = _resolved_pair_sites(mol, 0.5)
    unverified = pd.Series({
        "reviewed_site_label": "123triazole",
        "pka_type_canonical": None,
        "review_verified_transition_identity": False,
    })
    verified = unverified.copy()
    verified["review_verified_transition_identity"] = True

    assert _select_reviewed_site(unverified, sites) is None
    assert _select_reviewed_site(verified, sites)["label"] == "123triazole_acidity"


def test_neutral_imidazole_site_review_does_not_choose_an_edge():
    mol = Chem.MolFromSmiles("c1ncc[nH]1")
    _, sites = _resolved_pair_sites(mol, 0.5)
    site_only = pd.Series({
        "reviewed_site_label": "imidazole",
        "reviewed_site_atom_indices_json": "[0,1,2,3,4]",
        "pka_type_canonical": None,
        "review_verified_transition_identity": False,
    })
    assert _select_reviewed_site(site_only, sites) is None
    explicit = site_only.copy()
    explicit["reviewed_site_label"] = "imidazole_acidity"
    selected = _select_reviewed_site(explicit, sites)
    assert selected["label"] == "imidazole_acidity"
    assert _review_explicitly_selects_transition(explicit, selected)


@pytest.mark.parametrize(
    "smiles,transition_label,acid_charge,base_charge",
    [
        ("c1nn[nH]c1", "123triazole_acidity", 0, -1),
        ("c1nnc[nH]1", "124triazole_basicity", 1, 0),
        ("c1nnc[nH]1", "124triazole_acidity", 0, -1),
        ("c1ccc2[nH]ncc2c1", "indazole_basicity", 1, 0),
        ("c1ccc2[nH]ncc2c1", "indazole_acidity", 0, -1),
        ("c1cn[nH]c1", "pyrazole_basicity", 1, 0),
        ("c1cn[nH]c1", "pyrazole_acidity", 0, -1),
        ("c1ncc[nH]1", "imidazole_basicity", 1, 0),
        ("c1ncc[nH]1", "imidazole_acidity", 0, -1),
    ],
)
def test_supported_azole_transition_microstates_have_unit_charge_difference(
    smiles, transition_label, acid_charge, base_charge
):
    mol = Chem.MolFromSmiles(smiles)
    resolved, _ = resolve_overlapping_sites(find_sites_with_metadata(mol), 0.5)
    sites = [transition for site in resolved for transition in expand_pair_site_transitions(site)]
    site = next(candidate for candidate in sites if candidate["label"] == transition_label)
    result = generate_conjugate_microstates(
        mol, site, site["pair_family"], site["pair_member_form"], max_tautomers=8
    )
    assert result.status == "ok", result.note
    assert result.acid_charge == acid_charge
    assert result.base_charge == base_charge


@pytest.mark.parametrize(
    "smiles,pka_map",
    [
        ("c1nnc[nH]1", {"124triazole_basicity": 2.3, "124triazole_acidity": 9.8}),
        ("c1ccc2[nH]ncc2c1", {"indazole_basicity": 1.0, "indazole_acidity": 13.9}),
        ("c1cn[nH]c1", {"pyrazole_basicity": 2.49, "pyrazole_acidity": 14.21}),
        ("c1ncc[nH]1", {"imidazole_basicity": 7.0, "imidazole_acidity": 14.2}),
    ],
)
def test_amphoteric_azole_network_has_three_serial_charge_levels(smiles, pka_map):
    mol = Chem.MolFromSmiles(smiles)
    resolved, _ = resolve_overlapping_sites(find_sites_with_metadata(mol), 0.5)
    sites = [transition for site in resolved for transition in expand_pair_site_transitions(site)]
    network = enumerate_joint_protonation_network(mol, sites, 32, 4)
    assert {node["formal_charge"] for node in network["nodes"]} == {-1, 0, 1}
    assert {node["protonated_site_count"] for node in network["nodes"]} == {0, 1, 2}
    assert len({site["coupling_group_id"] for site in network["sites"]}) == 1
    site_ids = {site["label"]: site["site_id"] for site in network["sites"]}
    values = {site_ids[label]: value for label, value in pka_map.items()}
    steps, _ = coupled_network_thermodynamics(network, values, ph=7.4)
    assert [step["predicted_macro_pka"] for step in steps] == pytest.approx(
        sorted(pka_map.values(), reverse=True)
    )
    neutral_population = sum(
        node["network_population_at_ph"]
        for node in network["nodes"]
        if node["formal_charge"] == 0
    )
    assert neutral_population > 0.5


def test_stage1_loader_can_exclude_marvin_rows(tmp_path):
    from train_intrinsic_single_group_model import load_single_group_rows

    rows = []
    for method, pka in (("experimental", 4.7), ("marvin", 5.1)):
        rows.append({
            "source_file": "toy.sdf",
            "record_index": 0,
            "smiles": "CC(=O)O",
            "pka_value": pka,
            "pka_source_method": method,
            "final_group_label": "carboxylic_acid",
            "assignment_status": "single_group",
            "single_group_strict": True,
            "pka_type_canonical": "acidic",
            "distribution_ok": True,
        })
    path = tmp_path / "assignments.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    filtered = load_single_group_rows(str(path), measurement_method="experimental")
    assert len(filtered) == 1
    assert filtered.iloc[0]["pka_source_method"] == "experimental"


def test_independent_site_macro_pka_includes_statistical_factor():
    steps, site_map = independent_site_macro_pkas([("site_1", 10.0), ("site_2", 10.0)])
    assert len(steps) == 2
    assert steps[0]["predicted_macro_pka"] == pytest.approx(10.30103, abs=1e-5)
    assert steps[1]["predicted_macro_pka"] == pytest.approx(9.69897, abs=1e-5)
    assert set(site_map) == {"site_1", "site_2"}


def test_molecule_grouping_merges_protonation_but_keeps_stereochemistry():
    assert _molecule_group_key("CC(=O)O") == _molecule_group_key("CC(=O)[O-]")
    assert _molecule_group_key("C[C@H](N)C(=O)O") != _molecule_group_key("C[C@@H](N)C(=O)O")


def test_element_topology_mapping_ignores_protonation_charge():
    neutral = Chem.MolFromSmiles("CN(C)CCc1ccccc1")
    protonated = Chem.MolFromSmiles("C[NH+](C)CCc1ccccc1")
    assert neutral.HasSubstructMatch(_element_topology_query(protonated))


def test_release_builder_quarantines_ambiguous_macro_pka(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sdf_path = raw_dir / "toy.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    for smiles in ("CC(=O)O", "NCC(=O)O"):
        writer.write(Chem.MolFromSmiles(smiles))
    writer.close()

    assignments = pd.DataFrame(
        [
            {
                "source_file": "toy.sdf", "record_index": 0, "smiles": "CC(=O)O",
                "pka_value": 4.76, "pka_source_method": "experimental",
                "pka_type_raw": "acidic", "pka_type_canonical": "acidic",
                "atom_index_raw": None, "source_exact_duplicate_record_indices": "0",
            },
            {
                "source_file": "toy.sdf", "record_index": 1, "smiles": "NCC(=O)O",
                "pka_value": 9.6, "pka_source_method": "experimental",
                "pka_type_raw": None, "pka_type_canonical": None,
                "atom_index_raw": None, "source_exact_duplicate_record_indices": "1",
            },
            {
                "source_file": "toy.sdf", "record_index": 0, "smiles": "CC(=O)O",
                "pka_value": 99.0, "pka_source_method": "marvin",
                "pka_type_raw": "acidic", "pka_type_canonical": "acidic",
                "atom_index_raw": 2, "source_exact_duplicate_record_indices": "0",
            },
        ]
    )
    assignments_path = tmp_path / "assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    dataset, quarantine, _, _ = build_dataset(
        raw_dir=str(raw_dir),
        assignments_path=str(assignments_path),
        override_paths=[],
        output_path=str(tmp_path / "dataset.csv"),
        quarantine_path=str(tmp_path / "quarantine.csv"),
        clean_overrides_path=str(tmp_path / "overrides.csv"),
        override_quarantine_path=str(tmp_path / "override_quarantine.csv"),
        max_tautomers=4,
    )
    assert len(dataset) == 1
    assert dataset.iloc[0]["measurement_method"] == "experimental"
    assert not bool(dataset.iloc[0]["marvin_values_used"])
    assert json.loads(dataset.iloc[0]["site_atom_maps_json"])
    assert set(quarantine["quarantine_reason"]) == {"multiple_pair_sites_without_override"}


def test_release_builder_revalidates_override_and_selects_one_site(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    writer = Chem.SDWriter(str(raw_dir / "toy.sdf"))
    writer.write(Chem.MolFromSmiles("NCC(=O)O"))
    writer.close()

    row = {
        "source_file": "toy.sdf", "record_index": 0, "smiles": "NCC(=O)O",
        "pka_value": 9.6, "pka_source_method": "experimental",
        "pka_type_raw": None, "pka_type_canonical": None,
        "atom_index_raw": None, "source_exact_duplicate_record_indices": "0",
    }
    assignments_path = tmp_path / "assignments.csv"
    pd.DataFrame([row]).to_csv(assignments_path, index=False)
    override_path = tmp_path / "overrides_part.csv"
    pd.DataFrame([{**row, "override_group_label": "primary_amine", "note": "manual"}]).to_csv(
        override_path, index=False
    )

    dataset, quarantine, clean, override_quarantine = build_dataset(
        raw_dir=str(raw_dir),
        assignments_path=str(assignments_path),
        override_paths=[str(override_path)],
        output_path=str(tmp_path / "dataset.csv"),
        quarantine_path=str(tmp_path / "quarantine.csv"),
        clean_overrides_path=str(tmp_path / "clean.csv"),
        override_quarantine_path=str(tmp_path / "override_quarantine.csv"),
        max_tautomers=4,
    )
    assert len(dataset) == 1
    assert dataset.iloc[0]["site_label"] == "primary_amine"
    assert bool(dataset.iloc[0]["override_applied"])
    assert quarantine.empty
    assert len(clean) == 1
    assert override_quarantine.empty


def test_release_builder_uses_reviewer_atom_index_for_repeated_same_label_sites(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    writer = Chem.SDWriter(str(raw_dir / "toy.sdf"))
    writer.write(Chem.MolFromSmiles("Oc1ccccc1O"))
    writer.close()

    row = {
        "source_file": "toy.sdf", "record_index": 0, "smiles": "Oc1ccccc1O",
        "pka_value": 9.8, "pka_source_method": "experimental",
        "pka_type_raw": None, "pka_type_canonical": None,
        "atom_index_raw": None, "source_exact_duplicate_record_indices": "0",
    }
    assignments_path = tmp_path / "assignments.csv"
    pd.DataFrame([row]).to_csv(assignments_path, index=False)
    override_path = tmp_path / "overrides_part.csv"
    pd.DataFrame([{
        **row,
        "override_group_label": "phenol",
        "override_atom_index_raw": 0,
        "note": "manual repeated-site selection",
    }]).to_csv(override_path, index=False)

    dataset, quarantine, clean, override_quarantine = build_dataset(
        raw_dir=str(raw_dir),
        assignments_path=str(assignments_path),
        override_paths=[str(override_path)],
        output_path=str(tmp_path / "dataset.csv"),
        quarantine_path=str(tmp_path / "quarantine.csv"),
        clean_overrides_path=str(tmp_path / "clean.csv"),
        override_quarantine_path=str(tmp_path / "override_quarantine.csv"),
        max_tautomers=4,
    )

    assert len(dataset) == 1
    assert dataset.iloc[0]["site_label"] == "phenol"
    assert json.loads(dataset.iloc[0]["protonation_center_atom_maps_json"]) == [1]
    assert dataset.iloc[0]["override_atom_index_raw"] == 0
    assert quarantine.empty
    assert len(clean) == 1
    assert override_quarantine.empty


def test_network_mapping_preserves_reviewer_selected_atom_for_symmetric_sites():
    row = pd.Series({
        "site_family": "phenol_phenolate",
        "override_atom_index_raw": 16,
        "site_atom_maps_json": "[16,17]",
    })
    network_sites = [
        {
            "site_id": "site_2", "family": "phenol_phenolate",
            "atom_maps": [10, 11],
        },
        {
            "site_id": "site_4", "family": "phenol_phenolate",
            "atom_maps": [16, 17],
        },
    ]

    site_id, basis = _map_measurement_to_site(
        row,
        Chem.MolFromSmiles("Oc1cc(O)cc(O)c1"),
        network_sites,
    )

    assert site_id == "site_4"
    assert basis == "reviewer_selected_atom_map_on_representative"


def test_new_detector_family_requires_manual_review_before_stage1(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    writer = Chem.SDWriter(str(raw_dir / "toy.sdf"))
    writer.write(Chem.MolFromSmiles("CCO"))
    writer.close()
    row = {
        "source_file": "toy.sdf", "record_index": 0, "smiles": "CCO",
        "pka_value": 16.0, "pka_source_method": "experimental",
        "pka_type_raw": None, "pka_type_canonical": "acidic",
        "atom_index_raw": None, "source_exact_duplicate_record_indices": "0",
    }
    assignments_path = tmp_path / "assignments.csv"
    pd.DataFrame([row]).to_csv(assignments_path, index=False)

    kwargs = dict(
        raw_dir=str(raw_dir), assignments_path=str(assignments_path), override_paths=[],
        output_path=str(tmp_path / "dataset.csv"),
        quarantine_path=str(tmp_path / "quarantine.csv"),
        clean_overrides_path=str(tmp_path / "clean.csv"),
        override_quarantine_path=str(tmp_path / "override_q.csv"), max_tautomers=4,
    )
    dataset, quarantine, _, _ = build_dataset(**kwargs)
    assert dataset.empty
    assert quarantine.iloc[0]["quarantine_reason"] == "detector_v2_new_family_requires_manual_review"

    decisions_path = tmp_path / "decisions.csv"
    pd.DataFrame([{**row,
        "review_decision": "confirm_assignment",
        "reviewed_site_label": "primary_alcohol",
        "reviewed_site_atom_indices_json": "[1,2]",
        "review_note": "checked", "reviewed_at": "2026-09-04T00:00:00Z",
    }]).to_csv(decisions_path, index=False)
    dataset, quarantine, _, _ = build_dataset(
        **kwargs, outlier_review_decisions_path=str(decisions_path)
    )
    assert len(dataset) == 1
    assert quarantine.empty
    assert dataset.iloc[0]["site_family"] == "alcohol_alkoxide"


def test_release_builder_enforces_outlier_review_reassignment_and_exclusion(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    writer = Chem.SDWriter(str(raw_dir / "toy.sdf"))
    writer.write(Chem.MolFromSmiles("NCC(=O)O"))
    writer.write(Chem.MolFromSmiles("Oc1ccccc1"))
    writer.close()

    rows = [
        {
            "source_file": "toy.sdf", "record_index": 0, "smiles": "NCC(=O)O",
            "pka_value": 4.2, "pka_source_method": "experimental",
            "pka_type_raw": None, "pka_type_canonical": None,
            "atom_index_raw": None, "source_exact_duplicate_record_indices": "0",
        },
        {
            "source_file": "toy.sdf", "record_index": 1, "smiles": "Oc1ccccc1",
            "pka_value": 16.0, "pka_source_method": "experimental",
            "pka_type_raw": None, "pka_type_canonical": None,
            "atom_index_raw": None, "source_exact_duplicate_record_indices": "1",
        },
    ]
    assignments_path = tmp_path / "assignments.csv"
    pd.DataFrame(rows).to_csv(assignments_path, index=False)
    decisions_path = tmp_path / "outlier_decisions.csv"
    pd.DataFrame([
        {
            **rows[0], "review_decision": "reassign_site",
            "reviewed_site_label": "carboxylic_acid",
            "reviewed_site_atom_indices_json": "[2,3,4]",
            "review_note": "manual exact site", "reviewed_at": "2026-07-22T00:00:00Z",
        },
        {
            **rows[1], "review_decision": "exclude_implausible_pka",
            "reviewed_site_label": "", "reviewed_site_atom_indices_json": "[]",
            "review_note": "not a credible aqueous phenol pKa",
            "reviewed_at": "2026-07-22T00:00:01Z",
        },
    ]).to_csv(decisions_path, index=False)

    dataset, quarantine, _, _ = build_dataset(
        raw_dir=str(raw_dir), assignments_path=str(assignments_path), override_paths=[],
        output_path=str(tmp_path / "dataset.csv"),
        quarantine_path=str(tmp_path / "quarantine.csv"),
        clean_overrides_path=str(tmp_path / "clean.csv"),
        override_quarantine_path=str(tmp_path / "override_q.csv"),
        outlier_review_decisions_path=str(decisions_path), max_tautomers=4,
    )
    assert len(dataset) == 1
    accepted = dataset.iloc[0]
    assert accepted["site_label"] == "carboxylic_acid"
    assert accepted["site_assignment_basis"] == "manual_outlier_review_exact_site"
    assert accepted["outlier_review_decision"] == "reassign_site"
    assert len(quarantine) == 1
    rejected = quarantine.iloc[0]
    assert rejected["record_index"] == 1
    assert rejected["quarantine_reason"] == "manual_outlier_review:exclude_implausible_pka"
    assert rejected["quarantine_detail"] == "not a credible aqueous phenol pKa"


def test_confirmed_outlier_review_site_remains_atom_exact(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    writer = Chem.SDWriter(str(raw_dir / "toy.sdf"))
    writer.write(Chem.MolFromSmiles("NCCN"))
    writer.close()
    row = {
        "source_file": "toy.sdf", "record_index": 0, "smiles": "NCCN",
        "pka_value": 9.0, "pka_source_method": "experimental",
        "pka_type_raw": None, "pka_type_canonical": None,
        "atom_index_raw": None, "source_exact_duplicate_record_indices": "0",
    }
    assignments_path = tmp_path / "assignments.csv"
    pd.DataFrame([row]).to_csv(assignments_path, index=False)
    decisions_path = tmp_path / "outlier_decisions.csv"
    pd.DataFrame([{
        **row, "review_decision": "confirm_assignment",
        "reviewed_site_label": "primary_amine",
        "reviewed_site_atom_indices_json": "[0,1]",
        "review_note": "confirmed left-hand nitrogen",
        "reviewed_at": "2026-07-22T00:00:00Z",
    }]).to_csv(decisions_path, index=False)

    dataset, quarantine, _, _ = build_dataset(
        raw_dir=str(raw_dir), assignments_path=str(assignments_path), override_paths=[],
        output_path=str(tmp_path / "dataset.csv"),
        quarantine_path=str(tmp_path / "quarantine.csv"),
        clean_overrides_path=str(tmp_path / "clean.csv"),
        override_quarantine_path=str(tmp_path / "override_q.csv"),
        outlier_review_decisions_path=str(decisions_path), max_tautomers=4,
    )
    assert quarantine.empty
    assert len(dataset) == 1
    accepted = dataset.iloc[0]
    assert accepted["site_assignment_basis"] == "manual_outlier_review_confirmed_exact_site"
    assert json.loads(accepted["site_atom_maps_json"]) == [1, 2]


def test_release_builder_applies_assay_exclusion_and_unannotated_amine_guard(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    writer = Chem.SDWriter(str(raw_dir / "toy.sdf"))
    excluded = Chem.MolFromSmiles("Oc1ccccc1")
    excluded.SetProp("assay_web_id", "ASSAY_BAD")
    writer.write(excluded)
    writer.write(Chem.MolFromSmiles("CN(C)C"))
    writer.close()

    rows = [
        {
            "source_file": "toy.sdf", "record_index": 0, "smiles": "Oc1ccccc1",
            "pka_value": 16.0, "pka_source_method": "experimental",
            "pka_type_raw": None, "pka_type_canonical": None,
            "atom_index_raw": None, "site_identifier_raw": None,
            "source_exact_duplicate_record_indices": "0",
        },
        {
            "source_file": "toy.sdf", "record_index": 1, "smiles": "CN(C)C",
            "pka_value": 14.59, "pka_source_method": "experimental",
            "pka_type_raw": None, "pka_type_canonical": None,
            "atom_index_raw": None, "site_identifier_raw": None,
            "source_exact_duplicate_record_indices": "1",
        },
    ]
    assignments_path = tmp_path / "assignments.csv"
    pd.DataFrame(rows).to_csv(assignments_path, index=False)
    exclusions_path = tmp_path / "exclusions.csv"
    pd.DataFrame([{
        "scope": "assay", "source_file": "toy.sdf", "assay_id": "ASSAY_BAD",
        "record_index": None, "pka_value": None,
        "quarantine_reason": "wrong_endpoint", "quarantine_detail": "test",
        "evidence_url": "https://example.invalid/evidence",
    }]).to_csv(exclusions_path, index=False)

    dataset, quarantine, _, _ = build_dataset(
        raw_dir=str(raw_dir), assignments_path=str(assignments_path), override_paths=[],
        output_path=str(tmp_path / "dataset.csv"),
        quarantine_path=str(tmp_path / "quarantine.csv"),
        clean_overrides_path=str(tmp_path / "clean.csv"),
        override_quarantine_path=str(tmp_path / "override_q.csv"),
        measurement_exclusions_path=str(exclusions_path), max_tautomers=2,
    )
    assert dataset.empty
    assert set(quarantine["quarantine_reason"]) == {
        "wrong_endpoint", "unannotated_amine_pka_above_plausibility_guard",
    }
    excluded_row = quarantine[quarantine["quarantine_reason"] == "wrong_endpoint"].iloc[0]
    assert excluded_row["source_assay_id"] == "ASSAY_BAD"


def test_molecule_builder_keeps_measured_anchor_and_predicts_other_site(tmp_path, monkeypatch):
    import build_molecule_microstate_network_dataset as network_builder

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    writer = Chem.SDWriter(str(raw_dir / "toy.sdf"))
    writer.write(Chem.MolFromSmiles("NCC(=O)O"))
    writer.close()
    assignment = {
        "source_file": "toy.sdf", "record_index": 0, "smiles": "NCC(=O)O",
        "pka_value": 9.6, "pka_source_method": "experimental",
        "pka_type_raw": None, "pka_type_canonical": None,
        "atom_index_raw": None, "source_exact_duplicate_record_indices": "0",
    }
    assignments_path = tmp_path / "assignments.csv"
    pd.DataFrame([assignment]).to_csv(assignments_path, index=False)
    override_path = tmp_path / "override.csv"
    pd.DataFrame([{**assignment, "override_group_label": "primary_amine"}]).to_csv(override_path, index=False)
    transition_path = tmp_path / "transitions.csv"
    build_dataset(
        raw_dir=str(raw_dir), assignments_path=str(assignments_path), override_paths=[str(override_path)],
        output_path=str(transition_path), quarantine_path=str(tmp_path / "transition_q.csv"),
        clean_overrides_path=str(tmp_path / "clean.csv"),
        override_quarantine_path=str(tmp_path / "override_q.csv"), max_tautomers=2,
    )

    model_path = tmp_path / "stage1.pkl"
    with open(model_path, "wb") as handle:
        pickle.dump({
            "training_measurement_method": "experimental",
            "training_rows": 10,
            "feature_columns": [],
            "model": None,
            "fp_bits": 32,
            "fp_radius": 2,
        }, handle)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"scaffold": {"mae": 1.5}}))
    monkeypatch.setattr(
        network_builder,
        "predict_stage1_tasks",
        lambda tasks, bundle_path, workers: {task["task_id"]: 4.0 for task in tasks},
    )
    molecules, quarantine = network_builder.build_molecule_dataset(
        transitions_path=str(transition_path), stage1_model_path=str(model_path),
        stage1_metrics_path=str(metrics_path), output_path=str(tmp_path / "molecules.csv"),
        quarantine_path=str(tmp_path / "molecule_q.csv"), stage1_workers=1,
        max_tautomers_per_state=2,
    )
    assert quarantine.empty
    assert len(molecules) == 1
    row = molecules.iloc[0]
    assert row["detected_site_count"] == 2
    assert row["experimentally_anchored_site_count"] == 1
    assert row["stage1_prior_only_site_count"] == 1
    assert row["protonation_state_count"] == 4
    sites = json.loads(row["sites_json"])
    provenance = {site["family"]: site["local_pka_provenance"] for site in sites}
    assert provenance["amine"] == "stage1_reference_prior_zero_shot_exact_label_fallback"
    assert provenance["carboxyl"] == "stage1_reference_prior_zero_shot_exact_label_fallback"
    amine = next(site for site in sites if site["family"] == "amine")
    assert amine["experimental_anchor_pka"] == pytest.approx(9.6)
