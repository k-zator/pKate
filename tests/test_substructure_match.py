"""
Substructure Matching Tests
============================
Tests for the core substructure matching pipeline: site detection,
overlap resolution, label assignment, and determinism.
"""

import pytest
from rdkit import Chem

from substructure_match import (
    find_sites_with_metadata,
    resolve_overlapping_sites,
    assign_single_group_label,
    maximal_site_rank_key,
    extract_site_subgraph,
    LABEL_PRIORITY,
    LABEL_FAMILY,
    HIGH_PROFILE_FAMILIES,
    get_default_pattern_specs,
)


# ===================================================================
# Helpers
# ===================================================================

def _resolve(smiles, threshold=0.75):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Cannot parse: {smiles}"
    candidates = find_sites_with_metadata(mol)
    accepted, rejected = resolve_overlapping_sites(candidates, overlap_threshold=threshold)
    return accepted, rejected


def _labels(sites):
    return sorted(s["label"] for s in sites)


# ===================================================================
# 1. find_sites_with_metadata returns proper dict structure
# ===================================================================

def test_find_sites_returns_required_keys():
    """Every returned site dict must have the full set of metadata keys."""
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    candidates = find_sites_with_metadata(mol)
    assert len(candidates) > 0, "Aspirin should produce at least one candidate"
    required_keys = {"label", "atoms", "atom_set", "specificity", "priority", "family", "type", "smarts", "smarts_atoms"}
    for cand in candidates:
        missing = required_keys - set(cand.keys())
        assert not missing, f"Site {cand['label']} missing keys: {missing}"


def test_find_sites_metadata_types():
    """Verify types of metadata fields."""
    mol = Chem.MolFromSmiles("CC(=O)O")
    candidates = find_sites_with_metadata(mol)
    for cand in candidates:
        assert isinstance(cand["label"], str)
        assert isinstance(cand["atoms"], tuple)
        assert isinstance(cand["atom_set"], set)
        assert isinstance(cand["specificity"], int)
        assert isinstance(cand["priority"], int)
        assert isinstance(cand["family"], str)


# ===================================================================
# 2. Overlap resolution logic
# ===================================================================

def test_overlap_resolution_keeps_both_disjoint_groups():
    """p-aminobenzoic acid has amine + carboxylic acid on different atoms."""
    accepted, _ = _resolve("Nc1ccc(C(=O)O)cc1")
    labels = _labels(accepted)
    # Must find both groups — they are on different atoms
    assert "carboxylic_acid" in labels, f"Expected carboxylic_acid in {labels}"
    # The amine should be detected as aniline (aromatic N) or primary_amine
    has_n_group = any(l in labels for l in ("aniline", "primary_amine"))
    assert has_n_group, f"Expected an amine-type group in {labels}"


def test_carbon_boundary_overlap_does_not_remove_exocyclic_aniline():
    accepted, _ = _resolve("Nc1ccc2cc3c(ccc4ccccc43)nc2c1", threshold=0.5)
    labels = _labels(accepted)
    assert "aniline" in labels
    assert any(label in labels for label in ("quinoline", "pyridine"))


def test_anilide_nitrogen_is_not_resolved_as_aniline():
    accepted, _ = _resolve("CC(=O)Nc1ccccc1", threshold=0.5)
    assert "aniline" not in _labels(accepted)


def test_aryl_sulfonamide_nitrogen_is_not_resolved_as_aniline():
    accepted, _ = _resolve("CS(=O)(=O)Nc1ccccc1", threshold=0.5)
    assert "aniline" not in _labels(accepted)


def test_same_family_overlap_strict_threshold():
    """Two carboxyl groups on separate atoms should both survive.
    But carboxylic_acid and carboxylate on SAME atoms should not."""
    # Malonic acid: two separate -COOH groups
    accepted, _ = _resolve("OC(=O)CC(=O)O")
    carboxyl_count = sum(1 for s in accepted if s["label"] in ("carboxylic_acid", "carboxylate"))
    assert carboxyl_count >= 2, (
        f"Malonic acid should keep both carboxyl groups, found {carboxyl_count}"
    )


def test_high_profile_family_preferential_treatment():
    """Amide atoms should not be stolen by a lower-ranked pattern."""
    # Acetamide — amide should win over amine-like matches on the N
    accepted, rejected = _resolve("CC(=O)N")
    labels = _labels(accepted)
    assert "amide_primary" in labels, f"Amide must survive overlap resolution. Got: {labels}"
    # primary_amine should NOT match due to !$([N][C](=O)) exclusion
    assert "primary_amine" not in labels, f"primary_amine should be excluded by SMARTS on amide"


def test_subset_atoms_trigger_conflict():
    """When one site's atoms are a strict subset of another, the smaller is rejected."""
    # Acetic acid: carboxylate C(=O)[O-] atoms ⊂ carboxylic_acid C(=O)OH atoms
    # are on the same atoms — the higher ranked should win
    mol = Chem.MolFromSmiles("CC(=O)[O-]")
    candidates = find_sites_with_metadata(mol)
    accepted, rejected = resolve_overlapping_sites(candidates, overlap_threshold=0.75)
    labels = _labels(accepted)
    # carboxylate should win (higher priority 360 > anything else on same atoms)
    assert "carboxylate" in labels, f"carboxylate should survive. Got: {labels}"


# ===================================================================
# 3. assign_single_group_label
# ===================================================================

def test_single_group_acetic_acid():
    accepted, _ = _resolve("CC(=O)O")
    label = assign_single_group_label(accepted)
    assert label == "carboxylic_acid", f"Acetic acid single-group label should be carboxylic_acid, got {label}"


def test_single_group_methylamine():
    accepted, _ = _resolve("CN")
    label = assign_single_group_label(accepted)
    assert label == "primary_amine", f"Methylamine single-group label should be primary_amine, got {label}"


def test_single_group_phenol():
    accepted, _ = _resolve("Oc1ccccc1")
    label = assign_single_group_label(accepted)
    assert label == "phenol", f"Phenol single-group label should be phenol, got {label}"


def test_phenol_and_phenolate_are_not_labeled_as_ether():
    for smiles, expected in (("Oc1ccccc1", "phenol"), ("[O-]c1ccccc1", "phenolate")):
        accepted, _ = _resolve(smiles, threshold=0.5)
        labels = _labels(accepted)
        assert expected in labels
        assert "ether" not in labels


def test_nitro_survives_beside_aromatic_heterocycle_without_false_amine():
    accepted, _ = _resolve("O=[N+]([O-])c1ncc[nH]1", threshold=0.5)
    labels = _labels(accepted)
    assert "nitro" in labels
    assert "aniline" not in labels
    assert "tertiary_amine" not in labels


def test_nitro_survives_on_alkene_without_false_enamine():
    accepted, _ = _resolve("C=C[N+](=O)[O-]", threshold=0.5)
    labels = _labels(accepted)
    assert "nitro" in labels
    assert "enamine" not in labels


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("c1nn[nH]c1", "123triazole"),
        ("c1nn[n-]c1", "123triazolate_alt3"),
        ("c1nnc[nH]1", "124triazole"),
        ("c1n[nH+]c[nH]1", "124triazolium_alt2"),
        ("c1nnc[n-]1", "124triazolate_alt3"),
        ("c1ccc2[nH]ncc2c1", "indazole"),
        ("c1ccc2[n-]ncc2c1", "indazolate"),
        ("c1ccc2[nH][nH+]cc2c1", "indazolium_alt"),
        ("c1cn[n-]c1", "pyrazolate_alt"),
        ("c1c[n-]cn1", "imidazolate_alt"),
    ],
)
def test_supported_azole_charge_forms_are_structurally_distinguished(smiles, expected):
    accepted, _ = _resolve(smiles, threshold=0.5)
    assert expected in _labels(accepted)


def test_phosphate_acid_sites_and_ester_linkage_are_recognized():
    accepted, _ = _resolve("COP(=O)(O)O", threshold=0.5)
    labels = _labels(accepted)
    assert labels.count("phosphoric_acid") == 2
    assert "phosphate_ester" in labels


@pytest.mark.parametrize(
    "smiles,label",
    [
        ("CO", "primary_alcohol"),
        ("OC1NCCC1", "secondary_alcohol"),
        ("CC[O-]", "alkoxide"),
    ],
)
def test_expanded_alcohol_and_alkoxide_recognition(smiles, label):
    accepted, _ = _resolve(smiles, threshold=0.5)
    assert label in _labels(accepted)


def test_single_group_empty():
    label = assign_single_group_label([])
    assert label is None


# ===================================================================
# 4. LABEL_PRIORITY ordering
# ===================================================================

def test_priority_carboxylate_highest():
    assert LABEL_PRIORITY["carboxylate"] > LABEL_PRIORITY["carboxylic_acid"]


def test_priority_carboxylic_acid_above_phenol():
    assert LABEL_PRIORITY["carboxylic_acid"] > LABEL_PRIORITY["phenol"]


def test_priority_phenol_above_amine():
    assert LABEL_PRIORITY["phenol"] > LABEL_PRIORITY["primary_amine"]


def test_priority_amide_above_ester():
    assert LABEL_PRIORITY["amide_primary"] > LABEL_PRIORITY["ester"]


def test_priority_ammonium_above_amine():
    assert LABEL_PRIORITY["tertiary_ammonium"] > LABEL_PRIORITY["tertiary_amine"]


@pytest.mark.parametrize(
    "smiles,label",
    [("CC1(C)CN1", "aziridine"), ("CC1(C)C[NH2+]1", "aziridinium")],
)
def test_aziridine_subtype_survives_generic_amine_overlap(smiles, label):
    accepted, _ = _resolve(smiles)
    matches = [site for site in accepted if site["label"] == label]
    assert matches, _labels(accepted)
    assert matches[0]["family"] == "aziridine"


@pytest.mark.parametrize(
    "smiles,label,family",
    [
        ("CNN", "hydrazine", "hydrazine_like"),
        ("NNc1ccccc1", "conjugated_hydrazine", "hydrazine_like"),
        ("C[NH2+]N", "hydrazinium", "hydrazine_like"),
        ("CON", "hydroxylamine", "hydroxylamine_like"),
        ("CO[NH3+]", "hydroxylammonium", "hydroxylamine_like"),
        ("C[OH2+]", "protonated_alcohol", "alcohol_oxonium"),
        ("C[SH2+]", "protonated_thiol", "thiol_thiolium"),
        ("C[S+]([O-])C", "sulfoxide", "sulfoxide"),
        ("C[NH2+]c1ccccc1", "secondary_ammonium", "amine"),
    ],
)
def test_detector_2_1_charged_and_nn_motifs_survive_resolution(smiles, label, family):
    accepted, _ = _resolve(smiles, threshold=0.5)
    matching = [site for site in accepted if site["label"] == label]
    assert matching, (smiles, _labels(accepted))
    assert matching[0]["family"] == family


def test_charge_separated_aryl_sulfoxide_is_not_stolen_by_thiophenol():
    accepted, _ = _resolve("c1ccccc1[S+](C)[O-]", threshold=0.5)
    labels = _labels(accepted)
    assert "sulfoxide" in labels
    assert "thiophenol" not in labels


@pytest.mark.parametrize(
    "smiles,label,family",
    [
        ("C[N+](C)(C)C", "quaternary_ammonium", "permanent_ammonium"),
        ("[O-][n+]1ccccc1", "n-oxide", "n_oxide"),
        ("C[N+](C)(C)[O-]", "amine_n_oxide", "amine_n_oxide"),
        ("c1ccnnc1", "pyridazine", "pyridazine"),
        ("c1nnco1", "134_oxadiazole", "oxadiazole"),
        ("c1cnsn1", "125_thiadiazole", "thiadiazole"),
    ],
)
def test_detector_2_2_missing_nitrogen_chemistry(smiles, label, family):
    accepted, _ = _resolve(smiles, threshold=0.5)
    site = next((value for value in accepted if value["label"] == label), None)
    assert site is not None, (smiles, _labels(accepted))
    assert site["family"] == family


def test_n_oxide_has_explicit_oxygen_protonation_center():
    accepted, _ = _resolve("[O-][n+]1ccccc1", threshold=0.5)
    site = next(value for value in accepted if value["label"] == "n-oxide")
    center = next(iter(site["center_atom_set"]))
    assert site["atom_set"] == {0, 1}
    assert center == 0


def test_unmatched_nitrogen_receives_quarantinable_fallback():
    accepted, _ = _resolve("N=P(F)(F)F", threshold=0.5)
    fallback = [value for value in accepted if value.get("is_fallback")]
    assert len(fallback) == 1
    assert fallback[0]["label"] == "unclassified_nonring_N"
    assert fallback[0]["ionization_role"] == "unclassified_requires_quarantine"


@pytest.mark.parametrize(
    "smiles,expected_label,covered_nitrogens",
    [
        ("NS(=O)(=O)c1ccccc1", "sulfonamide", 1),
        ("CCNN(C)C", "hydrazine", 2),
        ("[N-]=[N+]=N", "azide", 3),
        ("NS(=O)(=O)O", "sulfamoyl_nitrogen", 1),
        ("NP(=O)(O)O", "phosphoramidate_nitrogen", 1),
    ],
)
def test_nonstandard_nitrogen_descriptors_cover_the_nitrogen_atoms(
    smiles, expected_label, covered_nitrogens
):
    mol = Chem.MolFromSmiles(smiles)
    sites = find_sites_with_metadata(mol)
    labels = {site["label"] for site in sites}
    specifically_covered = {
        atom_idx
        for site in sites
        if not site.get("is_fallback", False)
        for atom_idx in site["atom_set"]
        if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 7
    }
    assert expected_label in labels
    assert len(specifically_covered) == covered_nitrogens
    assert not any(site.get("is_fallback", False) for site in sites)


def test_priority_alcohol_lowest():
    min_prio = min(LABEL_PRIORITY.values())
    assert LABEL_PRIORITY["primary_alcohol"] == min_prio


# ===================================================================
# 5. LABEL_FAMILY consistency
# ===================================================================

def test_label_family_carboxyl():
    assert LABEL_FAMILY["carboxylic_acid"] == "carboxyl"
    assert LABEL_FAMILY["carboxylate"] == "carboxyl"


def test_label_family_amine_includes_heterocycles():
    for label in ("pyridine", "pyridinium", "imidazole", "imidazolium",
                   "primary_amine", "primary_ammonium"):
        assert LABEL_FAMILY[label] == "amine", f"{label} should be in 'amine' family"


def test_label_family_amide():
    for label in ("amide_primary", "amide_secondary", "amide_tertiary", "lactam"):
        assert LABEL_FAMILY[label] == "amide", f"{label} should be in 'amide' family"


def test_conjugate_families_use_pair_classifier_names():
    assert LABEL_FAMILY["phenol"] == LABEL_FAMILY["phenolate"] == "phenol_phenolate"
    assert LABEL_FAMILY["thiol"] == LABEL_FAMILY["sulfide"] == "thiol_thiolate"
    assert LABEL_FAMILY["amidine"] == LABEL_FAMILY["amidinium"] == "amidine_like"
    assert LABEL_FAMILY["guanidine"] == LABEL_FAMILY["guanidinium"] == "guanidine_like"
    assert LABEL_FAMILY["phosphoric_acid"] == LABEL_FAMILY["phosphate_anion"] == "phosphate_oxyacid"
    assert LABEL_FAMILY["primary_alcohol"] == LABEL_FAMILY["alkoxide"] == "alcohol_alkoxide"
    assert LABEL_FAMILY["hydrazine"] == LABEL_FAMILY["hydrazinium"] == "hydrazine_like"
    assert LABEL_FAMILY["conjugated_hydrazine"] == "hydrazine_like"
    assert LABEL_FAMILY["hydroxylamine"] == LABEL_FAMILY["hydroxylammonium"] == "hydroxylamine_like"
    assert LABEL_FAMILY["protonated_alcohol"] == "alcohol_oxonium"
    assert LABEL_FAMILY["protonated_thiol"] == "thiol_thiolium"


def test_high_profile_families_are_recognized():
    """All HIGH_PROFILE_FAMILIES should appear as values in LABEL_FAMILY."""
    family_values = set(LABEL_FAMILY.values())
    for hpf in HIGH_PROFILE_FAMILIES:
        assert hpf in family_values, f"HIGH_PROFILE_FAMILY '{hpf}' not found in LABEL_FAMILY values"


# ===================================================================
# 6. Determinism: repeated runs produce identical results
# ===================================================================

@pytest.mark.parametrize("smiles", [
    "CC(=O)O",               # acetic acid
    "Nc1ccc(C(=O)O)cc1",     # p-aminobenzoic acid
    "c1ncc[nH]1",            # imidazole
    "OC(=O)c1ccc(N)cc1",     # 4-aminobenzoic acid alternate
    "C[NH3+]",               # methylammonium
    "CC(=O)Oc1ccccc1C(=O)O", # aspirin
])
def test_determinism_10_runs(smiles):
    """Site detection and resolution must be deterministic across 10 runs."""
    results = []
    for _ in range(10):
        accepted, _ = _resolve(smiles)
        sig = tuple(sorted((s["label"], s["atoms"]) for s in accepted))
        results.append(sig)

    first = results[0]
    for i, r in enumerate(results[1:], 2):
        assert r == first, (
            f"Non-determinism detected on run {i} for {smiles}!\n"
            f"  Run 1: {first}\n"
            f"  Run {i}: {r}"
        )


# ===================================================================
# 7. TOY_CASES from toy_protonation_assignment_check.py
# ===================================================================

TOY_CASES = [
    # (smiles, expected_label_in_resolved, expected_family)
    ("CC(=O)O", "carboxylic_acid", "carboxyl"),
    ("CC(=O)[O-]", "carboxylate", "carboxyl"),
    ("O=C(O)c1ccccc1", "carboxylic_acid", "carboxyl"),
    ("O=C([O-])c1ccccc1", "carboxylate", "carboxyl"),
    ("CN", "primary_amine", "amine"),
    ("C[NH3+]", "primary_ammonium", "amine"),
    ("CNC", "secondary_amine", "amine"),
    ("C[NH2+]C", "secondary_ammonium", "amine"),
    ("CN(C)C", "tertiary_amine", "amine"),
    ("C[NH+](C)C", "tertiary_ammonium", "amine"),
    ("Nc1ccccc1", "aniline", "amine"),
    ("[NH3+]c1ccccc1", "aryl_ammonium", "amine"),
    ("c1ncc[nH]1", "imidazole", "amine"),
]


@pytest.mark.parametrize(
    "smiles,expected_label,expected_family",
    TOY_CASES,
    ids=[f"{c[1]}_{c[0][:15]}" for c in TOY_CASES],
)
def test_toy_case_label_and_family(smiles, expected_label, expected_family):
    """Toy validation cases must resolve to their expected label and family."""
    accepted, _ = _resolve(smiles, threshold=0.5)
    labels = _labels(accepted)
    assert expected_label in labels, (
        f"Expected '{expected_label}' for {smiles}, got {labels}"
    )
    # Verify family
    matching = [s for s in accepted if s["label"] == expected_label]
    assert matching, f"No site with label '{expected_label}' found"
    assert matching[0]["family"] == expected_family, (
        f"Expected family '{expected_family}' for {expected_label}, "
        f"got '{matching[0]['family']}'"
    )


# ===================================================================
# 8. maximal_site_rank_key produces a sortable tuple
# ===================================================================

def test_maximal_site_rank_key_ordering():
    """Higher specificity/priority sites should rank higher."""
    site_high = {
        "label": "carboxylic_acid",
        "atom_set": {0, 1, 2, 3},
        "smarts_atoms": 4,
        "priority": 350,
        "specificity": 40350,
    }
    site_low = {
        "label": "primary_alcohol",
        "atom_set": {0, 1},
        "smarts_atoms": 2,
        "priority": 165,
        "specificity": 20165,
    }
    assert maximal_site_rank_key(site_high) > maximal_site_rank_key(site_low)


# ===================================================================
# 9. extract_site_subgraph
# ===================================================================

def test_extract_site_subgraph_returns_smiles():
    mol = Chem.MolFromSmiles("CC(=O)O")
    smi = extract_site_subgraph(mol, atom_idx=1, radius=2)
    assert isinstance(smi, str)
    assert len(smi) > 0


# ===================================================================
# 10. Edge cases: empty / invalid molecules
# ===================================================================

def test_find_sites_on_empty_mol():
    """Single atom should return few or no sites."""
    mol = Chem.MolFromSmiles("[Na+]")
    candidates = find_sites_with_metadata(mol)
    # Sodium ion has no organic functional groups
    labels = [c["label"] for c in candidates]
    assert "carboxylic_acid" not in labels


def test_get_default_pattern_specs_not_empty():
    specs = get_default_pattern_specs()
    assert len(specs) > 50, "Should have 50+ pattern specs across acidic+basic"
    for spec in specs:
        assert "label" in spec
        assert "smarts" in spec
        assert "site_type" in spec
