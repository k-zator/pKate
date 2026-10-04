"""
SMARTS Pattern Regression Gate Tests
=====================================
These tests are the first line of defense against SMARTS library breakage.
Every test is designed to **SHOUT LOUDLY** if a SMARTS pattern no longer
matches its expected molecule or if a pattern string becomes invalid.

Run these tests in CI on every commit that touches SMARTS_library.py.
"""

import pytest
from rdkit import Chem

from SMARTS_library import BASIC_SMARTS, ACIDIC_SMARTS, ACIDIC_PKA, BASIC_PKA


# ===================================================================
# 1.  ALL SMARTS PATTERNS MUST BE VALID RDKit SMARTS
# ===================================================================

ALL_SMARTS = {**BASIC_SMARTS, **ACIDIC_SMARTS}


@pytest.mark.parametrize(
    "label,smarts",
    list(ALL_SMARTS.items()),
    ids=list(ALL_SMARTS.keys()),
)
def test_smarts_pattern_is_valid_rdkit(label, smarts):
    """Every SMARTS string must parse to a valid RDKit query molecule."""
    patt = Chem.MolFromSmarts(smarts)
    assert patt is not None, (
        f"\n\n"
        f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        f"  SMARTS BROKEN: '{label}'\n"
        f"  Pattern: {smarts}\n"
        f"  RDKit cannot parse this as a valid SMARTS string.\n"
        f"  ALL substructure matching using this pattern is DEAD.\n"
        f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    )


# ===================================================================
# 2.  ACIDIC SMARTS — positive match tests
# ===================================================================

ACIDIC_POSITIVE_CASES = [
    # (label, SMILES that MUST match, human-readable name)
    ("carboxylic_acid", "CC(=O)O", "acetic acid"),
    ("carboxylic_acid", "OC(=O)c1ccccc1", "benzoic acid"),
    ("carboxylic_acid", "OC(=O)CC(=O)O", "malonic acid"),
    ("phenol", "Oc1ccccc1", "phenol"),
    ("phenol", "Oc1ccc(O)cc1", "hydroquinone"),
    ("thiol", "CS", "methanethiol"),
    ("thiol", "c1ccc(S)cc1", "thiophenol"),
    ("primary_alcohol", "CCO", "ethanol"),
    ("primary_alcohol", "CO", "methanol"),
    ("secondary_alcohol", "CC(O)C", "isopropanol"),
    ("secondary_alcohol", "OC1NCCC1", "heteroatom-substituted secondary alcohol"),
    ("tertiary_alcohol", "CC(C)(C)O", "tert-butanol"),
    ("protonated_alcohol", "C[OH2+]", "methyloxonium"),
    ("protonated_thiol", "C[SH2+]", "protonated methanethiol"),
    ("hydrazinium", "C[NH2+]N", "methylhydrazinium"),
    ("hydroxylammonium", "CO[NH3+]", "O-methylhydroxylammonium"),
    ("phosphoric_acid", "COP(=O)(O)O", "methyl phosphate"),
    ("phosphonic_acid", "CP(=O)(O)O", "methylphosphonic acid"),
    ("phosphinic_acid", "CP(=O)(O)C", "dimethylphosphinic acid"),
    ("phosphoramidic_acid", "NP(=O)(O)OC", "methyl phosphoramidate"),
    ("sulfonic_acid", "CS(=O)(=O)O", "methanesulfonic acid"),
    ("primary_ammonium", "C[NH3+]", "methylammonium"),
    ("secondary_ammonium", "C[NH2+]C", "dimethylammonium"),
    ("secondary_ammonium", "C[NH2+]c1ccccc1", "N-methylanilinium"),
    ("tertiary_ammonium", "C[NH+](C)C", "trimethylammonium"),
    ("aryl_ammonium", "[NH3+]c1ccccc1", "anilinium"),
    ("pyridinium", "c1cc[nH+]cc1", "pyridinium"),
    ("quinolinium", "c1ccc2ccc[nH+]c2c1", "quinolinium"),
    ("isoquinolinium", "c1ccc2c[nH+]ccc2c1", "isoquinolinium"),
    ("pyrrolium", "[NH+]1=CC=CC1", "C2-protonated pyrrole"),
    ("iminium", "CC(C)=[NH2+]", "simple iminium"),
    ("imidazolium", "c1[nH]c[nH+]c1", "imidazolium"),
    ("pyrazolium", "c1cc[nH][nH+]1", "pyrazolium"),
    ("carbamic_acid", "NC(=O)O", "carbamic acid"),
    ("amidinium", "NC=[NH2+]", "formamidinium"),
]


@pytest.mark.parametrize(
    "label,smiles,name",
    ACIDIC_POSITIVE_CASES,
    ids=[f"{c[0]}_{c[2].replace(' ', '_')}" for c in ACIDIC_POSITIVE_CASES],
)
def test_acidic_smarts_matches_expected_molecule(label, smiles, name):
    """Each ACIDIC_SMARTS pattern must match its canonical positive example."""
    smarts = ACIDIC_SMARTS[label]
    patt = Chem.MolFromSmarts(smarts)
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Cannot parse SMILES: {smiles}"
    has_match = mol.HasSubstructMatch(patt)
    assert has_match, (
        f"\n\n"
        f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        f"  SMARTS REGRESSION: '{label}' NO LONGER MATCHES!\n"
        f"  SMILES: {smiles}  ({name})\n"
        f"  SMARTS: {smarts}\n"
        f"  Subgraph detection for this functional group is BROKEN.\n"
        f"  This will corrupt pKa prediction for all {label} molecules.\n"
        f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    )


# ===================================================================
# 3.  BASIC SMARTS — positive match tests
# ===================================================================

BASIC_POSITIVE_CASES = [
    ("primary_amine", "CN", "methylamine"),
    ("primary_amine", "NCC(=O)O", "glycine amine"),
    ("secondary_amine", "CNC", "dimethylamine"),
    ("tertiary_amine", "CN(C)C", "trimethylamine"),
    ("aniline", "Nc1ccccc1", "aniline"),
    ("pyridine", "c1ccncc1", "pyridine"),
    ("quinoline", "c1ccc2ncccc2c1", "quinoline"),
    ("isoquinoline", "c1ccc2cnccc2c1", "isoquinoline"),
    ("pyrimidine", "c1ncncc1", "pyrimidine"),
    ("pyrazine", "c1cnccn1", "pyrazine"),
    ("pyrrole", "c1cc[nH]c1", "pyrrole"),
    ("imidazole", "c1ncc[nH]1", "imidazole"),
    ("pyrazole", "c1cc[nH]n1", "pyrazole"),
    ("triazine", "c1ncncn1", "1,3,5-triazine"),
    ("amide_primary", "CC(=O)N", "acetamide"),
    ("amide_secondary", "CC(=O)NC", "N-methylacetamide"),
    ("amide_tertiary", "CC(=O)N(C)C", "N,N-dimethylacetamide"),
    ("ester", "CC(=O)OC", "methyl acetate"),
    ("ketone", "CC(=O)C", "acetone"),
    ("aldehyde", "CC=O", "acetaldehyde"),
    ("carboxylate", "CC(=O)[O-]", "acetate"),
    ("phenolate", "[O-]c1ccccc1", "phenolate"),
    ("alkoxide", "CC[O-]", "ethoxide"),
    ("hydrazine", "CNN", "methylhydrazine terminal nitrogen"),
    ("conjugated_hydrazine", "NNc1ccccc1", "phenylhydrazine terminal nitrogen"),
    ("hydroxylamine", "CON", "O-methylhydroxylamine"),
    ("phosphate_anion", "COP(=O)([O-])O", "methyl phosphate monoanion"),
    ("phosphonate_anion", "CP(=O)([O-])O", "methylphosphonate monoanion"),
    ("phosphinate_anion", "CP(=O)([O-])C", "dimethylphosphinate"),
    ("phosphoramidate_anion", "NP(=O)([O-])OC", "methyl phosphoramidate anion"),
    ("phosphate_ester", "COP(=O)(O)O", "methyl phosphate ester linkage"),
    ("imine", "CC=NC", "N-methylethanimine"),
    ("sulfide", "c1ccc([S-])cc1", "thiophenolate"),
    ("sulfamate_anion", "NS(=O)(=O)[O-]", "sulfamate anion"),
    ("sulfonamide", "CS(=O)(=O)NC", "N-methylmethanesulfonamide"),
    ("nitro", "c1ccc([N+](=O)[O-])cc1", "nitrobenzene"),
    ("sulfoxide", "C[S+]([O-])C", "charge-separated dimethyl sulfoxide"),
]


@pytest.mark.parametrize(
    "label,smiles,name",
    BASIC_POSITIVE_CASES,
    ids=[f"{c[0]}_{c[2].replace(' ', '_')}" for c in BASIC_POSITIVE_CASES],
)
def test_basic_smarts_matches_expected_molecule(label, smiles, name):
    """Each BASIC_SMARTS pattern must match its canonical positive example."""
    smarts = BASIC_SMARTS[label]
    patt = Chem.MolFromSmarts(smarts)
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Cannot parse SMILES: {smiles}"
    has_match = mol.HasSubstructMatch(patt)
    assert has_match, (
        f"\n\n"
        f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        f"  SMARTS REGRESSION: '{label}' NO LONGER MATCHES!\n"
        f"  SMILES: {smiles}  ({name})\n"
        f"  SMARTS: {smarts}\n"
        f"  Subgraph detection for this functional group is BROKEN.\n"
        f"  This will corrupt downstream group assignment.\n"
        f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    )


# ===================================================================
# 4.  NEGATIVE match tests — patterns must NOT match wrong molecules
# ===================================================================

NEGATIVE_CASES = [
    # (dict_name, label, wrong_smiles, why_it_should_not_match)
    ("acidic", "carboxylic_acid", "CC(=O)N", "amide has C(=O)N not C(=O)OH"),
    ("acidic", "carboxylic_acid", "CC(=O)OC", "ester has C(=O)OC, no OH"),
    ("acidic", "phenol", "COc1ccccc1", "anisole: O is bonded to C, not H"),
    ("acidic", "thiol", "CSC", "thioether — S has no H"),
    ("acidic", "primary_ammonium", "CN", "neutral amine is not NH3+"),
    ("acidic", "pyridinium", "c1ccncc1", "neutral pyridine is not nH+"),
    ("basic", "primary_amine", "CC(=O)N", "amide N excluded by !$([N][C](=O))"),
    ("basic", "aniline", "CC(=O)Nc1ccccc1", "anilide nitrogen is an amide, not an aniline base"),
    ("basic", "aniline", "CS(=O)(=O)Nc1ccccc1", "sulfonamide nitrogen is not an aniline base"),
    ("basic", "primary_amine", "C[NH3+]", "ammonium is NX4+, not NX3"),
    ("basic", "tertiary_amine", "C[N+](=O)[O-]", "nitro N+ is not a tertiary amine"),
    ("basic", "aniline", "c1ccccc1[N+](=O)[O-]", "nitro N+ is not an aniline"),
    ("basic", "enamine", "C=C[N+](=O)[O-]", "vinylic nitro N+ is not an enamine"),
    ("basic", "pyridine", "c1ccccc1", "benzene has no nitrogen"),
    ("basic", "ester", "CC(=O)O", "carboxylic acid: O has H, not OX2H0"),
    ("basic", "ketone", "CC=O", "aldehyde has H on carbonyl C"),
    ("basic", "sulfide", "C=S", "neutral thione sulfur is not a thiolate anion"),
    ("basic", "thiophenol", "c1ccccc1[S+](C)[O-]", "aryl sulfoxide is not thiophenol"),
    ("acidic", "secondary_ammonium", "CNC", "neutral sec amine — no charge"),
    ("acidic", "tertiary_ammonium", "CN(C)C", "neutral tert amine — no charge"),
]


@pytest.mark.parametrize(
    "dict_name,label,wrong_smiles,reason",
    NEGATIVE_CASES,
    ids=[f"NOT_{c[1]}_on_{c[2].replace(' ', '_')}" for c in NEGATIVE_CASES],
)
def test_smarts_does_not_match_wrong_molecule(dict_name, label, wrong_smiles, reason):
    """Patterns must NOT match structurally similar but chemically different molecules."""
    smarts_dict = ACIDIC_SMARTS if dict_name == "acidic" else BASIC_SMARTS
    smarts = smarts_dict[label]
    patt = Chem.MolFromSmarts(smarts)
    mol = Chem.MolFromSmiles(wrong_smiles)
    assert mol is not None, f"Cannot parse SMILES: {wrong_smiles}"
    has_match = mol.HasSubstructMatch(patt)
    assert not has_match, (
        f"\n\n"
        f"  SMARTS SELECTIVITY FAILURE: '{label}' matches {wrong_smiles}\n"
        f"  but it should NOT because: {reason}\n"
        f"  SMARTS: {smarts}\n"
        f"  This means the pattern is too broad and will cause false positives.\n"
    )


# ===================================================================
# 5.  Conjugate pair SMARTS consistency
# ===================================================================

# Conjugate pairs: for each acid→base pair in CONJUGATE_MAP (defined in
# functional_group_pka_analysis.py), both labels must exist in the SMARTS dicts.
CONJUGATE_PAIRS = [
    ("phosphoric_acid", "phosphate_anion"),
    ("phosphonic_acid", "phosphonate_anion"),
    ("phosphinic_acid", "phosphinate_anion"),
    ("phosphoramidic_acid", "phosphoramidate_anion"),
    ("primary_alcohol", "alkoxide"),
    ("secondary_alcohol", "alkoxide"),
    ("tertiary_alcohol", "alkoxide"),
    ("carboxylic_acid", "carboxylate"),
    ("sulfonic_acid", "sulfonate"),
    ("carbamic_acid", "carbamate"),
    ("sulfamic_acid", "sulfamate_anion"),
    ("phenol", "phenolate"),
    ("primary_amine", "primary_ammonium"),
    ("secondary_amine", "secondary_ammonium"),
    ("tertiary_amine", "tertiary_ammonium"),
    ("aniline", "aryl_ammonium"),
    ("pyridine", "pyridinium"),
    ("quinoline", "quinolinium"),
    ("isoquinoline", "isoquinolinium"),
    ("pyrimidine", "pyrimidinium"),
    ("pyrazine", "pyrazinium"),
    ("pyrrole", "pyrrolium"),
    ("triazine", "triazinium"),
    ("imidazole", "imidazolium"),
    ("imine", "iminium"),
]


@pytest.mark.parametrize(
    "acid_label,base_label",
    CONJUGATE_PAIRS,
    ids=[f"{a}_↔_{b}" for a, b in CONJUGATE_PAIRS],
)
def test_conjugate_pair_both_labels_exist_in_smarts(acid_label, base_label):
    """Both members of every conjugate pair must have SMARTS in the library."""
    all_labels = set(BASIC_SMARTS.keys()) | set(ACIDIC_SMARTS.keys())
    assert acid_label in all_labels, (
        f"Conjugate pair member '{acid_label}' has no SMARTS pattern!\n"
        f"Its pair partner '{base_label}' exists — this will break conjugate matching."
    )
    assert base_label in all_labels, (
        f"Conjugate pair member '{base_label}' has no SMARTS pattern!\n"
        f"Its pair partner '{acid_label}' exists — this will break conjugate matching."
    )


# ===================================================================
# 6.  pKa reference tables cross-reference SMARTS dicts
# ===================================================================

def test_acidic_pka_keys_all_have_smarts():
    """Every label in ACIDIC_PKA must have a corresponding SMARTS pattern."""
    all_labels = set(BASIC_SMARTS.keys()) | set(ACIDIC_SMARTS.keys())
    for label in ACIDIC_PKA:
        assert label in all_labels, (
            f"ACIDIC_PKA has reference pKa for '{label}' but no SMARTS pattern exists!\n"
            f"This reference value is DEAD — it can never be matched to a molecule."
        )


def test_basic_pka_keys_all_have_smarts():
    """Every label in BASIC_PKA must have a corresponding SMARTS pattern."""
    all_labels = set(BASIC_SMARTS.keys()) | set(ACIDIC_SMARTS.keys())
    for label in BASIC_PKA:
        assert label in all_labels, (
            f"BASIC_PKA has reference pKa for '{label}' but no SMARTS pattern exists!\n"
            f"This reference value is DEAD — it can never be matched to a molecule."
        )


# ===================================================================
# 7.  ACIDIC_PKA ranges are chemically plausible
# ===================================================================

def test_acidic_pka_ranges_plausible():
    """All ACIDIC_PKA values should be within (-10, 20) — broad but sane."""
    for label, pka in ACIDIC_PKA.items():
        assert -10 <= pka <= 20, (
            f"ACIDIC_PKA['{label}'] = {pka} is outside the plausible range (-10, 20).\n"
            f"This looks like a data-entry error."
        )


# ===================================================================
# 8.  No duplicate keys between BASIC and ACIDIC SMARTS
#     (except intentionally shared like thiol/phenol — those appear in
#      ACIDIC only and their neutral forms in BASIC)
# ===================================================================

def test_no_accidental_key_collisions():
    """Keys present in both dicts must be intentional."""
    basic_keys = set(BASIC_SMARTS.keys())
    acidic_keys = set(ACIDIC_SMARTS.keys())
    overlap = basic_keys & acidic_keys
    # Currently no keys should overlap — acidic and basic dicts are disjoint
    assert overlap == set(), (
        f"Labels appear in BOTH BASIC_SMARTS and ACIDIC_SMARTS: {overlap}\n"
        f"This will cause duplicate matches and confuse site assignment."
    )


# ===================================================================
# 9.  Edge-case SMARTS: complex heterocycles
# ===================================================================

HETEROCYCLE_EDGE_CASES = [
    ("basic", "imidazole", "c1nc2ccccc2[nH]1", "benzimidazole"),
    ("basic", "pyridine", "c1ccnc2ccccc12", "quinoline"),
    ("basic", "pyridine", "c1ccc2ncccc2c1", "isoquinoline-like"),
    ("basic", "thiazole", "c1cnsc1", "thiazole"),
    ("basic", "thiophene", "c1cscc1", "thiophene"),
    ("basic", "furan", "c1cocc1", "furan"),
    ("acidic", "tetrazolium", "C1=NN=[NH+]N1", "tetrazolium"),
    ("basic", "tetrazole", "c1nnn[nH]1", "tetrazole"),
]


@pytest.mark.parametrize(
    "dict_name,label,smiles,name",
    HETEROCYCLE_EDGE_CASES,
    ids=[f"{c[1]}_{c[3]}" for c in HETEROCYCLE_EDGE_CASES],
)
def test_heterocycle_edge_cases(dict_name, label, smiles, name):
    """Complex heterocyclic SMARTS must still match fused/substituted variants."""
    smarts_dict = ACIDIC_SMARTS if dict_name == "acidic" else BASIC_SMARTS
    smarts = smarts_dict[label]
    patt = Chem.MolFromSmarts(smarts)
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Cannot parse SMILES: {smiles}"
    has_match = mol.HasSubstructMatch(patt)
    assert has_match, (
        f"\n\n"
        f"  HETEROCYCLE EDGE CASE FAILURE: '{label}' does not match {smiles} ({name})\n"
        f"  SMARTS: {smarts}\n"
        f"  Fused/complex heterocycle matching is BROKEN.\n"
    )
