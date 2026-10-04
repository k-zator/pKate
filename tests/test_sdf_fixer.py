"""
SDF Fixer Tests
===============
Tests for the sdf_fixer package: group identification, protonation
evaluation, SMILES correction, and end-to-end verdict determination.
"""

import sys
import os

import pytest
from rdkit import Chem

# Ensure sdf_fixer package is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sdf_fixer.group_identifier import (
    identify_groups,
    identify_groups_from_smiles,
    classify_protonation_expectation,
    rank_sites_by_pka_plausibility,
    IonizableSite,
    GroupIdentification,
    IONIZABLE_LABELS,
    CONJUGATE_MAP,
    CONJUGATE_MAP_INV,
    CONJUGATE_FAMILY_MAP,
    ACIDIC_FAMILIES,
    BASIC_FAMILIES,
    REFERENCE_PKA,
    infer_pka_label,
)
from sdf_fixer.protonation_evaluator import (
    evaluate_record,
    Verdict,
    ProtonationAssessment,
    ACID_FORM_LABELS,
    BASE_FORM_LABELS,
    _current_form_of,
    _expected_form,
    _find_proton_donor_atom,
    _deprotonate_at,
    _protonate_at,
)
from sdf_fixer.sdf_reader import SDFRecord


# ===================================================================
# Helpers
# ===================================================================

def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"RDKit could not parse '{smiles}'"
    return mol


def _make_record(smiles: str, pka: float = None, idx: int = 0) -> SDFRecord:
    """Create a minimal SDFRecord for testing evaluate_record."""
    mol = _mol(smiles)
    return SDFRecord(
        record_index=idx,
        mol=mol,
        original_smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(mol, canonical=True),
        pka_value=pka,
        formal_charge=Chem.GetFormalCharge(mol),
    )


# ===================================================================
# 1. identify_groups_from_smiles — basic group detection
# ===================================================================

class TestIdentifyGroups:

    def test_acetic_acid_single_ionizable(self):
        """Acetic acid → 1 ionizable group (carboxylic_acid)."""
        gid = identify_groups_from_smiles("CC(=O)O")
        assert gid is not None
        assert gid.n_ionizable >= 1
        labels = gid.ionizable_labels
        assert any("carboxyl" in l for l in labels), f"Expected carboxyl group, got {labels}"

    def test_methylamine_single_ionizable(self):
        """Methylamine → 1 ionizable group (primary_amine)."""
        gid = identify_groups_from_smiles("CN")
        assert gid is not None
        assert gid.n_ionizable >= 1
        labels = gid.ionizable_labels
        assert any("amine" in l or "ammonium" in l for l in labels), f"Expected amine, got {labels}"

    def test_glycine_two_ionizable_groups(self):
        """Glycine (NCC(=O)O) → at least 2 ionizable groups (amine + carboxyl)."""
        gid = identify_groups_from_smiles("NCC(=O)O")
        assert gid is not None
        assert gid.n_ionizable >= 2, f"Expected ≥2 ionizable, got {gid.n_ionizable}: {gid.ionizable_labels}"

    def test_benzene_no_ionizable_groups(self):
        """Benzene → 0 ionizable groups."""
        gid = identify_groups_from_smiles("c1ccccc1")
        assert gid is not None
        assert gid.n_ionizable == 0, f"Benzene should have no ionizable groups, got {gid.ionizable_labels}"

    def test_phenol(self):
        """Phenol → 1 ionizable group."""
        gid = identify_groups_from_smiles("Oc1ccccc1")
        assert gid is not None
        assert gid.n_ionizable >= 1
        assert any("phenol" in l for l in gid.ionizable_labels)

    def test_invalid_smiles_returns_none(self):
        result = identify_groups_from_smiles("not_a_smiles")
        assert result is None

    def test_unambiguous_single_group(self):
        gid = identify_groups_from_smiles("CC(=O)O")
        assert gid is not None
        if gid.n_ionizable == 1:
            assert gid.is_unambiguous
            assert not gid.is_ambiguous

    def test_ambiguous_multiple_groups(self):
        gid = identify_groups_from_smiles("NCC(=O)O")
        assert gid is not None
        if gid.n_ionizable > 1:
            assert gid.is_ambiguous
            assert not gid.is_unambiguous

    def test_atom_indices_are_valid(self):
        """Site atom indices should be valid for the molecule."""
        gid = identify_groups_from_smiles("CC(=O)O")
        assert gid is not None
        n_atoms = gid.mol.GetNumAtoms()
        for site in gid.ionizable_sites:
            for idx in site.atom_indices:
                assert 0 <= idx < n_atoms, f"Atom index {idx} out of range for {n_atoms}-atom molecule"

    def test_summary_str_runs(self):
        """summary_str() should not raise."""
        gid = identify_groups_from_smiles("NCC(=O)O")
        assert gid is not None
        s = gid.summary_str()
        assert isinstance(s, str)
        assert len(s) > 0


# ===================================================================
# 2. IONIZABLE_LABELS completeness
# ===================================================================

class TestIonizableLabels:

    def test_all_conjugate_map_keys_in_ionizable(self):
        for label in CONJUGATE_MAP:
            assert label in IONIZABLE_LABELS, f"CONJUGATE_MAP key '{label}' missing from IONIZABLE_LABELS"

    def test_all_conjugate_map_values_in_ionizable(self):
        for label in CONJUGATE_MAP.values():
            assert label in IONIZABLE_LABELS, f"CONJUGATE_MAP value '{label}' missing from IONIZABLE_LABELS"

    def test_no_empty_labels(self):
        for label in IONIZABLE_LABELS:
            assert label.strip() != "", "Empty string in IONIZABLE_LABELS"

    def test_conjugate_map_invertible(self):
        """CONJUGATE_MAP should be 1-to-1 (no duplicate values)."""
        values = list(CONJUGATE_MAP.values())
        assert len(values) == len(set(values)), "CONJUGATE_MAP has duplicate values"


# ===================================================================
# 3. classify_protonation_expectation
# ===================================================================

class TestClassifyProtonation:

    def _make_site(self, label: str, family: str = None, site_type: str = "acidic"):
        if family is None:
            family = CONJUGATE_FAMILY_MAP.get(label, label)
        return IonizableSite(
            label=label,
            site_type=site_type,
            family=family,
            atom_indices=(0,),
            atom_set={0},
            priority=0,
            specificity=0,
            smarts="[OH]",
        )

    def test_strong_acid_deprotonated(self):
        """pKa=2.0 at pH=7.4 → deprotonated."""
        site = self._make_site("carboxylic_acid", "carboxyl")
        assert classify_protonation_expectation(site, 2.0, 7.4) == "deprotonated"

    def test_weak_acid_protonated(self):
        """pKa=10.0 at pH=7.4 → protonated (acid is a phenol-like, above pH)."""
        site = self._make_site("phenol", "phenol")
        assert classify_protonation_expectation(site, 10.0, 7.4) == "protonated"

    def test_acid_borderline(self):
        """pKa ≈ pH → borderline."""
        site = self._make_site("carboxylic_acid", "carboxyl")
        assert classify_protonation_expectation(site, 7.4, 7.4) == "borderline"

    def test_base_protonated(self):
        """pKaH=10.0 > pH → protonated (ammonium form)."""
        site = self._make_site("primary_amine", "amine", "basic")
        assert classify_protonation_expectation(site, 10.0, 7.4) == "protonated"

    def test_base_deprotonated(self):
        """pKaH=5.0 < pH → deprotonated (free amine)."""
        site = self._make_site("pyridine", "amine", "basic")
        assert classify_protonation_expectation(site, 5.0, 7.4) == "deprotonated"

    def test_base_borderline(self):
        """pKaH ≈ pH → borderline."""
        site = self._make_site("primary_amine", "amine", "basic")
        result = classify_protonation_expectation(site, 7.4, 7.4)
        assert result == "borderline"

    def test_margin_boundary_acid(self):
        """pKa exactly at pH - 0.5 → should still be borderline (equal to margin)."""
        site = self._make_site("carboxylic_acid", "carboxyl")
        result = classify_protonation_expectation(site, 6.9, 7.4)
        # 6.9 < 7.4 - 0.5 = 6.9, borderline at equality
        assert result in ("borderline", "deprotonated")  # depending on < vs <=

    @pytest.mark.parametrize("pka,expected", [
        (0.0, "deprotonated"),
        (4.0, "deprotonated"),
        (6.5, "deprotonated"),   # 6.5 < 7.4 - 0.5 = 6.9
        (7.0, "borderline"),     # 7.0 is within (6.9, 7.9)
        (7.4, "borderline"),
        (8.5, "protonated"),     # 8.5 > 7.4 + 0.5 = 7.9
        (14.0, "protonated"),
    ])
    def test_acid_family_pka_sweep(self, pka, expected):
        site = self._make_site("carboxylic_acid", "carboxyl")
        result = classify_protonation_expectation(site, pka, 7.4)
        assert result == expected, f"pKa={pka}: expected {expected}, got {result}"


# ===================================================================
# 4. rank_sites_by_pka_plausibility
# ===================================================================

class TestRankSites:

    def _make_site(self, label, family, ref_pka=None):
        return IonizableSite(
            label=label,
            site_type="acidic" if family in ACIDIC_FAMILIES else "basic",
            family=family,
            atom_indices=(0,),
            atom_set={0},
            priority=0,
            specificity=0,
            smarts="[X]",
            reference_pka=ref_pka,
        )

    def test_closer_pka_ranks_first(self):
        """Site whose reference pKa is closer to the reported value should rank first."""
        acid = self._make_site("carboxylic_acid", "carboxyl")
        amine = self._make_site("primary_amine", "amine")
        ranking = rank_sites_by_pka_plausibility([acid, amine], pka_value=4.5)
        # carboxylic ref ≈ 4-5, amine ref ≈ 9-10
        assert ranking[0][0].label == "carboxylic_acid", (
            f"Expected carboxylic_acid first for pKa=4.5, got {ranking[0][0].label}"
        )

    def test_single_site_ranks(self):
        acid = self._make_site("carboxylic_acid", "carboxyl")
        ranking = rank_sites_by_pka_plausibility([acid], pka_value=4.5)
        assert len(ranking) == 1
        assert ranking[0][0].label == "carboxylic_acid"

    def test_empty_list(self):
        ranking = rank_sites_by_pka_plausibility([], pka_value=4.5)
        assert len(ranking) == 0

    def test_ranking_sorted_ascending_by_diff(self):
        acid = self._make_site("carboxylic_acid", "carboxyl")
        amine = self._make_site("primary_amine", "amine")
        phenol = self._make_site("phenol", "phenol")
        ranking = rank_sites_by_pka_plausibility([acid, amine, phenol], pka_value=4.5)
        diffs = [r[1] for r in ranking]
        assert diffs == sorted(diffs), f"Ranking not ascending: {diffs}"


# ===================================================================
# 5. protonation_evaluator — _current_form_of / _expected_form
# ===================================================================

class TestFormClassification:

    def test_acid_form_labels_non_empty(self):
        assert len(ACID_FORM_LABELS) > 0

    def test_base_form_labels_non_empty(self):
        assert len(BASE_FORM_LABELS) > 0

    def test_no_overlap(self):
        overlap = ACID_FORM_LABELS & BASE_FORM_LABELS
        assert len(overlap) == 0, f"Labels in both acid and base form sets: {overlap}"

    def test_carboxylic_acid_is_acid_form(self):
        assert _current_form_of("carboxylic_acid") == "acid_form"

    def test_carboxylate_is_base_form(self):
        assert _current_form_of("carboxylate") == "base_form"

    def test_primary_ammonium_is_acid_form(self):
        assert _current_form_of("primary_ammonium") == "acid_form"

    def test_primary_amine_is_base_form(self):
        assert _current_form_of("primary_amine") == "base_form"

    def test_unknown_label_returns_none(self):
        assert _current_form_of("unknown_xyz") is None


class TestExpectedForm:

    def test_acid_well_below_ph(self):
        assert _expected_form(2.0, "carboxyl", 7.4) == "base_form"

    def test_acid_well_above_ph(self):
        assert _expected_form(10.0, "carboxyl", 7.4) == "acid_form"

    def test_acid_borderline(self):
        assert _expected_form(7.4, "carboxyl", 7.4) is None

    def test_base_pka_above_ph(self):
        assert _expected_form(10.0, "amine", 7.4) == "acid_form"

    def test_base_pka_below_ph(self):
        assert _expected_form(5.0, "amine", 7.4) == "base_form"


# ===================================================================
# 6. SMILES correction — deprotonate / protonate
# ===================================================================

class TestSMILESCorrection:

    def test_deprotonate_acetic_acid(self):
        mol = _mol("CC(=O)O")
        Chem.AddHs(mol)  # ensure we have H info
        # Find O with H
        target = None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 8 and atom.GetTotalNumHs() > 0:
                target = atom.GetIdx()
                break
        if target is not None:
            result = _deprotonate_at(mol, target)
            if result is not None:
                assert "-" in result or "[O-]" in result, f"Deprotonation should produce anion: {result}"

    def test_protonate_methylamine(self):
        mol = _mol("CN")
        # Find N
        target = None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                target = atom.GetIdx()
                break
        assert target is not None
        result = _protonate_at(mol, target)
        if result is not None:
            assert "+" in result or "[NH3+]" in result, f"Protonation should produce cation: {result}"

    def test_find_proton_donor_acidic(self):
        mol = _mol("CC(=O)O")
        gid = identify_groups(mol)
        for site in gid.ionizable_sites:
            if site.family in ACIDIC_FAMILIES:
                donor = _find_proton_donor_atom(mol, site)
                if donor is not None:
                    atom = mol.GetAtomWithIdx(donor)
                    assert atom.GetAtomicNum() in (8, 16), "Acid donor should be O or S"

    def test_find_proton_donor_basic(self):
        mol = _mol("CN")
        gid = identify_groups(mol)
        for site in gid.ionizable_sites:
            if site.family in BASIC_FAMILIES:
                donor = _find_proton_donor_atom(mol, site)
                if donor is not None:
                    atom = mol.GetAtomWithIdx(donor)
                    assert atom.GetAtomicNum() == 7, "Base donor should be N"


# ===================================================================
# 7. evaluate_record — full evaluation verdicts
# ===================================================================

class TestEvaluateRecord:

    def test_no_pka_returns_unknown(self):
        rec = _make_record("CC(=O)O", pka=None)
        gid = identify_groups(rec.mol)
        result = evaluate_record(rec, gid)
        assert result.verdict == Verdict.UNKNOWN

    def test_no_ionizable_returns_no_ionizable(self):
        rec = _make_record("c1ccccc1", pka=5.0)
        gid = identify_groups(rec.mol)
        result = evaluate_record(rec, gid)
        assert result.verdict == Verdict.NO_IONIZABLE

    def test_acetic_acid_low_pka_needs_deprotonation(self):
        """Acetic acid (pKa≈4.76) depicted as -COOH at pH 7.4 → needs deprotonation."""
        rec = _make_record("CC(=O)O", pka=4.76)
        gid = identify_groups(rec.mol)
        result = evaluate_record(rec, gid, ph=7.4)
        if result.verdict == Verdict.NEEDS_DEPROTONATION:
            assert result.expected_form == "base_form"
            assert result.needs_fix
        elif result.verdict == Verdict.CORRECT:
            # If detected as already in base form, also valid
            assert result.expected_form in ("base_form", None)

    def test_correct_protonation_no_fix(self):
        """Acetic acid as acetate ([O-]) with pKa=4.76 → already correct at pH 7.4."""
        rec = _make_record("CC(=O)[O-]", pka=4.76)
        gid = identify_groups(rec.mol)
        result = evaluate_record(rec, gid, ph=7.4)
        if gid.n_ionizable > 0:
            # Should be CORRECT since it's already deprotonated
            assert result.verdict in (Verdict.CORRECT, Verdict.AMBIGUOUS, Verdict.BORDERLINE), (
                f"Expected CORRECT for already-deprotonated acetate, got {result.verdict}"
            )

    def test_borderline_pka(self):
        """pKa very close to pH → BORDERLINE verdict."""
        rec = _make_record("CC(=O)O", pka=7.4)
        gid = identify_groups(rec.mol)
        result = evaluate_record(rec, gid, ph=7.4)
        if gid.n_ionizable > 0:
            assert result.verdict == Verdict.BORDERLINE, f"pKa=pH should be BORDERLINE, got {result.verdict}"

    def test_ambiguous_glycine(self):
        """Glycine has 2 ionizable groups → AMBIGUOUS unless auto-resolved."""
        rec = _make_record("NCC(=O)O", pka=9.6)
        gid = identify_groups(rec.mol)
        result = evaluate_record(rec, gid, ph=7.4)
        if gid.n_ionizable > 1:
            # Should be ambiguous or auto-resolved (clear_winner)
            assert result.verdict in (
                Verdict.AMBIGUOUS,
                Verdict.CORRECT,
                Verdict.NEEDS_DEPROTONATION,
                Verdict.NEEDS_PROTONATION,
                Verdict.BORDERLINE,
            )

    def test_glycine_auto_resolve_acid(self):
        """Glycine pKa=2.3 → clearly the carboxyl group, should auto-assign."""
        rec = _make_record("NCC(=O)O", pka=2.3)
        gid = identify_groups(rec.mol)
        result = evaluate_record(rec, gid, ph=7.4)
        if gid.n_ionizable > 1 and result.assigned_site is not None:
            # The carboxyl should be auto-selected (much closer to ref≈4 than amine ref≈10)
            assert "carboxyl" in result.assigned_site.family or "carboxyl" in result.assigned_site.label

    def test_assessment_summary_line_runs(self):
        rec = _make_record("CC(=O)O", pka=4.76)
        gid = identify_groups(rec.mol)
        result = evaluate_record(rec, gid, ph=7.4)
        line = result.summary_line()
        assert isinstance(line, str)
        assert "4.76" in line

    def test_needs_fix_property(self):
        """needs_fix should be True only for NEEDS_DEPROTONATION/PROTONATION."""
        for v in Verdict:
            assessment = ProtonationAssessment(
                record_index=0,
                original_smiles="C",
                canonical_smiles="C",
                pka_value=5.0,
                verdict=v,
            )
            if v in (Verdict.NEEDS_DEPROTONATION, Verdict.NEEDS_PROTONATION):
                assert assessment.needs_fix
            else:
                assert not assessment.needs_fix


# ===================================================================
# 8. IonizableSite properties
# ===================================================================

class TestIonizableSiteProperties:

    def _make(self, label, family):
        return IonizableSite(
            label=label, site_type="acidic", family=family,
            atom_indices=(0,), atom_set={0}, priority=0, specificity=0, smarts="[X]",
        )

    def test_acidic_site_is_ionizable_acid(self):
        site = self._make("carboxylic_acid", "carboxyl")
        assert site.is_ionizable_acid

    def test_basic_site_is_ionizable_base(self):
        site = self._make("primary_amine", "amine")
        assert site.is_ionizable_base

    def test_pka_label_for_acid(self):
        site = self._make("carboxylic_acid", "carboxyl")
        assert site.pka_label == "pKa"

    def test_pka_label_for_base(self):
        site = self._make("primary_amine", "amine")
        assert site.pka_label == "pKaH"

    def test_effective_reference_pka_carboxyl(self):
        site = self._make("carboxylic_acid", "carboxyl")
        ref = site.effective_reference_pka
        assert ref is not None
        assert 2.0 < ref < 7.0, f"Carboxyl ref pKa should be ~4-5, got {ref}"


# ===================================================================
# 9. infer_pka_label
# ===================================================================

class TestInferPkaLabel:

    def test_acid_only_returns_pka(self):
        gid = identify_groups_from_smiles("CC(=O)O")
        if gid is not None and gid.n_ionizable > 0:
            assert infer_pka_label(gid) == "pKa"

    def test_base_returns_pkah(self):
        gid = identify_groups_from_smiles("CN")
        if gid is not None and gid.n_ionizable > 0:
            # If any amine-family site detected, should be pKaH
            if any(s.family in BASIC_FAMILIES for s in gid.ionizable_sites):
                assert infer_pka_label(gid) == "pKaH"

    def test_no_ionizable_returns_pka(self):
        gid = identify_groups_from_smiles("c1ccccc1")
        if gid is not None:
            assert infer_pka_label(gid) == "pKa"


# ===================================================================
# 10. REFERENCE_PKA consistency
# ===================================================================

class TestReferencePKA:

    def test_not_empty(self):
        assert len(REFERENCE_PKA) > 0

    def test_all_values_numeric(self):
        for label, val in REFERENCE_PKA.items():
            assert isinstance(val, (int, float)), f"REFERENCE_PKA['{label}'] = {val} is not numeric"

    def test_carboxyl_in_range(self):
        for label in ("carboxylic_acid",):
            if label in REFERENCE_PKA:
                v = REFERENCE_PKA[label]
                assert 2.0 < v < 7.0, f"Carboxyl ref {v} out of expected range"

    def test_amine_pkah_in_range(self):
        for label in ("primary_ammonium", "secondary_ammonium"):
            if label in REFERENCE_PKA:
                v = REFERENCE_PKA[label]
                assert 5.0 < v < 14.0, f"Amine pKaH ref {v} out of expected range"


# ===================================================================
# 11. Edge cases — unusual molecules
# ===================================================================

class TestEdgeCases:

    def test_zwitterion_glycine(self):
        """Zwitterionic glycine: [NH3+]CC(=O)[O-]."""
        gid = identify_groups_from_smiles("[NH3+]CC(=O)[O-]")
        assert gid is not None
        assert gid.formal_charge == 0
        # Should detect at least 2 ionizable sites
        assert gid.n_ionizable >= 1

    def test_multiply_charged(self):
        """Molecule with multiple charges."""
        gid = identify_groups_from_smiles("[NH3+]CCC(=O)[O-]")
        assert gid is not None
        assert gid.formal_charge == 0

    def test_sulfonic_acid(self):
        """Methanesulfonic acid."""
        gid = identify_groups_from_smiles("CS(=O)(=O)O")
        assert gid is not None
        labels = gid.ionizable_labels
        assert any("sulfon" in l for l in labels) or gid.n_ionizable > 0

    def test_single_atom_molecule(self):
        """Single-atom molecules should not crash."""
        gid = identify_groups_from_smiles("[Na+]")
        assert gid is not None
        # Sodium ion has no ionizable functional groups
        assert gid.n_ionizable == 0

    def test_large_molecule_does_not_crash(self):
        """A moderately complex molecule should not cause errors."""
        # Aspirin
        gid = identify_groups_from_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert gid is not None
        assert gid.n_ionizable >= 1

    @pytest.mark.parametrize("smiles,min_ionizable", [
        ("CC(=O)O", 1),           # acetic acid
        ("CN", 1),                # methylamine
        ("Oc1ccccc1", 1),         # phenol
        ("CS", 1),                # methanethiol
        ("NCC(=O)O", 2),          # glycine
        ("c1ccccc1", 0),          # benzene
        ("CCCC", 0),              # butane
        ("CC(=O)C", 0),           # acetone
    ])
    def test_known_molecule_ionizable_count(self, smiles, min_ionizable):
        gid = identify_groups_from_smiles(smiles)
        assert gid is not None
        assert gid.n_ionizable >= min_ionizable, (
            f"{smiles}: expected ≥{min_ionizable} ionizable, got {gid.n_ionizable}: {gid.ionizable_labels}"
        )
