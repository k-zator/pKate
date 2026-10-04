"""
protonation_evaluator.py
========================
Given a parsed SDF record and its identified ionizable groups, evaluate
whether the molecule's SMILES correctly represents the protonation state
implied by the reported pKa value (at pH 7.4).

Key concepts
------------
The pKa *always* describes the deprotonation equilibrium  HA ⇌ A⁻ + H⁺.

* **Acid families** (carboxyl, phenol, …): the reported value is the
  *acid's own* pKa.  pKa < pH → deprotonated at that pH.
* **Base families** (amine, pyridine, …): the reported value is **pKaH**,
  i.e. the pKa of the *conjugate acid* (e.g. R-NH₃⁺).
  pKaH > pH → protonated at that pH.

Despite the SDF files labelling both quantities as "pKa", the correct
thermodynamic name for bases is pKaH.  Throughout this module the
variable ``pka_value`` holds whichever number was reported — true pKa
for acids, pKaH for bases — and the Henderson–Hasselbalch comparison
treats them identically:

    value > pH  →  acid form dominates  (protonated);
    value < pH  →  base form dominates  (deprotonated).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem  # type: ignore

from sdf_fixer.sdf_reader import SDFRecord
from sdf_fixer.group_identifier import (
    ACIDIC_FAMILIES,
    BASIC_FAMILIES,
    CONJUGATE_MAP,
    CONJUGATE_MAP_INV,
    CONJUGATE_FAMILY_MAP,
    GroupIdentification,
    IonizableSite,
    classify_protonation_expectation,
    rank_sites_by_pka_plausibility,
)


# ---------------------------------------------------------------------------
# Acid / base form label sets (built from CONJUGATE_MAP)
# ---------------------------------------------------------------------------

# "Acid form" = the label of the species that carries the dissociable proton.
# For acidic families this is e.g. carboxylic_acid (-COOH).
# For basic families this is e.g. primary_ammonium (-NH₃⁺).
ACID_FORM_LABELS: Set[str] = set()
BASE_FORM_LABELS: Set[str] = set()

for _acid_label, _base_label in CONJUGATE_MAP.items():
    family = CONJUGATE_FAMILY_MAP.get(_acid_label, "")
    if family in ACIDIC_FAMILIES:
        # acid_label is the protonated form, base_label is deprotonated
        ACID_FORM_LABELS.add(_acid_label)
        BASE_FORM_LABELS.add(_base_label)
    elif family in BASIC_FAMILIES:
        # For bases, CONJUGATE_MAP goes base→acid  (amine→ammonium)
        # so _acid_label is actually the neutral "base form"
        # and _base_label is the protonated "acid form"
        BASE_FORM_LABELS.add(_acid_label)
        ACID_FORM_LABELS.add(_base_label)
    else:
        # Fallback: treat the key as acid form
        ACID_FORM_LABELS.add(_acid_label)
        BASE_FORM_LABELS.add(_base_label)


# ---------------------------------------------------------------------------
# Data containers for the evaluation result
# ---------------------------------------------------------------------------

class Verdict(Enum):
    """What should happen to the SMILES."""
    CORRECT = "correct"                   # already in the right form
    NEEDS_DEPROTONATION = "needs_deprotonation"  # should lose H
    NEEDS_PROTONATION = "needs_protonation"      # should gain H
    BORDERLINE = "borderline"             # pKa ≈ pH, ambiguous
    AMBIGUOUS = "ambiguous"               # multiple ionizable groups
    NO_IONIZABLE = "no_ionizable"         # nothing detected
    UNKNOWN = "unknown"                   # can't determine


@dataclass
class ProtonationAssessment:
    """Assessment of one record's protonation state."""

    record_index: int
    original_smiles: str
    canonical_smiles: str
    pka_value: Optional[float]
    verdict: Verdict
    # The ionizable site the pKa is most likely associated with.
    assigned_site: Optional[IonizableSite] = None
    # Current protonation form of that site in the SMILES.
    current_form: Optional[str] = None          # "acid_form" / "base_form" / None
    # Expected protonation form given the pKa.
    expected_form: Optional[str] = None         # "acid_form" / "base_form" / None
    # Corrected SMILES (only set when we can compute it).
    corrected_smiles: Optional[str] = None
    # Human-readable explanation of what happened / should happen.
    explanation: str = ""

    @property
    def needs_fix(self) -> bool:
        return self.verdict in {
            Verdict.NEEDS_DEPROTONATION,
            Verdict.NEEDS_PROTONATION,
        }

    def summary_line(self) -> str:
        fix = "FIX" if self.needs_fix else "OK " if self.verdict == Verdict.CORRECT else "???"
        corr = f" → {self.corrected_smiles}" if self.corrected_smiles else ""
        return (
            f"[{fix}] #{self.record_index:>4d}  pKa={self.pka_value}  "
            f"{self.verdict.value:<22s}  {self.original_smiles}{corr}"
        )


# ---------------------------------------------------------------------------
# Determine current protonation form from the detected label
# ---------------------------------------------------------------------------

def _current_form_of(label: str) -> Optional[str]:
    """Return ``'acid_form'`` or ``'base_form'`` depending on label."""
    if label in ACID_FORM_LABELS:
        return "acid_form"
    if label in BASE_FORM_LABELS:
        return "base_form"
    return None


def _expected_form(pka_value: float, family: str, ph: float = 7.4) -> Optional[str]:
    """
    Return ``'acid_form'`` (protonated) or ``'base_form'`` (deprotonated)
    based on whether pKa is above or below pH.

    Universal Henderson–Hasselbalch:
      pKa > pH  →  acid form dominates  (protonated)
      pKa < pH  →  base form dominates  (deprotonated)
    """
    margin = 0.5
    if pka_value > ph + margin:
        return "acid_form"
    elif pka_value < ph - margin:
        return "base_form"
    return None  # borderline


# ---------------------------------------------------------------------------
# SMILES correction via RDKit — modify protonation on the assigned site
# ---------------------------------------------------------------------------

def _find_proton_donor_atom(mol: Chem.Mol, site: IonizableSite) -> Optional[int]:
    """
    Identify the atom in *site* that carries the dissociable proton (for
    deprotonation) or should receive one (for protonation).

    Heuristic:
    - For acidic families (carboxyl, phenol, …): the O or S atom with an H.
    - For basic families (amine, …): the N atom.
    """
    family = site.family

    if family in ACIDIC_FAMILIES:
        # Look for O or S bearing an explicit/implicit H inside the site
        for idx in site.atom_set:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() in (8, 16):  # O, S
                total_h = atom.GetTotalNumHs()
                if total_h > 0:
                    return idx
        # fallback: any O/S in the site
        for idx in site.atom_set:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() in (8, 16):
                return idx

    elif family in BASIC_FAMILIES:
        # Look for N inside the site
        for idx in site.atom_set:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 7:
                return idx

    return None


def _deprotonate_at(mol: Chem.Mol, atom_idx: int) -> Optional[str]:
    """Remove one H from *atom_idx*, reduce formal charge by 1, return SMILES."""
    rw = Chem.RWMol(copy.deepcopy(mol))
    atom = rw.GetAtomWithIdx(atom_idx)
    fc = atom.GetFormalCharge()
    num_hs = atom.GetNumExplicitHs()
    total_hs = atom.GetTotalNumHs()

    if num_hs > 0:
        atom.SetNumExplicitHs(num_hs - 1)
    elif total_hs > 0:
        # H is implicit; make it explicit then remove
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(total_hs - 1)

    atom.SetFormalCharge(fc - 1)

    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw, canonical=True)
    except Exception:
        return None


def _protonate_at(mol: Chem.Mol, atom_idx: int) -> Optional[str]:
    """Add one H to *atom_idx*, increase formal charge by 1, return SMILES."""
    rw = Chem.RWMol(copy.deepcopy(mol))
    atom = rw.GetAtomWithIdx(atom_idx)
    fc = atom.GetFormalCharge()
    num_hs = atom.GetNumExplicitHs()

    atom.SetNumExplicitHs(num_hs + 1)
    atom.SetFormalCharge(fc + 1)

    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw, canonical=True)
    except Exception:
        return None


def _apply_correction(
    mol: Chem.Mol,
    site: IonizableSite,
    verdict: Verdict,
) -> Optional[str]:
    """
    Try to produce a corrected canonical SMILES by modifying protonation
    on the assigned site.  Returns *None* if the correction cannot be
    applied cleanly.
    """
    target_atom = _find_proton_donor_atom(mol, site)
    if target_atom is None:
        return None

    if verdict == Verdict.NEEDS_DEPROTONATION:
        return _deprotonate_at(mol, target_atom)
    elif verdict == Verdict.NEEDS_PROTONATION:
        return _protonate_at(mol, target_atom)
    return None


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_record(
    rec: SDFRecord,
    gid: GroupIdentification,
    ph: float = 7.4,
) -> ProtonationAssessment:
    """
    Evaluate a single SDF record: determine whether the SMILES already
    reflects the correct protonation state for the reported pKa at *ph*,
    and if not, compute the corrected SMILES.

    Parameters
    ----------
    rec : SDFRecord
        Parsed SDF record (contains mol, SMILES, pKa, …).
    gid : GroupIdentification
        Result of ``identify_groups()`` on the record's molecule.
    ph : float
        Reference pH (default 7.4).

    Returns
    -------
    ProtonationAssessment
    """
    pka = rec.pka_value

    # ---- no pKa → nothing to evaluate ----
    if pka is None:
        return ProtonationAssessment(
            record_index=rec.record_index,
            original_smiles=rec.original_smiles,
            canonical_smiles=rec.canonical_smiles,
            pka_value=None,
            verdict=Verdict.UNKNOWN,
            explanation="No pKa value in record.",
        )

    # ---- no ionizable groups ----
    if gid.n_ionizable == 0:
        return ProtonationAssessment(
            record_index=rec.record_index,
            original_smiles=rec.original_smiles,
            canonical_smiles=rec.canonical_smiles,
            pka_value=pka,
            verdict=Verdict.NO_IONIZABLE,
            explanation="No ionizable functional group detected.",
        )

    # ---- multiple ionizable groups ----
    if gid.is_ambiguous:
        # Rank sites by how close the reported pKa is to the reference.
        ranking = rank_sites_by_pka_plausibility(gid.ionizable_sites, pka)
        best_site, best_diff = ranking[0]
        second_diff = ranking[1][1] if len(ranking) > 1 else 999.0

        # Auto-fix when the best-matching site is unambiguously better:
        #   • the best match is within 8 pKa units of the reference, AND
        #   • it's ≥ 5 pKa units closer than the runner-up
        # This covers e.g. carboxylic acid (ref≈4) + amine (ref≈33):
        # a reported pKa of 3.5 clearly belongs to the acid.
        clear_winner = (best_diff < 8.0 and (second_diff - best_diff) >= 5.0)

        if clear_winner:
            # Treat exactly like the single-group path below.
            site = best_site
            cur = _current_form_of(site.label)
            exp = _expected_form(pka, site.family, ph)

            if exp is None:
                return ProtonationAssessment(
                    record_index=rec.record_index,
                    original_smiles=rec.original_smiles,
                    canonical_smiles=rec.canonical_smiles,
                    pka_value=pka,
                    verdict=Verdict.BORDERLINE,
                    assigned_site=site,
                    current_form=cur,
                    expected_form=exp,
                    explanation=(
                        f"pKa={pka} is near pH {ph} for [{site.label}] "
                        f"(auto-selected from {gid.n_ionizable} ionizable groups, |Δ|={best_diff:.1f}); "
                        f"protonation state is borderline."
                    ),
                )

            if cur == exp:
                return ProtonationAssessment(
                    record_index=rec.record_index,
                    original_smiles=rec.original_smiles,
                    canonical_smiles=rec.canonical_smiles,
                    pka_value=pka,
                    verdict=Verdict.CORRECT,
                    assigned_site=site,
                    current_form=cur,
                    expected_form=exp,
                    explanation=(
                        f"[{site.label}] (auto-selected from {gid.n_ionizable} groups, |Δ|={best_diff:.1f}) "
                        f"is already in the expected "
                        f"{'protonated' if exp == 'acid_form' else 'deprotonated'} form."
                    ),
                )

            if exp == "base_form":
                verdict = Verdict.NEEDS_DEPROTONATION
                action = "deprotonation"
            else:
                verdict = Verdict.NEEDS_PROTONATION
                action = "protonation"

            corrected = _apply_correction(rec.mol, site, verdict)

            return ProtonationAssessment(
                record_index=rec.record_index,
                original_smiles=rec.original_smiles,
                canonical_smiles=rec.canonical_smiles,
                pka_value=pka,
                verdict=verdict,
                assigned_site=site,
                current_form=cur,
                expected_form=exp,
                corrected_smiles=corrected,
                explanation=(
                    f"[{site.label}] (auto-selected from {gid.n_ionizable} groups, |Δ|={best_diff:.1f}) "
                    f"detected as {'acid' if cur == 'acid_form' else 'base'} form, "
                    f"but pKa={pka} at pH {ph} implies {action} is needed."
                ),
            )

        # Truly ambiguous: multiple plausible groups → needs user input.
        cur = _current_form_of(best_site.label)
        exp = _expected_form(pka, best_site.family, ph)

        labels = ", ".join(s.label for s in gid.ionizable_sites)
        return ProtonationAssessment(
            record_index=rec.record_index,
            original_smiles=rec.original_smiles,
            canonical_smiles=rec.canonical_smiles,
            pka_value=pka,
            verdict=Verdict.AMBIGUOUS,
            assigned_site=best_site,
            current_form=cur,
            expected_form=exp,
            explanation=(
                f"Multiple ionizable groups [{labels}]; "
                f"best pKa match is [{best_site.label}] (|Δ|={best_diff:.1f}), "
                f"runner-up |Δ|={second_diff:.1f} — too close to auto-assign."
            ),
        )

    # ---- unambiguous: single ionizable group ----
    site = gid.ionizable_sites[0]
    cur = _current_form_of(site.label)
    exp = _expected_form(pka, site.family, ph)

    if exp is None:
        return ProtonationAssessment(
            record_index=rec.record_index,
            original_smiles=rec.original_smiles,
            canonical_smiles=rec.canonical_smiles,
            pka_value=pka,
            verdict=Verdict.BORDERLINE,
            assigned_site=site,
            current_form=cur,
            expected_form=exp,
            explanation=(
                f"pKa={pka} is near pH {ph} for [{site.label}]; "
                f"protonation state is borderline."
            ),
        )

    if cur == exp:
        return ProtonationAssessment(
            record_index=rec.record_index,
            original_smiles=rec.original_smiles,
            canonical_smiles=rec.canonical_smiles,
            pka_value=pka,
            verdict=Verdict.CORRECT,
            assigned_site=site,
            current_form=cur,
            expected_form=exp,
            explanation=(
                f"[{site.label}] is already in the expected "
                f"{'protonated' if exp == 'acid_form' else 'deprotonated'} form."
            ),
        )

    # Current ≠ expected → needs amendment.
    if exp == "base_form":
        verdict = Verdict.NEEDS_DEPROTONATION
        action = "deprotonation"
    else:
        verdict = Verdict.NEEDS_PROTONATION
        action = "protonation"

    corrected = _apply_correction(rec.mol, site, verdict)

    return ProtonationAssessment(
        record_index=rec.record_index,
        original_smiles=rec.original_smiles,
        canonical_smiles=rec.canonical_smiles,
        pka_value=pka,
        verdict=verdict,
        assigned_site=site,
        current_form=cur,
        expected_form=exp,
        corrected_smiles=corrected,
        explanation=(
            f"[{site.label}] (family={site.family}) detected as "
            f"{'acid' if cur == 'acid_form' else 'base'} form, "
            f"but pKa={pka} at pH {ph} implies {action} is needed."
        ),
    )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    records: List[SDFRecord],
    gids: List[GroupIdentification],
    ph: float = 7.4,
) -> List[ProtonationAssessment]:
    """Evaluate every record, return list of assessments."""
    return [evaluate_record(rec, gid, ph) for rec, gid in zip(records, gids)]
