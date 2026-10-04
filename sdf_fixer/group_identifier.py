"""
group_identifier.py
====================
Identify ionizable functional groups on a molecule and classify their
acid/base character.  Uses the SMARTS library and substructure matching
already present in ``scripts/`` — this module wraps those functions and
adds higher-level reasoning about which group(s) a reported pKa likely
corresponds to.
"""

from __future__ import annotations

import sys
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem  # type: ignore

# ---------------------------------------------------------------------------
# Make the existing ``scripts/`` importable without installing them.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from substructure_match import (  # noqa: E402
    find_sites_with_metadata,
    resolve_overlapping_sites,
    maximal_site_rank_key,
)
from SMARTS_library import ACIDIC_PKA, BASIC_PKA  # noqa: E402


# ---------------------------------------------------------------------------
# Conjugate-pair and family mappings (mirrored from functional_group_pka_analysis
# so this module stays independent of that heavier script).
# ---------------------------------------------------------------------------

CONJUGATE_FAMILY_MAP: Dict[str, str] = {
    "carboxylic_acid": "carboxyl",
    "carboxylate": "carboxyl",
    "sulfonic_acid": "sulfonyl_oxyacid",
    "sulfonate": "sulfonyl_oxyacid",
    "carbamic_acid": "carbamic",
    "carbamate": "carbamic",
    "sulfamic_acid": "sulfamic",
    "sulfamate": "sulfamic",
    "iminium": "imine_iminium",
    "imine": "imine_iminium",
    "primary_amine": "amine",
    "secondary_amine": "amine",
    "tertiary_amine": "amine",
    "aniline": "amine",
    "primary_ammonium": "amine",
    "secondary_ammonium": "amine",
    "tertiary_ammonium": "amine",
    "quaternary_ammonium": "amine",
    "aryl_ammonium": "amine",
    "pyridine": "amine",
    "pyridinium": "amine",
    "pyrimidine": "amine",
    "pyrimidinium": "amine",
    "pyrazine": "amine",
    "pyrazinium": "amine",
    "pyrrole": "amine",
    "pyrrolium": "amine",
    "triazine": "amine",
    "triazinium": "amine",
    "imidazole": "amine",
    "imidazolium": "amine",
    "phenol": "phenol",
    "phenolate": "phenol",
    "primary_alcohol": "alcohol",
    "secondary_alcohol": "alcohol",
    "tertiary_alcohol": "alcohol",
    "thiol": "thiol",
    "thioide": "thiol",
}

# Labels whose pKa describes *loss of a proton* from the acid form.
# If the measured pKa is below physiological pH the group would be
# deprotonated (negatively charged or neutral depending on the pair).
ACIDIC_FAMILIES: Set[str] = {
    "carboxyl", "sulfonyl_oxyacid", "carbamic", "sulfamic", "phenol",
    "alcohol", "thiol",
}

# Labels whose reported "pKa" is really pKaH — the deprotonation
# equilibrium of the *conjugate acid*.  If pKaH is above physiological
# pH the group is protonated (e.g. R-NH₃⁺).
BASIC_FAMILIES: Set[str] = {"amine", "imine_iminium"}

# Map acid→base and base→acid conjugate pair labels.
CONJUGATE_MAP: Dict[str, str] = {
    "carboxylic_acid": "carboxylate",
    "sulfonic_acid": "sulfonate",
    "carbamic_acid": "carbamate",
    "sulfamic_acid": "sulfamate",
    "phenol": "phenolate",
    "primary_amine": "primary_ammonium",
    "secondary_amine": "secondary_ammonium",
    "tertiary_amine": "tertiary_ammonium",
    "aniline": "aryl_ammonium",
    "pyridine": "pyridinium",
    "pyrimidine": "pyrimidinium",
    "pyrazine": "pyrazinium",
    "pyrrole": "pyrrolium",
    "triazine": "triazinium",
    "imidazole": "imidazolium",
    "imine": "iminium",
}
CONJUGATE_MAP_INV = {v: k for k, v in CONJUGATE_MAP.items()}

# Combined literature pKa reference for rough plausibility checks.
REFERENCE_PKA: Dict[str, float] = {}
REFERENCE_PKA.update(ACIDIC_PKA)
REFERENCE_PKA.update(BASIC_PKA)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class IonizableSite:
    """A single ionizable functional group detected on a molecule."""

    label: str                          # e.g. "carboxylic_acid", "pyridine"
    site_type: str                      # "acidic" or "basic"
    family: str                         # conjugate family key
    atom_indices: Tuple[int, ...]       # matched atom indices
    atom_set: Set[int]
    priority: int
    specificity: int
    smarts: str
    reference_pka: Optional[float] = None    # literature range midpoint
    conjugate_label: Optional[str] = None    # label of the conjugate form

    @property
    def is_ionizable_acid(self) -> bool:
        """True when this group can *lose* a proton to become its conjugate base."""
        return self.family in ACIDIC_FAMILIES or self.label in CONJUGATE_MAP

    @property
    def is_ionizable_base(self) -> bool:
        """True when this group can *gain* a proton to become its conjugate acid."""
        return self.family in BASIC_FAMILIES or self.label in CONJUGATE_MAP_INV

    @property
    def pka_label(self) -> str:
        """Return ``'pKaH'`` for base families, ``'pKa'`` for acid families.

        The SDF "pKa" column always records the deprotonation equilibrium of
        the *relevant acid* species.  For bases this means the reported value
        is pKaH (= pKa of the conjugate acid), not a true acid pKa.
        """
        if self.family in BASIC_FAMILIES:
            return "pKaH"
        return "pKa"

    @property
    def effective_reference_pka(self) -> Optional[float]:
        """Reference pKa in the SDF-reported frame (pKaH for bases)."""
        return _effective_reference_pka(self)


@dataclass
class GroupIdentification:
    """Full identification result for a molecule."""

    mol: Chem.Mol
    smiles: str
    all_resolved_sites: List[Dict]       # raw dicts from substructure_match
    all_rejected_sites: List[Dict]
    ionizable_sites: List[IonizableSite] # filtered to groups that participate in proton transfer
    non_ionizable_sites: List[Dict]      # resolved groups that are NOT ionizable
    formal_charge: int
    atom_charges: Dict[int, int]

    @property
    def n_ionizable(self) -> int:
        return len(self.ionizable_sites)

    @property
    def is_unambiguous(self) -> bool:
        """True when there is exactly one ionizable group → automatic fix."""
        return self.n_ionizable == 1

    @property
    def is_ambiguous(self) -> bool:
        """True when there are multiple ionizable groups → user input needed."""
        return self.n_ionizable > 1

    @property
    def ionizable_labels(self) -> List[str]:
        return [s.label for s in self.ionizable_sites]

    def summary_str(self) -> str:
        parts = [f"SMILES: {self.smiles}  (charge={self.formal_charge})"]
        parts.append(f"  Resolved groups ({len(self.all_resolved_sites)} total):")
        for site in self.all_resolved_sites:
            parts.append(f"    - {site['label']} ({site['type']})  atoms={site['atoms']}")
        if self.ionizable_sites:
            parts.append(f"  Ionizable groups ({self.n_ionizable}):")
            for isite in self.ionizable_sites:
                conj = isite.conjugate_label or "n/a"
                ref = f"ref_pKa≈{isite.reference_pka:.1f}" if isite.reference_pka is not None else "ref_pKa=?"
                parts.append(
                    f"    [{isite.label}] family={isite.family}, type={isite.site_type}, "
                    f"conj={conj}, {ref}, atoms={isite.atom_indices}"
                )
        else:
            parts.append("  No ionizable groups detected.")
        if self.non_ionizable_sites:
            parts.append(f"  Non-ionizable groups ({len(self.non_ionizable_sites)}):")
            for site in self.non_ionizable_sites:
                parts.append(f"    - {site['label']} ({site['type']})  atoms={site['atoms']}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Which labels are considered "ionizable" (participate in proton exchange)?
# ---------------------------------------------------------------------------

IONIZABLE_LABELS: Set[str] = set(CONJUGATE_MAP.keys()) | set(CONJUGATE_MAP_INV.keys()) | {
    # Acidic-only (no simple conjugate base SMARTS in the library)
    "phosphoric_acid",
    "sulfinic_acid",
    "sulfenic_acid",
    "peroxide",
    "primary_alcohol",
    "secondary_alcohol",
    "tertiary_alcohol",
    "thiol",
    "phenol",
    # Acidic forms already in SMARTS
    "carboxylic_acid",
    "sulfonic_acid",
    "carbamic_acid",
    "sulfamic_acid",
}


def _is_ionizable_label(label: str) -> bool:
    """Return True when *label* corresponds to a group that can gain or lose H⁺."""
    return label in IONIZABLE_LABELS


def infer_pka_label(gid: "GroupIdentification") -> str:
    """Return ``'pKaH'`` if the most likely ionizable site is a base, else ``'pKa'``.

    For molecules with no ionizable groups, returns ``'pKa'`` as a
    conservative default (the value is a deprotonation equilibrium of
    *some* acid species even if we can't identify it).
    """
    if not gid.ionizable_sites:
        return "pKa"
    # If *any* site is basic the reported value is likely pKaH.
    for site in gid.ionizable_sites:
        if site.family in BASIC_FAMILIES:
            return "pKaH"
    return "pKa"


# ---------------------------------------------------------------------------
# Core identification function
# ---------------------------------------------------------------------------

def identify_groups(
    mol: Chem.Mol,
    overlap_threshold: float = 0.5,
) -> GroupIdentification:
    """
    Identify all functional groups on *mol*, partition them into ionizable
    and non-ionizable, and annotate with conjugate-pair information.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object (should already be sanitized).
    overlap_threshold : float
        Passed to ``resolve_overlapping_sites``.

    Returns
    -------
    GroupIdentification
    """
    smiles = Chem.MolToSmiles(mol, canonical=True)

    candidates = find_sites_with_metadata(mol)
    resolved, rejected = resolve_overlapping_sites(
        candidates,
        overlap_threshold=overlap_threshold,
    )

    # Partition resolved sites.
    ionizable: List[IonizableSite] = []
    non_ionizable: List[Dict] = []

    for site in resolved:
        label = site["label"]
        if _is_ionizable_label(label):
            family = CONJUGATE_FAMILY_MAP.get(label, label)
            conj = CONJUGATE_MAP.get(label) or CONJUGATE_MAP_INV.get(label)
            ref_pka = REFERENCE_PKA.get(label)

            ionizable.append(
                IonizableSite(
                    label=label,
                    site_type=site["type"],
                    family=family,
                    atom_indices=site["atoms"],
                    atom_set=set(site["atom_set"]),
                    priority=site["priority"],
                    specificity=site["specificity"],
                    smarts=site["smarts"],
                    reference_pka=ref_pka,
                    conjugate_label=conj,
                )
            )
        else:
            non_ionizable.append(site)

    # Atom-level charges.
    atom_charges: Dict[int, int] = {}
    total_charge = 0
    for atom in mol.GetAtoms():
        fc = atom.GetFormalCharge()
        if fc != 0:
            atom_charges[atom.GetIdx()] = int(fc)
        total_charge += fc

    return GroupIdentification(
        mol=mol,
        smiles=smiles,
        all_resolved_sites=resolved,
        all_rejected_sites=rejected,
        ionizable_sites=ionizable,
        non_ionizable_sites=non_ionizable,
        formal_charge=int(total_charge),
        atom_charges=atom_charges,
    )


# ---------------------------------------------------------------------------
# Convenience: identify from SMILES string
# ---------------------------------------------------------------------------

def identify_groups_from_smiles(
    smiles: str,
    overlap_threshold: float = 0.5,
) -> Optional[GroupIdentification]:
    """Convenience wrapper: parse SMILES then call `identify_groups`."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return identify_groups(mol, overlap_threshold=overlap_threshold)


# ---------------------------------------------------------------------------
# pKa-based protonation reasoning
# ---------------------------------------------------------------------------

def classify_protonation_expectation(
    site: IonizableSite,
    pka_value: float,
    ph: float = 7.4,
) -> str:
    """
    Given a *site* and a *pka_value*, return the expected protonation state
    label at *ph*.

    Returns
    -------
    str
        ``"protonated"`` — the site should carry the proton (acid form for
        acids, protonated form for bases).

        ``"deprotonated"`` — the site has lost the proton (conjugate base
        for acids, neutral form for bases).

        ``"borderline"`` — pKa is within ~0.5 units of pH, state uncertain.
    """
    margin = 0.5
    if site.family in ACIDIC_FAMILIES:
        # Acid: pKa < pH → deprotonated (lost H)
        if pka_value < ph - margin:
            return "deprotonated"
        elif pka_value > ph + margin:
            return "protonated"
        return "borderline"
    elif site.family in BASIC_FAMILIES:
        # Base: pKa > pH → protonated (gained H)
        if pka_value > ph + margin:
            return "protonated"
        elif pka_value < ph - margin:
            return "deprotonated"
        return "borderline"
    # Fallback for families we have no acid/base classification for.
    return "unknown"


def rank_sites_by_pka_plausibility(
    sites: List[IonizableSite],
    pka_value: float,
) -> List[Tuple[IonizableSite, float]]:
    """
    Rank ionizable sites by how close the reported pKa is to the
    literature reference for that functional group.

    **Important**: the SDF "pKa" column always records the deprotonation
    equilibrium of the *relevant acid* species.  For bases (amines,
    pyridines …) the reported value is therefore **pKaH** — i.e. the pKa
    of the conjugate acid (ammonium, pyridinium …), *not* the pKa for
    removing H from the neutral base.

    So when a site is detected in its base form (e.g. ``primary_amine``)
    we compare against the conjugate acid's reference pKa (e.g.
    ``ACIDIC_PKA["primary_ammonium"] ≈ 9``), which is the number in the
    same frame of reference as the SDF value.

    Returns a list of ``(site, abs_difference)`` sorted ascending
    (best match first).
    """
    scored: List[Tuple[IonizableSite, float]] = []
    for site in sites:
        ref = _effective_reference_pka(site)
        if ref is not None:
            diff = abs(pka_value - ref)
        else:
            diff = 999.0
        scored.append((site, diff))
    scored.sort(key=lambda x: x[1])
    return scored


def _effective_reference_pka(site: IonizableSite) -> Optional[float]:
    """
    Return the reference pKa to compare against the SDF's reported value.

    * For labels in ACIDIC_PKA (acid forms like ``carboxylic_acid``,
      ``primary_ammonium``, ``pyridinium`` …) the reference is the acid's
      own pKa — this is exactly the quantity the SDF records.
    * For labels in BASIC_PKA only (neutral base forms like
      ``primary_amine``, ``pyridine`` …) the SDF value is pKaH of the
      conjugate acid, so use ``ACIDIC_PKA[conjugate_label]``.
    * Fall back to the site's own reference_pka if neither lookup works.
    """
    label = site.label

    # If the label is directly in ACIDIC_PKA it's already in pKaH frame.
    if label in ACIDIC_PKA:
        return ACIDIC_PKA[label]

    # Neutral base form → look up conjugate acid's pKa.
    conj = site.conjugate_label
    if conj and conj in ACIDIC_PKA:
        return ACIDIC_PKA[conj]

    # Also check CONJUGATE_MAP / INV to find the acid's pKa.
    conj2 = CONJUGATE_MAP.get(label) or CONJUGATE_MAP_INV.get(label)
    if conj2 and conj2 in ACIDIC_PKA:
        return ACIDIC_PKA[conj2]

    # Last resort: use whatever reference_pka was assigned at construction.
    return site.reference_pka
