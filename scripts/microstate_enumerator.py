"""Generate atom-mapped conjugate microstates for a resolved ionization site.

The functions in this module operate on a complete RDKit molecule.  Atom maps
refer to the input heavy-atom indices (one-based), so the same atom can be
followed through protonation and tautomer enumeration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem, rdBase  # type: ignore
from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore


DEFAULT_MAX_PROTONATION_STATES = 4096


ACIDIC_SITE_FAMILIES = {
    "alcohol_oxonium",
    "thiol_thiolium",
    "carboxyl",
    "sulfonyl_oxyacid",
    "carbamic",
    "sulfamic",
    "phenol_phenolate",
    "thiol_thiolate",
    "phosphate_oxyacid",
    "phosphonate_oxyacid",
    "phosphinate_oxyacid",
    "phosphoramidate_oxyacid",
    "alcohol_alkoxide",
    "tetrazole_tetrazolate",
    "123triazole_acidity",
    "124triazole_acidity",
    "indazole_acidity",
    "pyrazole_acidity",
    "imidazole_acidity",
}

BASIC_SITE_FAMILIES = {
    "hydrazine_like",
    "hydroxylamine_like",
    "amine",
    "aziridine",
    "imine_iminium",
    "amidine_like",
    "guanidine_like",
    "n_oxide",
    "amine_n_oxide",
    "pyridazine",
    "benzimidazole",
    "oxazole",
    "isoxazole",
    "thiazole",
    "isothiazole",
    "124triazole_basicity",
    "indazole_basicity",
    "pyrazole_basicity",
    "imidazole_basicity",
}

N_H_ACID_SITE_FAMILIES = {
    "123triazole_acidity", "124triazole_acidity", "indazole_acidity",
    "pyrazole_acidity", "imidazole_acidity",
}


@dataclass(frozen=True)
class MicrostateResult:
    acid_microstates: Tuple[str, ...]
    base_microstates: Tuple[str, ...]
    site_atom_maps: Tuple[int, ...]
    protonation_center_maps: Tuple[int, ...]
    input_member_form: str
    acid_charge: int
    base_charge: int
    status: str
    note: str = ""
    acid_tautomer_enumeration_truncated: bool = False
    base_tautomer_enumeration_truncated: bool = False


def formal_charge(mol: Chem.Mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def ensure_heavy_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy with stable, unique one-based maps on every heavy atom."""
    mapped = Chem.Mol(mol)
    used = {
        atom.GetAtomMapNum()
        for atom in mapped.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetAtomMapNum() > 0
    }
    preserve_existing = len(used) == sum(
        1 for atom in mapped.GetAtoms() if atom.GetAtomicNum() > 1
    )
    if preserve_existing:
        return mapped

    for atom in mapped.GetAtoms():
        if atom.GetAtomicNum() > 1:
            atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mapped


def mapped_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _copy_and_sanitize(mol: Chem.Mol) -> Optional[Chem.Mol]:
    candidate = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(candidate)
    except Exception:
        return None
    return candidate


def _site_indices(site: Dict, mol: Chem.Mol) -> Tuple[int, ...]:
    raw_indices = site.get("atom_set", site.get("atoms", site.get("atom_indices", ())))
    indices = tuple(sorted(int(idx) for idx in raw_indices))
    if any(idx < 0 or idx >= mol.GetNumAtoms() for idx in indices):
        raise ValueError("Resolved site contains an atom index outside the molecule")
    return indices


def _site_map_numbers(mol: Chem.Mol, indices: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(mol.GetAtomWithIdx(idx).GetAtomMapNum() for idx in indices))


def _has_double_bond_to_carbon(atom: Chem.Atom) -> bool:
    return any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetOtherAtom(atom).GetAtomicNum() == 6
        for bond in atom.GetBonds()
    )


def find_protonation_centers(
    mol: Chem.Mol,
    site: Dict,
    family: str,
    member_form: str,
) -> Tuple[int, ...]:
    """Return chemically plausible proton-bearing/proton-accepting atom indices."""
    indices = _site_indices(site, mol)
    atoms = [mol.GetAtomWithIdx(idx) for idx in indices]
    label = str(site.get("label", ""))

    # New SMARTS may mark the actual proton-transfer atom with an atom map in
    # the query. Prefer that declaration to element-based inference. Legacy
    # patterns without a mapped centre continue through the rules below.
    # Detector records use ``center_atom_set``. Joint-network records copy
    # those atoms into the immutable ``center_indices`` field before walking
    # through other protonation states. Reading both representations is
    # essential: otherwise explicit O centres in N-oxides and explicit N
    # centres in tetrazoles are silently lost after the first network step.
    raw_explicit_centers = site.get(
        "center_atom_set", site.get("center_indices", ())
    )
    explicit_centers = tuple(sorted(
        int(idx) for idx in raw_explicit_centers if int(idx) in indices
    ))
    if explicit_centers:
        return explicit_centers

    if label == "pyrrole" and member_form == "base_form":
        # Pyrrole's thermodynamic conjugate acid is formed at an alpha carbon
        # (C2/C5), while the positive charge can be drawn on N by resonance.
        # Exclude a fusion atom with no hydrogen: changing that atom to sp3
        # would break two rings and is not the ordinary pyrrole pKaH process.
        nitrogens = [atom for atom in atoms if atom.GetAtomicNum() == 7]
        if len(nitrogens) != 1:
            return ()
        nitrogen = nitrogens[0]
        alpha_carbons = [
            neighbor
            for neighbor in nitrogen.GetNeighbors()
            if neighbor.GetIdx() in indices
            and neighbor.GetAtomicNum() == 6
            and (
                neighbor.GetTotalNumHs() > 0
                or mol.GetRingInfo().NumAtomRings(neighbor.GetIdx()) <= 1
            )
        ]
        return tuple(atom.GetIdx() for atom in alpha_carbons)

    if label == "pyrrolium" and member_form == "acid_form":
        # In the explicit C2-protonated resonance form the added proton is on
        # the saturated alpha carbon, not on the positively charged nitrogen.
        saturated_alpha = [
            atom
            for atom in atoms
            if atom.GetAtomicNum() == 6
            and atom.GetHybridization() == Chem.HybridizationType.SP3
            and atom.GetTotalNumHs() > 0
            and any(
                neighbor.GetAtomicNum() == 7 and neighbor.GetIdx() in indices
                for neighbor in atom.GetNeighbors()
            )
        ]
        return tuple(atom.GetIdx() for atom in saturated_alpha)

    if family in N_H_ACID_SITE_FAMILIES:
        nitrogens = [atom for atom in atoms if atom.GetAtomicNum() == 7]
        if member_form == "acid_form":
            preferred = [atom for atom in nitrogens if atom.GetTotalNumHs() > 0]
        else:
            preferred = [atom for atom in nitrogens if atom.GetFormalCharge() < 0]
        # Do not invent an N-H acidity coordinate for an N-substituted azole.
        # A genuine acid form must contain N-H and a genuine conjugate base
        # must contain a negatively charged N in at least one resonance form.
        return tuple(atom.GetIdx() for atom in preferred)

    if family in ACIDIC_SITE_FAMILIES:
        hetero = [atom for atom in atoms if atom.GetAtomicNum() in {8, 16}]
        if member_form == "acid_form":
            preferred = [atom for atom in hetero if atom.GetTotalNumHs() > 0]
        else:
            preferred = [atom for atom in hetero if atom.GetFormalCharge() < 0]
        return tuple(atom.GetIdx() for atom in (preferred or hetero))

    if family in BASIC_SITE_FAMILIES:
        nitrogens = [atom for atom in atoms if atom.GetAtomicNum() == 7]
        if member_form == "acid_form":
            positive = [
                atom for atom in nitrogens
                if atom.GetFormalCharge() > 0 and atom.GetTotalNumHs() > 0
            ]
            return tuple(atom.GetIdx() for atom in positive)

        acceptors = [
            atom for atom in nitrogens
            if atom.GetFormalCharge() <= 0
            and not (atom.GetIsAromatic() and atom.GetTotalNumHs() > 0)
            # An aromatic N with three heavy-atom neighbours is the
            # substituted pyrrole-like atom. Its lone pair belongs to the
            # aromatic sextet; the degree-two pyridine-like N is the ordinary
            # Brønsted basic centre.
            and not (atom.GetIsAromatic() and atom.GetDegree() > 2)
        ]
        if family in {"imine_iminium", "amidine_like", "guanidine_like"}:
            imine_like = [atom for atom in acceptors if _has_double_bond_to_carbon(atom)]
            if imine_like:
                return tuple(atom.GetIdx() for atom in imine_like)
        aromatic_acceptors = [atom for atom in acceptors if atom.GetIsAromatic()]
        if aromatic_acceptors:
            return tuple(atom.GetIdx() for atom in aromatic_acceptors)
        return tuple(atom.GetIdx() for atom in acceptors)

    return ()


def _set_protonation(mol: Chem.Mol, atom_idx: int, protonated: bool) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(atom_idx)
    total_h = int(atom.GetTotalNumHs())

    if protonated:
        if atom.GetFormalCharge() > 0:
            return _copy_and_sanitize(rw.GetMol())
        atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(total_h + 1)
    else:
        if total_h < 1:
            return None
        atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(total_h - 1)

    atom.UpdatePropertyCache(strict=False)
    return _copy_and_sanitize(rw.GetMol())


def _pyrrole_carbon_path(
    mol: Chem.Mol,
    site_indices: Sequence[int],
    nitrogen_idx: int,
    center_idx: int,
) -> Optional[List[int]]:
    """Return C2-C3-C4-C5, avoiding the shorter path through pyrrole N."""
    allowed = set(site_indices) - {nitrogen_idx}
    other_alpha = [
        atom.GetIdx()
        for atom in mol.GetAtomWithIdx(nitrogen_idx).GetNeighbors()
        if atom.GetIdx() in allowed and atom.GetIdx() != center_idx
    ]
    if len(other_alpha) != 1:
        return None
    target_idx = other_alpha[0]

    def walk(current: int, visited: set[int]) -> Optional[List[int]]:
        if current == target_idx:
            return [current]
        for neighbor in mol.GetAtomWithIdx(current).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in allowed or neighbor_idx in visited:
                continue
            suffix = walk(neighbor_idx, visited | {neighbor_idx})
            if suffix is not None:
                return [current, *suffix]
        return None

    path = walk(center_idx, {center_idx})
    if path is None or len(path) != 4:
        return None
    return path


def _fused_ring_system(
    mol: Chem.Mol,
    seed_indices: Sequence[int],
    *,
    aromatic_only: bool = False,
) -> Tuple[set[int], set[Tuple[int, int]]]:
    """Return rings fused to the seed ring and all bonds in those rings."""
    rings = [tuple(int(idx) for idx in ring) for ring in mol.GetRingInfo().AtomRings()]
    component = set(int(idx) for idx in seed_indices)
    selected: List[Tuple[int, ...]] = []
    changed = True
    while changed:
        changed = False
        for ring in rings:
            if ring in selected or len(component.intersection(ring)) < 2:
                continue
            ring_pairs = list(zip(ring, ring[1:] + ring[:1]))
            if aromatic_only and not all(
                mol.GetBondBetweenAtoms(begin, end) is not None
                and mol.GetBondBetweenAtoms(begin, end).GetIsAromatic()
                for begin, end in ring_pairs
            ):
                continue
            selected.append(ring)
            component.update(ring)
            changed = True

    ring_bonds: set[Tuple[int, int]] = set()
    for ring in selected:
        for begin, end in zip(ring, ring[1:] + ring[:1]):
            ring_bonds.add(tuple(sorted((int(begin), int(end)))))
    return component, ring_bonds


def _pyrrole_protonated_double_bonds(
    mol: Chem.Mol,
    component: set[int],
    ring_bonds: set[Tuple[int, int]],
    nitrogen_idx: int,
    center_idx: int,
) -> Optional[set[Tuple[int, int]]]:
    """Find a valence-valid Kekule matching after pyrrole C protonation."""
    # Determine which ring atoms still need one pi-bond after converting every
    # bond in the aromatic component to a single bond and applying the proton
    # transfer. Counting actual non-ring bond order is essential for fused
    # lactams: a carbonyl-bearing ring carbon already has its full valence and
    # must not be forced into an additional ring double bond.
    required: set[int] = set()
    for idx in component:
        atom = mol.GetAtomWithIdx(idx)
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge() + (1 if idx == nitrogen_idx else 0)
        if atomic_num == 6:
            target_valence = 4
        elif atomic_num == 7:
            target_valence = 4 if formal_charge > 0 else (2 if formal_charge < 0 else 3)
        elif atomic_num == 8:
            target_valence = 3 if formal_charge > 0 else (1 if formal_charge < 0 else 2)
        elif atomic_num == 16:
            target_valence = 2
        else:
            return None

        hydrogen_valence = int(atom.GetTotalNumHs()) + (1 if idx == center_idx else 0)
        bond_valence = 0.0
        for bond in atom.GetBonds():
            pair = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            bond_valence += 1.0 if pair in ring_bonds else float(bond.GetBondTypeAsDouble())
        deficit = float(target_valence) - (float(hydrogen_valence) + bond_valence)
        if idx == center_idx:
            if abs(deficit) > 1.0e-6:
                return None
            continue
        rounded = int(round(deficit))
        if abs(deficit - rounded) > 1.0e-6 or rounded not in {0, 1}:
            return None
        if rounded == 1:
            required.add(idx)

    adjacency: Dict[int, List[int]] = {idx: [] for idx in required}
    for begin, end in ring_bonds:
        if begin in required and end in required:
            adjacency[begin].append(end)
            adjacency[end].append(begin)
    for neighbors in adjacency.values():
        neighbors.sort()

    def match(remaining: set[int]) -> Optional[set[Tuple[int, int]]]:
        if not remaining:
            return set()
        begin = min(remaining)
        for end in adjacency.get(begin, []):
            if end not in remaining:
                continue
            suffix = match(remaining - {begin, end})
            if suffix is not None:
                return {tuple(sorted((begin, end))), *suffix}
        return None

    return match(required)


def _set_pyrrole_c2_protonation(
    mol: Chem.Mol,
    site: Dict,
    center_idx: int,
    protonated: bool,
) -> Optional[Chem.Mol]:
    """Interconvert pyrrole and its C2-protonated pyrrolium resonance form."""
    site_indices = _site_indices(site, mol)
    nitrogens = [idx for idx in site_indices if mol.GetAtomWithIdx(idx).GetAtomicNum() == 7]
    if len(nitrogens) != 1:
        return None
    nitrogen_idx = nitrogens[0]
    carbon_path = _pyrrole_carbon_path(
        mol,
        site_indices,
        nitrogen_idx=nitrogen_idx,
        center_idx=center_idx,
    )
    if carbon_path is None:
        return None

    rw = Chem.RWMol(mol)
    nitrogen = rw.GetAtomWithIdx(nitrogen_idx)
    center = rw.GetAtomWithIdx(center_idx)
    nitrogen_h = int(nitrogen.GetTotalNumHs())
    center_h = int(center.GetTotalNumHs())

    stored_component = {
        int(idx) for idx in site.get("aromatic_component_indices", ())
    }
    stored_ring_bonds = {
        tuple(sorted((int(pair[0]), int(pair[1]))))
        for pair in site.get("aromatic_component_bonds", ())
        if len(pair) == 2
    }

    if protonated:
        if nitrogen.GetFormalCharge() != 0:
            return None
        component, ring_bonds = (
            (stored_component, stored_ring_bonds)
            if stored_component and stored_ring_bonds
            else _fused_ring_system(rw, site_indices, aromatic_only=True)
        )
        double_bonds = _pyrrole_protonated_double_bonds(
            rw, component, ring_bonds, nitrogen_idx, center_idx
        )
        if not ring_bonds or double_bonds is None:
            return None
        for begin_idx, end_idx in ring_bonds:
            bond = rw.GetBondBetweenAtoms(begin_idx, end_idx)
            if bond is None:
                return None
            bond.SetIsAromatic(False)
            bond.SetBondType(
                Chem.BondType.DOUBLE
                if (begin_idx, end_idx) in double_bonds
                else Chem.BondType.SINGLE
            )
        for idx in component:
            rw.GetAtomWithIdx(idx).SetIsAromatic(False)
        nitrogen.SetFormalCharge(1)
        nitrogen.SetNoImplicit(True)
        nitrogen.SetNumExplicitHs(nitrogen_h)
        center.SetNoImplicit(True)
        center.SetNumExplicitHs(center_h + 1)
    else:
        if nitrogen.GetFormalCharge() <= 0 or center_h < 1:
            return None
        nitrogen.SetFormalCharge(nitrogen.GetFormalCharge() - 1)
        nitrogen.SetNoImplicit(True)
        nitrogen.SetNumExplicitHs(nitrogen_h)
        center.SetNoImplicit(True)
        center.SetNumExplicitHs(center_h - 1)
        component, ring_bonds = (
            (stored_component, stored_ring_bonds)
            if stored_component and stored_ring_bonds
            else _fused_ring_system(rw, site_indices)
        )
        if not ring_bonds:
            return None
        for idx in component:
            rw.GetAtomWithIdx(idx).SetIsAromatic(True)
        for begin_idx, end_idx in ring_bonds:
            bond = rw.GetBondBetweenAtoms(begin_idx, end_idx)
            if bond is None:
                return None
            bond.SetBondType(Chem.BondType.AROMATIC)
            bond.SetIsAromatic(True)

    for atom in rw.GetAtoms():
        atom.UpdatePropertyCache(strict=False)
    return _copy_and_sanitize(rw.GetMol())


def _transform_site_protonation(
    mol: Chem.Mol,
    site: Dict,
    center_idx: int,
    protonated: bool,
) -> Optional[Chem.Mol]:
    if str(site.get("label", "")) in {"pyrrole", "pyrrolium"}:
        return _set_pyrrole_c2_protonation(mol, site, center_idx, protonated)
    return _set_protonation(mol, center_idx, protonated)


def _enumerate_tautomers_with_status(
    mol: Chem.Mol,
    max_tautomers: int = 16,
) -> Tuple[Tuple[str, ...], bool]:
    if max_tautomers < 1:
        raise ValueError("max_tautomers must be at least one")

    target_charge = formal_charge(mol)
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers)
    enumerator.SetMaxTransforms(max(100, max_tautomers * 20))

    smiles = set()
    truncated = False
    with rdBase.BlockLogs():
        try:
            result = enumerator.Enumerate(mol)
            tautomers: Iterable[Chem.Mol] = result
            truncated = str(result.status) != "Completed"
        except Exception:
            tautomers = (mol,)
        for tautomer in tautomers:
            if formal_charge(tautomer) != target_charge:
                continue
            sanitized = _copy_and_sanitize(tautomer)
            if sanitized is not None:
                smiles.add(mapped_smiles(sanitized))
            if len(smiles) >= max_tautomers:
                break
    if not smiles:
        smiles.add(mapped_smiles(mol))
    return tuple(sorted(smiles)), truncated


def enumerate_tautomers(mol: Chem.Mol, max_tautomers: int = 16) -> Tuple[str, ...]:
    """Enumerate charge-preserving tautomers and return deterministic mapped SMILES."""
    return _enumerate_tautomers_with_status(mol, max_tautomers)[0]


def _unique_molecules(mols: Iterable[Chem.Mol]) -> List[Chem.Mol]:
    unique: Dict[str, Chem.Mol] = {}
    for mol in mols:
        unique.setdefault(mapped_smiles(mol), mol)
    return [unique[key] for key in sorted(unique)]


def generate_conjugate_microstates(
    mol: Chem.Mol,
    site: Dict,
    family: str,
    member_form: str,
    max_tautomers: int = 16,
) -> MicrostateResult:
    """Generate the full-molecule HA and A microstate ensembles for one site.

    Other ionizable sites remain in their input form.  The returned acid state
    must have total formal charge exactly one unit above the base state.
    """
    if member_form not in {"acid_form", "base_form"}:
        return MicrostateResult((), (), (), (), member_form, 0, 0, "invalid", "unknown member form")
    if family not in ACIDIC_SITE_FAMILIES | BASIC_SITE_FAMILIES:
        return MicrostateResult((), (), (), (), member_form, 0, 0, "invalid", "unsupported family")

    mapped = ensure_heavy_atom_maps(mol)
    indices = _site_indices(site, mapped)
    centers = find_protonation_centers(mapped, site, family, member_form)
    site_maps = _site_map_numbers(mapped, indices)
    center_maps = _site_map_numbers(mapped, centers)
    if not centers:
        return MicrostateResult((), (), site_maps, (), member_form, 0, 0, "invalid", "no protonation center")

    if member_form == "acid_form":
        acid_mols = [mapped]
        base_mols = [
            candidate
            for idx in centers
            if (candidate := _transform_site_protonation(mapped, site, idx, False))
        ]
    else:
        base_mols = [mapped]
        acid_mols = [
            candidate
            for idx in centers
            if (candidate := _transform_site_protonation(mapped, site, idx, True))
        ]

    acid_mols = _unique_molecules(acid_mols)
    base_mols = _unique_molecules(base_mols)
    if not acid_mols or not base_mols:
        return MicrostateResult((), (), site_maps, center_maps, member_form, 0, 0, "invalid", "conjugate transform failed")

    acid_charges = {formal_charge(candidate) for candidate in acid_mols}
    base_charges = {formal_charge(candidate) for candidate in base_mols}
    if len(acid_charges) != 1 or len(base_charges) != 1:
        return MicrostateResult((), (), site_maps, center_maps, member_form, 0, 0, "invalid", "inconsistent charges")
    acid_charge = next(iter(acid_charges))
    base_charge = next(iter(base_charges))
    if acid_charge - base_charge != 1:
        return MicrostateResult(
            (), (), site_maps, center_maps, member_form, acid_charge, base_charge,
            "invalid", "acid/base charge difference is not +1",
        )

    acid_enumerations = [
        _enumerate_tautomers_with_status(candidate, max_tautomers=max_tautomers)
        for candidate in acid_mols
    ]
    base_enumerations = [
        _enumerate_tautomers_with_status(candidate, max_tautomers=max_tautomers)
        for candidate in base_mols
    ]
    acid_direct = [mapped_smiles(candidate) for candidate in acid_mols]
    base_direct = [mapped_smiles(candidate) for candidate in base_mols]
    acid_all_set = {state for states, _ in acid_enumerations for state in states}
    base_all_set = {state for states, _ in base_enumerations for state in states}
    # The direct one-proton transform is the chemically interpretable
    # reference.  RDKit may also enumerate remote prototropic tautomers; keep
    # those in the ensemble, but never let lexical SMILES order promote one to
    # reference status.
    acid_all = list(dict.fromkeys(acid_direct)) + sorted(acid_all_set - set(acid_direct))
    base_all = list(dict.fromkeys(base_direct)) + sorted(base_all_set - set(base_direct))
    acid_truncated = any(flag for _, flag in acid_enumerations) or len(acid_all) > max_tautomers
    base_truncated = any(flag for _, flag in base_enumerations) or len(base_all) > max_tautomers
    acid_states = acid_all[:max_tautomers]
    base_states = base_all[:max_tautomers]

    note = "pyrrole conjugate acid enumerated by alpha-carbon (C2/C5) protonation" if str(site.get("label", "")) in {"pyrrole", "pyrrolium"} else ""
    if len(center_maps) > 1:
        prefix = f"{note}; " if note else ""
        note = prefix + "multiple protonation centers retained as an enumerated ensemble"
    return MicrostateResult(
        tuple(acid_states),
        tuple(base_states),
        site_maps,
        center_maps,
        member_form,
        acid_charge,
        base_charge,
        "ok",
        note,
        acid_truncated,
        base_truncated,
    )


def enumerate_joint_protonation_network(
    mol: Chem.Mol,
    sites: Sequence[Dict],
    max_protonation_states: int = DEFAULT_MAX_PROTONATION_STATES,
    max_tautomers_per_state: int = 8,
) -> Dict[str, object]:
    """Enumerate coupled protonation coordinates and their one-proton edges.

    Ordinary sites have levels 0/1.  Amphoteric azoles expose two transitions
    on one shared coordinate (-1/0 and 0/+1), which prevents the impossible
    states produced by treating those transitions as independent switches.
    """
    if max_protonation_states < 1:
        raise ValueError("max_protonation_states must be at least one")
    if max_tautomers_per_state < 1:
        raise ValueError("max_tautomers_per_state must be at least one")

    mapped = ensure_heavy_atom_maps(mol)
    site_records = []
    for site in sites:
        family = str(site.get("pair_family", site.get("family", "")))
        member_form = str(site.get("pair_member_form", ""))
        if family not in ACIDIC_SITE_FAMILIES | BASIC_SITE_FAMILIES:
            continue
        if member_form not in {"acid_form", "base_form"}:
            continue
        indices = _site_indices(site, mapped)
        centers = find_protonation_centers(mapped, site, family, member_form)
        if not centers:
            continue
        site_maps = _site_map_numbers(mapped, indices)
        center_maps = _site_map_numbers(mapped, centers)
        aromatic_component_indices: Tuple[int, ...] = ()
        aromatic_component_bonds: Tuple[Tuple[int, int], ...] = ()
        if str(site.get("label", "")) in {"pyrrole", "pyrrolium"}:
            component, component_bonds = _fused_ring_system(
                mapped, indices, aromatic_only=True
            )
            aromatic_component_indices = tuple(sorted(component))
            aromatic_component_bonds = tuple(sorted(component_bonds))
        site_records.append(
            {
                "label": str(site.get("label", "")),
                "family": family,
                "input_member_form": member_form,
                "atom_indices": indices,
                "center_indices": centers,
                "aromatic_component_indices": aromatic_component_indices,
                "aromatic_component_bonds": aromatic_component_bonds,
                "atom_maps": site_maps,
                "center_maps": center_maps,
                "coupling_group": str(site.get("coupling_group", family)),
                "acid_level": int(site.get("acid_level", 1)),
                "base_level": int(site.get("base_level", 0)),
            }
        )

    site_records.sort(
        key=lambda record: (
            record["center_maps"],
            record["atom_maps"],
            record["family"],
            record["label"],
        )
    )
    for idx, record in enumerate(site_records):
        record["site_id"] = f"site_{idx + 1}"

    grouped_records: Dict[Tuple[str, Tuple[int, ...]], List[Dict]] = {}
    for record in site_records:
        key = (record["coupling_group"], record["atom_maps"])
        grouped_records.setdefault(key, []).append(record)
    ordered_groups = sorted(grouped_records.items(), key=lambda item: (item[0][1], item[0][0]))
    for group_index, (_, records) in enumerate(ordered_groups, start=1):
        group_id = f"group_{group_index}"
        for record in records:
            record["coupling_group_id"] = group_id

    def input_level(records: Sequence[Dict]) -> Optional[int]:
        levels = {
            int(record["acid_level"] if record["input_member_form"] == "acid_form" else record["base_level"])
            for record in records
        }
        return next(iter(levels)) if len(levels) == 1 else None

    def enumerate_group_levels(
        start_mol: Chem.Mol,
        records: Sequence[Dict],
        start_level: int,
    ) -> List[Tuple[Chem.Mol, int, Dict[str, int]]]:
        queue: List[Tuple[Chem.Mol, int, Dict[str, int]]] = [(Chem.Mol(start_mol), start_level, {})]
        unique: Dict[Tuple[str, int], Tuple[Chem.Mol, int, Dict[str, int]]] = {}
        while queue:
            state_mol, level, chosen = queue.pop(0)
            state_key = (mapped_smiles(state_mol), level)
            if state_key in unique:
                continue
            unique[state_key] = (state_mol, level, chosen)
            for record in records:
                acid_level = int(record["acid_level"])
                base_level = int(record["base_level"])
                if level == base_level:
                    target_level, current_form, protonated = acid_level, "base_form", True
                elif level == acid_level:
                    target_level, current_form, protonated = base_level, "acid_form", False
                else:
                    continue
                centers = find_protonation_centers(
                    state_mol, record, str(record["family"]), current_form
                )
                for center_idx in centers:
                    candidate = _transform_site_protonation(
                        state_mol, record, center_idx, protonated
                    )
                    if candidate is None:
                        continue
                    next_chosen = dict(chosen)
                    next_chosen[str(record["site_id"])] = int(
                        state_mol.GetAtomWithIdx(center_idx).GetAtomMapNum()
                    )
                    queue.append((candidate, target_level, next_chosen))
        return [unique[key] for key in sorted(unique, key=lambda value: (value[1], value[0]))]

    working: List[Tuple[Chem.Mol, Dict[str, int], Dict[str, int]]] = [(mapped, {}, {})]
    truncated = False
    incomplete_sites = set()
    for _, records in ordered_groups:
        group_id = str(records[0]["coupling_group_id"])
        start_level = input_level(records)
        if start_level is None:
            incomplete_sites.update(str(record["site_id"]) for record in records)
            continue
        next_states = []
        allowed_levels = {
            int(value)
            for record in records
            for value in (record["acid_level"], record["base_level"])
        }
        for state_mol, levels, chosen_centers in working:
            local_states = enumerate_group_levels(state_mol, records, start_level)
            reached = {level for _, level, _ in local_states}
            if reached != allowed_levels:
                incomplete_sites.update(str(record["site_id"]) for record in records)
            for candidate, level, local_centers in local_states:
                next_states.append((
                    candidate,
                    {**levels, group_id: level},
                    {**chosen_centers, **local_centers},
                ))

        unique = {}
        for state_mol, levels, chosen_centers in next_states:
            key = (mapped_smiles(state_mol), tuple(sorted(levels.items())))
            unique.setdefault(key, (state_mol, levels, chosen_centers))
        ordered = [unique[key] for key in sorted(unique)]
        if len(ordered) > max_protonation_states:
            truncated = True
            ordered = ordered[:max_protonation_states]
        working = ordered

    nodes = []
    minimum_levels = {
        str(records[0]["coupling_group_id"]): min(
            int(value)
            for record in records
            for value in (record["acid_level"], record["base_level"])
        )
        for _, records in ordered_groups
    }
    for state_mol, levels, chosen_centers in working:
        forms = {}
        for record in site_records:
            level = levels.get(str(record["coupling_group_id"]))
            if level == int(record["acid_level"]):
                forms[str(record["site_id"])] = "acid_form"
            elif level == int(record["base_level"]):
                forms[str(record["site_id"])] = "base_form"
            else:
                forms[str(record["site_id"])] = "outside_transition"
        reference = mapped_smiles(state_mol)
        tautomers, tautomer_truncated = _enumerate_tautomers_with_status(
            state_mol,
            max_tautomers=max_tautomers_per_state,
        )
        signature = reference + "|" + repr(sorted(levels.items()))
        node_id = "state_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
        nodes.append(
            {
                "node_id": node_id,
                "reference_atom_mapped_smiles": reference,
                "formal_charge": formal_charge(state_mol),
                "protonated_site_count": sum(
                    int(level) - minimum_levels[group_id]
                    for group_id, level in levels.items()
                ),
                "site_forms": dict(sorted(forms.items())),
                "group_levels": dict(sorted(levels.items())),
                "chosen_protonation_center_maps": dict(sorted(chosen_centers.items())),
                "tautomers": list(tautomers),
                "tautomer_enumeration_truncated": tautomer_truncated,
            }
        )
    nodes.sort(key=lambda node: (node["protonated_site_count"], node["node_id"]))

    edges = []
    for record in site_records:
        site_id = record["site_id"]
        group_id = str(record["coupling_group_id"])
        buckets: Dict[Tuple[Tuple[str, int], ...], Dict[str, List[Dict]]] = {}
        for node in nodes:
            levels = node["group_levels"]
            level = levels.get(group_id)
            if level == int(record["acid_level"]):
                member_form = "acid_form"
            elif level == int(record["base_level"]):
                member_form = "base_form"
            else:
                continue
            other = tuple(sorted((key, value) for key, value in levels.items() if key != group_id))
            buckets.setdefault(other, {"acid_form": [], "base_form": []})[member_form].append(node)
        for other, forms in sorted(buckets.items()):
            for acid_node in forms["acid_form"]:
                for base_node in forms["base_form"]:
                    if int(acid_node["formal_charge"]) - int(base_node["formal_charge"]) != 1:
                        continue
                    signature = f"{site_id}|{acid_node['node_id']}|{base_node['node_id']}"
                    edges.append(
                        {
                            "edge_id": "edge_" + hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12],
                            "site_id": site_id,
                            "acid_node_id": acid_node["node_id"],
                            "base_node_id": base_node["node_id"],
                            "other_site_forms": dict(other),
                            "coupling_group_id": group_id,
                            "proton_delta": 1,
                            "charge_delta": 1,
                        }
                    )
    edges.sort(key=lambda edge: (edge["site_id"], edge["acid_node_id"], edge["base_node_id"]))

    public_sites = [
        {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in record.items()
            if key not in {
                "atom_indices",
                "center_indices",
                "aromatic_component_indices",
                "aromatic_component_bonds",
            }
        }
        for record in site_records
    ]
    return {
        "sites": public_sites,
        "nodes": nodes,
        "edges": edges,
        "protonation_state_enumeration_truncated": truncated,
        "incomplete_site_ids": sorted(incomplete_sites),
        "max_protonation_states": int(max_protonation_states),
        "max_tautomers_per_state": int(max_tautomers_per_state),
    }
