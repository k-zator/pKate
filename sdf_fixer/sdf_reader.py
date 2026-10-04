"""
sdf_reader.py
=============
Parse SDF files and extract per-record molecule objects together with all
associated properties (SMILES, pKa, temperature, ionization type, formal
charges, etc.).

The reader is intentionally *read-only* — it never mutates the molecule.
Protonation-state fixing is handled downstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from rdkit import Chem  # type: ignore


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SDFRecord:
    """One molecule entry from an SDF file."""

    record_index: int
    mol: Chem.Mol
    # The SMILES stored in the SDF property block (as the author wrote it)
    original_smiles: str
    # Canonical SMILES regenerated from the molblock coordinates / graph
    canonical_smiles: str
    pka_value: Optional[float] = None
    temperature: Optional[float] = None
    name: Optional[str] = None
    ionization_type: Optional[str] = None
    pka_type: Optional[str] = None
    # Formal charge on the molecule as encoded in the molblock
    formal_charge: int = 0
    # Per-atom charges extracted from M CHG lines (atom_idx -> charge)
    atom_charges: Dict[int, int] = field(default_factory=dict)
    # All raw SDF properties for anything we didn't explicitly parse
    raw_properties: Dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SDFRecord(idx={self.record_index}, name={self.name!r}, "
            f"smiles={self.original_smiles!r}, pKa={self.pka_value}, "
            f"charge={self.formal_charge})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
        return v
    except (TypeError, ValueError):
        return None


def _safe_str(value: Optional[str]) -> Optional[str]:
    """Return *None* for empty / 'nan' strings."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _extract_atom_charges(mol: Chem.Mol) -> Dict[int, int]:
    """Return {atom_index: formal_charge} for every charged atom."""
    charges: Dict[int, int] = {}
    for atom in mol.GetAtoms():
        fc = atom.GetFormalCharge()
        if fc != 0:
            charges[atom.GetIdx()] = int(fc)
    return charges


def _molecule_formal_charge(mol: Chem.Mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------

def read_sdf(sdf_path: str, *, removeHs: bool = False) -> List[SDFRecord]:
    """
    Read an SDF file and return a list of `SDFRecord` objects.

    Parameters
    ----------
    sdf_path : str
        Path to the ``.sdf`` file.
    removeHs : bool
        If *True* RDKit will strip explicit hydrogens while parsing.
        Default *False* to preserve them (needed for charge analysis).

    Returns
    -------
    list[SDFRecord]
        One entry per successfully-parsed molecule.
    """
    records: List[SDFRecord] = []

    with open(sdf_path, "rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, removeHs=removeHs)
        for record_idx, mol in enumerate(supplier):
            if mol is None:
                continue

            # ---------- properties ----------
            props: Dict[str, str] = {}
            for prop_name in mol.GetPropNames():
                props[prop_name] = mol.GetProp(prop_name)

            original_smiles = _safe_str(props.get("SMILES")) or ""
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            pka_value = _safe_float(props.get("pKa"))
            temperature = _safe_float(props.get("temp"))
            name = _safe_str(props.get("name"))

            ionization_type = _safe_str(props.get("ionization_type"))
            pka_type = _safe_str(props.get("type"))

            # ---------- charges ----------
            atom_charges = _extract_atom_charges(mol)
            formal_charge = _molecule_formal_charge(mol)

            records.append(
                SDFRecord(
                    record_index=record_idx,
                    mol=mol,
                    original_smiles=original_smiles,
                    canonical_smiles=canonical_smiles,
                    pka_value=pka_value,
                    temperature=temperature,
                    name=name,
                    ionization_type=ionization_type,
                    pka_type=pka_type,
                    formal_charge=formal_charge,
                    atom_charges=atom_charges,
                    raw_properties=props,
                )
            )

    return records


def iter_sdf(sdf_path: str, *, removeHs: bool = False) -> Iterable[SDFRecord]:
    """
    Lazy iterator version of `read_sdf` — yields one `SDFRecord` at a time.
    Useful for very large files.
    """
    with open(sdf_path, "rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, removeHs=removeHs)
        for record_idx, mol in enumerate(supplier):
            if mol is None:
                continue

            props: Dict[str, str] = {}
            for prop_name in mol.GetPropNames():
                props[prop_name] = mol.GetProp(prop_name)

            original_smiles = _safe_str(props.get("SMILES")) or ""
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

            pka_value = _safe_float(props.get("pKa"))
            temperature = _safe_float(props.get("temp"))
            name = _safe_str(props.get("name"))

            ionization_type = _safe_str(props.get("ionization_type"))
            pka_type = _safe_str(props.get("type"))

            atom_charges = _extract_atom_charges(mol)
            formal_charge = _molecule_formal_charge(mol)

            yield SDFRecord(
                record_index=record_idx,
                mol=mol,
                original_smiles=original_smiles,
                canonical_smiles=canonical_smiles,
                pka_value=pka_value,
                temperature=temperature,
                name=name,
                ionization_type=ionization_type,
                pka_type=pka_type,
                formal_charge=formal_charge,
                atom_charges=atom_charges,
                raw_properties=props,
            )
