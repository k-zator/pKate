#!/usr/bin/env python3
"""Structure-dependent protein-environment corrections for ligand pKa terms.

The aqueous pKate model remains the reference state.  This module extracts a
small, auditable set of local descriptors from a supplied protein--ligand
complex, predicts an environmental ``delta pKa``, and adds that residual to
the Stage 2 one-body free-energy term.  Pair terms are retained and the
macroscopic ladder is recomputed from the corrected free-energy model.

The first model is calibrated on experimental protein-residue pKas.  Transfer
to drug-like ligand groups is therefore deliberately shrunk and its
uncertainty is reported separately from aqueous-model uncertainty.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import pickle
import shlex
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS

from stage2_free_energy_coupling import (
    DEFAULT_TEMPERATURE_K,
    R_KJ_MOL_K,
    contextual_edge_pkas,
    macro_pka_values_from_free_energy,
    pair_coupling_map,
)


PROTEIN_ENVIRONMENT_SCHEMA_VERSION = "1.1.0"
PROTEIN_ENVIRONMENT_FEATURE_SCHEMA_VERSION = "1.0.0"
PROTEIN_ENVIRONMENT_METHOD = (
    "pkadr_grouped_residual_model_with_ligand_analogue_transfer_shrinkage"
)
DEFAULT_ENVIRONMENT_MODEL = (
    "data/processed/ml_models_experimental_only/protein_environment/"
    "protein_environment_model.pkl"
)

WATER_RESIDUES = {"HOH", "WAT", "DOD"}
METAL_ELEMENTS = {"CA", "CO", "CU", "FE", "MG", "MN", "NI", "ZN"}
POLAR_ELEMENTS = {"N", "O", "P", "S"}
STANDARD_AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
}

# Coarse atom-charge proxies are intentionally sparse.  They represent the
# signed local electrostatic field without pretending to be force-field
# partial charges or a Poisson--Boltzmann calculation.
NEGATIVE_PROTEIN_ATOMS = {
    ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLU", "OE1"), ("GLU", "OE2"),
}
POSITIVE_PROTEIN_ATOMS = {
    ("LYS", "NZ"),
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
}

FEATURE_COLUMNS = (
    "site_is_acidic",
    "protein_heavy_atoms_4a",
    "protein_heavy_atoms_6a",
    "protein_heavy_atoms_8a",
    "protein_polar_atoms_4a",
    "protein_polar_atoms_6a",
    "negative_atoms_6a",
    "positive_atoms_6a",
    "signed_charge_over_distance",
    "hbond_compatible_atoms_3_5a",
    "nearby_titratable_residues_6a",
    "nearby_titratable_residues_10a",
    "waters_4a",
    "waters_6a",
    "organic_heterogen_atoms_6a",
    "metals_4a",
    "min_protein_distance",
    "min_water_distance",
    "min_organic_heterogen_distance",
    "min_metal_distance",
    "chemistry_carboxyl",
    "chemistry_imidazole",
    "chemistry_amine",
    "chemistry_thiol",
    "chemistry_phenol",
)

CHEMISTRY_IS_ACIDIC = {
    "carboxyl": True,
    "imidazole": False,
    "amine": False,
    "thiol": True,
    "phenol": True,
}

PROTEIN_RESIDUE_CHEMISTRY = {
    "ASP": "carboxyl",
    "GLU": "carboxyl",
    "HIS": "imidazole",
    "LYS": "amine",
    "CYS": "thiol",
    "TYR": "phenol",
}
PROTEIN_RESIDUE_REFERENCE_PKA = {
    "ASP": 3.90,
    "GLU": 4.30,
    "HIS": 6.50,
    "LYS": 10.50,
    "CYS": 8.30,
    "TYR": 10.10,
}
JOINT_PROTEIN_NETWORK_METHOD = (
    "explicit_nearby_binary_residue_nodes_with_centered_screened_coulomb_coupling"
)


@dataclass(frozen=True)
class StructureAtom:
    """One first-model structure atom in author residue numbering."""

    group: str
    atom_name: str
    element: str
    residue_name: str
    chain: str
    residue_id: str
    insertion_code: str
    x: float
    y: float
    z: float

    @property
    def coordinate(self) -> np.ndarray:
        return np.asarray((self.x, self.y, self.z), dtype=float)

    @property
    def residue_key(self) -> Tuple[str, str, str, str]:
        return (
            self.chain,
            self.residue_id,
            self.insertion_code,
            self.residue_name,
        )


@dataclass(frozen=True)
class LigandSelector:
    """Optional residue identity used to remove the ligand from its environment."""

    chain: Optional[str] = None
    residue_name: Optional[str] = None
    residue_id: Optional[str] = None

    def matches(self, atom: StructureAtom) -> bool:
        if self.chain is not None and atom.chain != str(self.chain):
            return False
        if (
            self.residue_name is not None
            and atom.residue_name != str(self.residue_name).upper()
        ):
            return False
        if self.residue_id is not None and atom.residue_id != str(self.residue_id):
            return False
        return any(
            value is not None
            for value in (self.chain, self.residue_name, self.residue_id)
        )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _truthy(value: object) -> bool:
    """Parse serialized booleans without treating the string ``False`` as true."""
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" else path.open("r", encoding="utf-8", errors="replace")


def _parse_pdb(path: Path) -> list[StructureAtom]:
    atoms: list[StructureAtom] = []
    model_number = 0
    with _open_text(path) as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                model_number += 1
                continue
            if record == "ENDMDL" and model_number in {0, 1}:
                break
            if model_number > 1 or record not in {"ATOM", "HETATM"}:
                continue
            if len(line) < 54 or line[16:17] not in {" ", "A", "1"}:
                continue
            try:
                x, y, z = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper()
            if not element:
                element = "".join(ch for ch in atom_name if ch.isalpha())[:1].upper()
            atoms.append(StructureAtom(
                group=record,
                atom_name=atom_name,
                element=element,
                residue_name=line[17:20].strip().upper(),
                chain=line[21:22].strip(),
                residue_id=line[22:26].strip(),
                insertion_code=line[26:27].strip(),
                x=x,
                y=y,
                z=z,
            ))
    return atoms


def _parse_mmcif(path: Path) -> list[StructureAtom]:
    atoms: list[StructureAtom] = []
    with _open_text(path) as handle:
        lines = iter(handle)
        for line in lines:
            if line.strip() != "loop_":
                continue
            headers: list[str] = []
            first_data = ""
            for candidate in lines:
                stripped = candidate.strip()
                if stripped.startswith("_"):
                    headers.append(stripped.split()[0])
                    continue
                first_data = stripped
                break
            if not headers or not headers[0].startswith("_atom_site."):
                continue
            columns = [header.split(".", 1)[1] for header in headers]
            index = {name: idx for idx, name in enumerate(columns)}
            required = {
                "group_PDB", "type_symbol", "auth_atom_id", "auth_comp_id",
                "auth_asym_id", "auth_seq_id", "pdbx_PDB_ins_code",
                "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num",
            }
            missing = required - set(index)
            if missing:
                raise ValueError(
                    f"mmCIF atom_site loop is missing fields: {sorted(missing)}"
                )

            def consume(row: str) -> bool:
                if not row or row.startswith("#") or row.startswith("_") or row == "loop_":
                    return False
                tokens = shlex.split(row)
                if len(tokens) < len(columns):
                    raise ValueError("Multiline mmCIF atom rows are not supported")
                group = tokens[index["group_PDB"]]
                if group not in {"ATOM", "HETATM"}:
                    return False
                if tokens[index["pdbx_PDB_model_num"]] not in {"1", ".", "?"}:
                    return True
                insertion = tokens[index["pdbx_PDB_ins_code"]]
                if insertion in {".", "?"}:
                    insertion = ""
                atoms.append(StructureAtom(
                    group=group,
                    atom_name=tokens[index["auth_atom_id"]],
                    element=tokens[index["type_symbol"]].upper(),
                    residue_name=tokens[index["auth_comp_id"]].upper(),
                    chain=tokens[index["auth_asym_id"]],
                    residue_id=tokens[index["auth_seq_id"]],
                    insertion_code=insertion,
                    x=float(tokens[index["Cartn_x"]]),
                    y=float(tokens[index["Cartn_y"]]),
                    z=float(tokens[index["Cartn_z"]]),
                ))
                return True

            if first_data:
                consume(first_data)
            for candidate in lines:
                if not consume(candidate.strip()):
                    break
            break
    return atoms


def read_structure_atoms(path: str | Path) -> list[StructureAtom]:
    """Read the first coordinate model from PDB or PDBx/mmCIF."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Protein structure does not exist: {source}")
    name = source.name.lower()
    if name.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz")):
        atoms = _parse_mmcif(source)
    elif name.endswith((".pdb", ".ent", ".pdb.gz", ".ent.gz")):
        atoms = _parse_pdb(source)
    else:
        raise ValueError(
            "Protein structure must be PDB or mmCIF, optionally gzip-compressed"
        )
    if not atoms:
        raise ValueError(f"No first-model atoms were read from {source}")
    return atoms


def _load_ligand_molecule(path: Path) -> Chem.Mol:
    name = path.name.lower()
    molecule: Optional[Chem.Mol]
    if name.endswith((".sdf", ".sd")):
        supplier = Chem.SDMolSupplier(str(path), removeHs=False)
        molecule = next((mol for mol in supplier if mol is not None), None)
    elif name.endswith((".mol", ".mdl")):
        molecule = Chem.MolFromMolFile(str(path), removeHs=False)
    elif name.endswith(".mol2"):
        molecule = Chem.MolFromMol2File(str(path), removeHs=False)
    elif name.endswith(".pdb"):
        molecule = Chem.MolFromPDBFile(str(path), removeHs=False)
    else:
        raise ValueError("Ligand structure must be SDF, MOL, MOL2, or PDB")
    if molecule is None:
        raise ValueError(f"RDKit could not read the ligand structure: {path}")
    if molecule.GetNumConformers() == 0:
        raise ValueError("Ligand structure has no coordinates")
    return molecule


def map_ligand_coordinates(
    query_molecule: Chem.Mol,
    ligand_path: str | Path,
) -> Tuple[Dict[int, np.ndarray], Dict[str, object]]:
    """Map pKate heavy-atom map numbers onto a posed ligand structure."""
    source = Path(ligand_path)
    if not source.is_file():
        raise FileNotFoundError(f"Ligand structure does not exist: {source}")
    query = Chem.RemoveHs(Chem.Mol(query_molecule))
    ligand = Chem.RemoveHs(_load_ligand_molecule(source))
    if query.GetNumAtoms() != ligand.GetNumAtoms():
        raise ValueError(
            "Ligand structure and input SMILES must contain the same heavy atoms: "
            f"SMILES={query.GetNumAtoms()} ligand={ligand.GetNumAtoms()}"
        )
    mapped_query = Chem.Mol(query)
    for idx, atom in enumerate(mapped_query.GetAtoms(), start=1):
        if atom.GetAtomMapNum() <= 0:
            atom.SetAtomMapNum(idx)
    mcs = rdFMCS.FindMCS(
        [mapped_query, ligand],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        timeout=10,
    )
    if mcs.canceled or mcs.numAtoms != mapped_query.GetNumAtoms():
        raise ValueError(
            "Could not establish a complete element/connectivity mapping between "
            "the input SMILES and posed ligand"
        )
    pattern = Chem.MolFromSmarts(mcs.smartsString)
    query_matches = mapped_query.GetSubstructMatches(pattern, uniquify=False)
    ligand_matches = ligand.GetSubstructMatches(pattern, uniquify=False)
    if not query_matches or not ligand_matches:
        raise ValueError("Complete ligand graph mapping produced no atom match")
    query_match = min(query_matches)
    ligand_match = min(ligand_matches)
    conformer = ligand.GetConformer()
    coordinates: Dict[int, np.ndarray] = {}
    for query_index, ligand_index in zip(query_match, ligand_match):
        point = conformer.GetAtomPosition(int(ligand_index))
        atom_map = int(mapped_query.GetAtomWithIdx(int(query_index)).GetAtomMapNum())
        coordinates[atom_map] = np.asarray((point.x, point.y, point.z), dtype=float)
    if len(coordinates) != mapped_query.GetNumAtoms():
        raise ValueError("Ligand mapping did not cover every heavy atom")
    return coordinates, {
        "ligand_structure_sha256": _sha256(source),
        "mapping_method": "full_graph_element_match_bond_order_agnostic",
        "heavy_atom_count": int(len(coordinates)),
        "symmetry_match_count": int(len(query_matches) * len(ligand_matches)),
    }


def residue_site_center(
    atoms: Sequence[StructureAtom],
    *,
    chain: str,
    residue_id: str,
    residue_name: str,
) -> Tuple[np.ndarray, Tuple[str, str, str, str]]:
    """Return the conventional ionization-center centroid for a protein residue."""
    residue_name = str(residue_name).upper()
    candidates = [
        atom for atom in atoms
        if atom.chain == str(chain)
        and atom.residue_id == str(residue_id)
        and atom.residue_name == residue_name
    ]
    if not candidates:
        raise KeyError(f"Missing target residue {chain}:{residue_name}{residue_id}")
    center_names = {
        "ASP": {"OD1", "OD2"},
        "GLU": {"OE1", "OE2"},
        "HIS": {"ND1", "NE2"},
        "LYS": {"NZ"},
        "CYS": {"SG"},
        "TYR": {"OH"},
    }
    selected = [
        atom.coordinate for atom in candidates
        if atom.atom_name in center_names.get(residue_name, set())
    ]
    if not selected:
        raise KeyError(
            f"Target residue {chain}:{residue_name}{residue_id} lacks ionization-center atoms"
        )
    return np.mean(np.asarray(selected), axis=0), candidates[0].residue_key


def ligand_site_center(
    site: Mapping[str, object],
    coordinates_by_map: Mapping[int, np.ndarray],
) -> np.ndarray:
    """Return the centroid of a mapped ligand site's declared center atoms."""
    maps = [int(value) for value in site.get("center_maps", [])]
    if not maps:
        maps = [int(value) for value in site.get("atom_maps", [])]
    missing = [value for value in maps if value not in coordinates_by_map]
    if missing:
        raise KeyError(f"Posed ligand is missing mapped site atoms: {missing}")
    if not maps:
        raise KeyError("Ionizable site has no mapped center atoms")
    return np.mean(np.asarray([coordinates_by_map[value] for value in maps]), axis=0)


def _distance(center: np.ndarray, atom: StructureAtom) -> float:
    return float(np.linalg.norm(center - atom.coordinate))


def _minimum(distances: Iterable[float], cap: float = 12.0) -> float:
    values = list(distances)
    return float(min(min(values), cap)) if values else float(cap)


def environment_descriptors(
    atoms: Sequence[StructureAtom],
    center: np.ndarray,
    *,
    chemistry: str,
    excluded_residue_keys: Iterable[Tuple[str, str, str, str]] = (),
    ligand_coordinates: Iterable[np.ndarray] = (),
    ligand_selector: Optional[LigandSelector] = None,
) -> Dict[str, float]:
    """Calculate deterministic local descriptors around one ionization center."""
    if chemistry not in CHEMISTRY_IS_ACIDIC:
        raise ValueError(f"Unsupported environment chemistry: {chemistry}")
    excluded = set(excluded_residue_keys)
    ligand_xyz = [np.asarray(value, dtype=float) for value in ligand_coordinates]

    environment: list[StructureAtom] = []
    for atom in atoms:
        if atom.element == "H" or atom.residue_key in excluded:
            continue
        # Every descriptor is capped at 12 A, so distant atoms cannot affect
        # the result and should not inflate repeated per-site calculations.
        if _distance(center, atom) > 12.0:
            continue
        if ligand_selector is not None and ligand_selector.matches(atom):
            continue
        if ligand_xyz and min(
            float(np.linalg.norm(atom.coordinate - coord)) for coord in ligand_xyz
        ) <= 0.35:
            continue
        environment.append(atom)

    heterogen_groups: Dict[Tuple[str, str, str, str], list[StructureAtom]] = {}
    for atom in environment:
        if atom.group == "HETATM" and atom.residue_name not in WATER_RESIDUES:
            heterogen_groups.setdefault(atom.residue_key, []).append(atom)
    organic_keys = {
        key for key, group in heterogen_groups.items()
        if any(atom.element == "C" for atom in group)
    }

    protein = [atom for atom in environment if atom.group == "ATOM"]
    water = [
        atom for atom in environment
        if atom.group == "HETATM" and atom.residue_name in WATER_RESIDUES
    ]
    organic = [atom for atom in environment if atom.residue_key in organic_keys]
    metals = [atom for atom in environment if atom.element in METAL_ELEMENTS]
    distances = {id(atom): _distance(center, atom) for atom in environment}

    def count(pool: Sequence[StructureAtom], cutoff: float) -> float:
        return float(sum(distances[id(atom)] <= cutoff for atom in pool))

    polar_protein = [atom for atom in protein if atom.element in POLAR_ELEMENTS]
    negative = [
        atom for atom in protein
        if (atom.residue_name, atom.atom_name) in NEGATIVE_PROTEIN_ATOMS
    ]
    positive = [
        atom for atom in protein
        if (atom.residue_name, atom.atom_name) in POSITIVE_PROTEIN_ATOMS
    ]
    signed_charge = sum(
        -0.5 / max(distances[id(atom)], 1.5) for atom in negative
        if distances[id(atom)] <= 12.0
    ) + sum(
        (1.0 / 3.0) / max(distances[id(atom)], 1.5) for atom in positive
        if distances[id(atom)] <= 12.0
    )
    titratable_keys = {
        atom.residue_key for atom in protein
        if atom.residue_name in {"ASP", "GLU", "HIS", "LYS", "CYS", "TYR", "ARG"}
    }
    titratable_minimum: Dict[Tuple[str, str, str, str], float] = {}
    for atom in protein:
        if atom.residue_key in titratable_keys:
            titratable_minimum[atom.residue_key] = min(
                distances[id(atom)],
                titratable_minimum.get(atom.residue_key, float("inf")),
            )

    result = {
        "site_is_acidic": float(CHEMISTRY_IS_ACIDIC[chemistry]),
        "protein_heavy_atoms_4a": count(protein, 4.0),
        "protein_heavy_atoms_6a": count(protein, 6.0),
        "protein_heavy_atoms_8a": count(protein, 8.0),
        "protein_polar_atoms_4a": count(polar_protein, 4.0),
        "protein_polar_atoms_6a": count(polar_protein, 6.0),
        "negative_atoms_6a": count(negative, 6.0),
        "positive_atoms_6a": count(positive, 6.0),
        "signed_charge_over_distance": float(signed_charge),
        "hbond_compatible_atoms_3_5a": count(polar_protein, 3.5),
        "nearby_titratable_residues_6a": float(sum(
            value <= 6.0 for value in titratable_minimum.values()
        )),
        "nearby_titratable_residues_10a": float(sum(
            value <= 10.0 for value in titratable_minimum.values()
        )),
        "waters_4a": count(water, 4.0),
        "waters_6a": count(water, 6.0),
        "organic_heterogen_atoms_6a": count(organic, 6.0),
        "metals_4a": count(metals, 4.0),
        "min_protein_distance": _minimum(_distance(center, atom) for atom in protein),
        "min_water_distance": _minimum(_distance(center, atom) for atom in water),
        "min_organic_heterogen_distance": _minimum(
            _distance(center, atom) for atom in organic
        ),
        "min_metal_distance": _minimum(_distance(center, atom) for atom in metals),
    }
    for name in CHEMISTRY_IS_ACIDIC:
        result[f"chemistry_{name}"] = float(name == chemistry)
    return {column: float(result[column]) for column in FEATURE_COLUMNS}


def ligand_analogue(site: Mapping[str, object]) -> Tuple[Optional[str], float, str]:
    """Map a ligand transition to the nearest PKAD-R residue chemistry."""
    family = str(site.get("family", ""))
    label = str(site.get("label", ""))
    if family == "carboxyl":
        return "carboxyl", 0.90, "direct_carboxylate_analogue"
    if family in {"imidazole_basicity", "benzimidazole"} or "imidazol" in label:
        return "imidazole", 0.80, "histidine_ring_analogue"
    if family == "amine":
        factor = 0.65 if label in {
            "primary_amine", "secondary_amine", "tertiary_amine"
        } else 0.50
        return "amine", factor, "lysine_amine_analogue"
    if family == "thiol_thiolate":
        return "thiol", 0.75, "cysteine_thiol_analogue"
    if family == "phenol_phenolate":
        return "phenol", 0.75, "tyrosine_phenol_analogue"
    return None, 0.0, "no_experimental_protein_residue_analogue"


def _load_bundle(path: str | Path) -> Dict[str, object]:
    with Path(path).open("rb") as handle:
        bundle = pickle.load(handle)
    required = {
        "model", "feature_columns", "schema_version", "feature_schema_version",
        "method", "training_rows", "oof_mae", "oof_p90_abs_error",
        "chemistry_p90_abs_error", "training_source_sha256",
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"Protein-environment model missing provenance: {sorted(missing)}")
    if str(bundle["schema_version"]) != PROTEIN_ENVIRONMENT_SCHEMA_VERSION:
        raise ValueError("Protein-environment model schema is incompatible")
    if str(bundle["feature_schema_version"]) != PROTEIN_ENVIRONMENT_FEATURE_SCHEMA_VERSION:
        raise ValueError("Protein-environment feature schema is incompatible")
    if str(bundle["method"]) != PROTEIN_ENVIRONMENT_METHOD:
        raise ValueError("Protein-environment model method is incompatible")
    if list(bundle["feature_columns"]) != list(FEATURE_COLUMNS):
        raise ValueError("Protein-environment feature columns are incompatible")
    return bundle


def _prediction_for_site(
    bundle: Mapping[str, object],
    descriptors: Mapping[str, float],
    *,
    chemistry: str,
    transfer_factor: float,
    direct_protein_site: bool = False,
) -> Dict[str, object]:
    frame = pd.DataFrame([descriptors]).reindex(columns=FEATURE_COLUMNS)
    direct_artifact = bundle.get("direct_bound_ligand_model")
    if direct_artifact is not None and not direct_protein_site:
        if list(direct_artifact.get("feature_columns", [])) != list(FEATURE_COLUMNS):
            raise ValueError("Direct bound-ligand model feature columns are incompatible")
        raw_delta = float(direct_artifact["model"].predict(frame)[0])
        clip_low, clip_high = direct_artifact.get("prediction_clip", (-5.0, 5.0))
        clipped_delta = float(np.clip(raw_delta, float(clip_low), float(clip_high)))
        model_p90 = float(direct_artifact["oof_p90_abs_error"])
        confidence_score = float(min(0.85, math.exp(-model_p90 / 3.0)))
        return {
            "raw_protein_residue_delta_pka": None,
            "clipped_protein_residue_delta_pka": None,
            "raw_direct_bound_ligand_delta_pka": raw_delta,
            "clipped_direct_bound_ligand_delta_pka": clipped_delta,
            "transfer_shrinkage": 1.0,
            "representation_correction_scale": 1.0,
            "representation_calibration_tier": (
                "direct_bound_ligand_grouped_oof_calibration"
            ),
            "prediction_basis": "direct_bound_ligand_calibration",
            "uses_direct_bound_ligand_calibration": True,
            "delta_pka": clipped_delta,
            "delta_pka_p90_error": model_p90,
            "confidence_score": confidence_score,
            "confidence_tier": (
                "medium" if confidence_score >= 0.40
                else "low" if confidence_score >= 0.15
                else "very_low"
            ),
        }
    raw_delta = float(bundle["model"].predict(frame)[0])
    clip_low, clip_high = bundle.get("prediction_clip", (-5.0, 5.0))
    clipped_delta = float(np.clip(raw_delta, float(clip_low), float(clip_high)))
    calibration = dict(bundle.get("calibration", {}))
    chemistry_calibration = dict(
        dict(calibration.get("chemistry", {})).get(chemistry, {})
    )
    representation_scale = float(
        chemistry_calibration.get("correction_scale", 1.0)
    )
    representation_scale = max(0.0, min(1.0, representation_scale))
    evidence_tier = str(
        chemistry_calibration.get(
            "correction_evidence_tier", "legacy_model_without_calibration_gate"
        )
    )
    corrected_delta = float(
        clipped_delta * float(transfer_factor) * representation_scale
    )
    chemistry_p90 = dict(bundle["chemistry_p90_abs_error"])
    model_p90 = float(chemistry_calibration.get(
        "absolute_error_radius_90",
        chemistry_p90.get(chemistry, bundle["oof_p90_abs_error"]),
    ))
    transfer_penalty = (
        0.0
        if direct_protein_site
        else (1.0 - float(transfer_factor)) * max(
            abs(clipped_delta), 1.0
        ) + 0.75
    )
    uncertainty = float(math.sqrt(model_p90 ** 2 + transfer_penalty ** 2))
    confidence_score = float(min(0.85 if direct_protein_site else 0.65, max(
        0.0,
        transfer_factor
        * representation_scale
        * math.exp(-model_p90 / 3.0),
    )))
    confidence_tier = (
        "medium" if confidence_score >= 0.40
        else "low" if confidence_score >= 0.15
        else "very_low"
    )
    return {
        "raw_protein_residue_delta_pka": raw_delta,
        "clipped_protein_residue_delta_pka": clipped_delta,
        "transfer_shrinkage": float(transfer_factor),
        "representation_correction_scale": representation_scale,
        "representation_calibration_tier": evidence_tier,
        "prediction_basis": (
            "protein_residue_model" if direct_protein_site
            else "protein_residue_analogue_transfer"
        ),
        "uses_direct_bound_ligand_calibration": False,
        "delta_pka": corrected_delta,
        "delta_pka_p90_error": uncertainty,
        "confidence_score": confidence_score,
        "confidence_tier": confidence_tier,
    }


def _protein_site_id(atom: StructureAtom) -> str:
    chain = atom.chain or "blank"
    insertion = atom.insertion_code or ""
    return f"protein_{chain}_{atom.residue_name}_{atom.residue_id}{insertion}"


def _screened_coulomb_coupling(
    distance_angstrom: float,
    *,
    dielectric: float,
    debye_length_angstrom: float,
    maximum_absolute_coupling: float,
) -> float:
    """Return log10 coupling for two +1 protonation charge increments."""
    distance = max(float(distance_angstrom), 1.5)
    screening = math.exp(-distance / float(debye_length_angstrom))
    delta_g = 138.935456 * screening / (float(dielectric) * distance)
    log10_coupling = -delta_g / (
        R_KJ_MOL_K * DEFAULT_TEMPERATURE_K * math.log(10.0)
    )
    return float(np.clip(
        log10_coupling,
        -float(maximum_absolute_coupling),
        float(maximum_absolute_coupling),
    ))


def add_joint_protein_nodes(
    *,
    sites: list[Dict[str, object]],
    nodes: list[Dict[str, object]],
    edges: list[Dict[str, object]],
    steps: list[Dict[str, object]],
    free_energy: Dict[str, object],
    structure_atoms: Sequence[StructureAtom],
    ligand_coordinates: Mapping[int, np.ndarray],
    bundle: Mapping[str, object],
    ligand_selector: Optional[LigandSelector],
    radius_angstrom: float = 8.0,
    maximum_protein_sites: int = 6,
    maximum_joint_nodes: int = 4096,
    coupling_dielectric: float = 20.0,
    coupling_debye_length_angstrom: float = 10.0,
    maximum_absolute_coupling: float = 0.75,
    coupling_reference_ph: float = 7.0,
) -> Tuple[
    list[Dict[str, object]],
    list[Dict[str, object]],
    list[Dict[str, object]],
    list[Dict[str, object]],
    Dict[str, object],
    Dict[str, object],
]:
    """Cross a ligand network with nearby binary protein-residue coordinates.

    PKAD-R calibrates protein one-body residuals, but not microscopic pair
    couplings.  Dynamic ligand--protein and protein--protein couplings therefore
    use a separately labelled, centered screened-Coulomb approximation.
    """
    numeric_parameters = (
        radius_angstrom,
        coupling_dielectric,
        coupling_debye_length_angstrom,
        maximum_absolute_coupling,
        coupling_reference_ph,
    )
    if not all(math.isfinite(float(value)) for value in numeric_parameters):
        raise ValueError("joint protein-network parameters must be finite")
    if radius_angstrom <= 0.0:
        raise ValueError("joint protein-site radius must be positive")
    if maximum_protein_sites < 0 or maximum_joint_nodes < 1:
        raise ValueError("joint protein-site and node caps must be non-negative")
    if coupling_dielectric <= 0.0 or coupling_debye_length_angstrom <= 0.0:
        raise ValueError("joint coupling dielectric and Debye length must be positive")
    if maximum_absolute_coupling < 0.0:
        raise ValueError("joint maximum absolute coupling cannot be negative")
    ligand_sites = [
        site for site in sites
        if str(site.get("component_type", "ligand")) == "ligand"
    ]
    ligand_centers = {
        str(site["site_id"]): ligand_site_center(site, ligand_coordinates)
        for site in ligand_sites
    }
    if not ligand_centers:
        raise ValueError("joint protein-site enumeration requires a ligand site")
    residue_atoms: Dict[Tuple[str, str, str, str], list[StructureAtom]] = {}
    for atom in structure_atoms:
        if (
            atom.group == "ATOM"
            and atom.residue_name in PROTEIN_RESIDUE_CHEMISTRY
        ):
            residue_atoms.setdefault(atom.residue_key, []).append(atom)

    candidates = []
    unavailable_center_residues = []
    for residue_key, atoms in residue_atoms.items():
        chain, residue_id, insertion, residue_name = residue_key
        try:
            center, _ = residue_site_center(
                structure_atoms,
                chain=chain,
                residue_id=residue_id,
                residue_name=residue_name,
            )
        except KeyError as exc:
            nearest_resolved_atom = min(
                float(np.linalg.norm(atom.coordinate - ligand_center))
                for atom in atoms
                for ligand_center in ligand_centers.values()
            )
            if nearest_resolved_atom <= float(radius_angstrom) + 5.0:
                unavailable_center_residues.append({
                    "chain": chain,
                    "residue_name": residue_name,
                    "residue_id": residue_id,
                    "insertion_code": insertion,
                    "nearest_resolved_atom_distance_angstrom": nearest_resolved_atom,
                    "reason": str(exc),
                })
            continue
        nearest = min(
            float(np.linalg.norm(center - ligand_center))
            for ligand_center in ligand_centers.values()
        )
        if nearest <= float(radius_angstrom):
            candidates.append((nearest, residue_key, center, atoms[0]))
    candidates.sort(key=lambda item: (item[0], item[1]))

    selected = []
    cap_excluded = []
    for candidate in candidates:
        next_count = len(selected) + 1
        tractable = (
            next_count <= int(maximum_protein_sites)
            and len(nodes) * (2 ** next_count) <= int(maximum_joint_nodes)
        )
        (selected if tractable else cap_excluded).append(candidate)

    if not selected:
        report = {
            "method": JOINT_PROTEIN_NETWORK_METHOD,
            "status": "no_nearby_tractable_protein_sites",
            "candidate_site_count": int(len(candidates)),
            "selected_site_count": 0,
            "cap_excluded_site_count": int(len(cap_excluded)),
            "unavailable_center_site_count": int(len(unavailable_center_residues)),
            "unavailable_center_residues": unavailable_center_residues,
            "original_ligand_node_count": int(len(nodes)),
            "joint_node_count": int(len(nodes)),
        }
        return sites, nodes, edges, steps, free_energy, report

    protein_sites: list[Dict[str, object]] = []
    site_centers = dict(ligand_centers)
    for nearest, residue_key, center, representative in selected:
        chain, residue_id, insertion, residue_name = residue_key
        chemistry = PROTEIN_RESIDUE_CHEMISTRY[residue_name]
        descriptors = environment_descriptors(
            structure_atoms,
            center,
            chemistry=chemistry,
            excluded_residue_keys=[residue_key],
            ligand_coordinates=ligand_coordinates.values(),
            ligand_selector=ligand_selector,
        )
        prediction = _prediction_for_site(
            bundle,
            descriptors,
            chemistry=chemistry,
            transfer_factor=1.0,
            direct_protein_site=True,
        )
        reference_pka = PROTEIN_RESIDUE_REFERENCE_PKA[residue_name]
        predicted_pka = reference_pka + float(prediction["delta_pka"])
        uncertainty = float(prediction["delta_pka_p90_error"])
        site_id = _protein_site_id(representative)
        calibration_enabled = bool(
            float(prediction["representation_correction_scale"]) > 0.0
        )
        site = {
            "site_id": site_id,
            "component_type": "protein_residue",
            "family": "protein_residue",
            "label": f"protein_{residue_name.lower()}",
            "atom_maps": [],
            "center_maps": [],
            "input_member_form": "unknown",
            "coupling_group_id": site_id,
            "acid_level": 1,
            "base_level": 0,
            "protein_chain": chain,
            "protein_residue_id": residue_id,
            "protein_insertion_code": insertion,
            "protein_residue_name": residue_name,
            "protein_site_center_xyz": [float(value) for value in center],
            "protein_nearest_ligand_site_distance_angstrom": float(nearest),
            "stage1_intrinsic_pka": float(reference_pka),
            "local_pka_used": float(predicted_pka),
            "local_pka_confidence": (
                "medium_protein_residue_calibration"
                if calibration_enabled else "low_reference_only"
            ),
            "stage1_predicted_pka_ci_low": predicted_pka - uncertainty,
            "stage1_predicted_pka_ci_high": predicted_pka + uncertainty,
            "stage2_one_body_pka": float(predicted_pka),
            "stage2_predicted_macro_pka": float(predicted_pka),
            "stage2_predicted_macro_pka_ci_low": predicted_pka - uncertainty,
            "stage2_predicted_macro_pka_ci_high": predicted_pka + uncertainty,
            "stage2_applied": calibration_enabled,
            "stage2_prediction_confidence": prediction["confidence_tier"],
            "protein_environment_evaluated": True,
            "protein_environment_applied": calibration_enabled,
            "protein_environment_chemistry": chemistry,
            "protein_environment_descriptors": descriptors,
            **{
                f"protein_environment_{key}": value
                for key, value in prediction.items()
            },
        }
        protein_sites.append(site)
        site_centers[site_id] = center

    original_nodes = [deepcopy(node) for node in nodes]
    original_edges = [deepcopy(edge) for edge in edges]
    protein_count = len(protein_sites)
    joint_nodes: list[Dict[str, object]] = []
    joint_node_lookup: Dict[str, Dict[str, object]] = {}
    node_id_map: Dict[Tuple[str, int], str] = {}
    for configuration in range(2 ** protein_count):
        for original in original_nodes:
            node = deepcopy(original)
            original_id = str(original["node_id"])
            joint_id = f"{original_id}__protein_{configuration:0{protein_count}b}"
            node_id_map[(original_id, configuration)] = joint_id
            node["node_id"] = joint_id
            node["ligand_origin_node_id"] = original_id
            forms = dict(node.get("site_forms") or {})
            levels = dict(node.get("group_levels") or {})
            if not levels:
                # Legacy/minimal networks may encode configurations only in
                # site_forms.  Once protein levels are added, the generic
                # configuration signature would otherwise forget the ligand
                # coordinate and collapse acid/base nodes together.
                levels.update({
                    str(site["site_id"]): int(
                        forms.get(str(site["site_id"])) == "acid_form"
                    )
                    for site in ligand_sites
                })
            added_protons = 0
            for index, site in enumerate(protein_sites):
                occupied = int((configuration >> index) & 1)
                site_id = str(site["site_id"])
                forms[site_id] = "acid_form" if occupied else "base_form"
                levels[site_id] = occupied
                added_protons += occupied
            node["site_forms"] = forms
            node["group_levels"] = levels
            node["ligand_protonated_site_count"] = int(
                original.get("protonated_site_count", 0)
            )
            node["protein_protonated_site_count"] = int(added_protons)
            node["protonated_site_count"] = int(
                original.get("protonated_site_count", 0)
            ) + added_protons
            node["joint_protein_configuration_index"] = int(configuration)
            joint_nodes.append(node)
            joint_node_lookup[joint_id] = node

    joint_edges: list[Dict[str, object]] = []
    for configuration in range(2 ** protein_count):
        for original in original_edges:
            edge = deepcopy(original)
            edge["edge_id"] = f"{original['edge_id']}__protein_{configuration}"
            for key in ("acid_node_id", "base_node_id", "from_node_id", "to_node_id"):
                if key in edge:
                    edge[key] = node_id_map[(str(original[key]), configuration)]
            other = dict(edge.get("other_site_forms") or {})
            for index, site in enumerate(protein_sites):
                other[str(site["site_id"])] = int((configuration >> index) & 1)
            edge["other_site_forms"] = other
            joint_edges.append(edge)

    for protein_index, protein_site in enumerate(protein_sites):
        site_id = str(protein_site["site_id"])
        for configuration in range(2 ** protein_count):
            if (configuration >> protein_index) & 1:
                continue
            acid_configuration = configuration | (1 << protein_index)
            for original in original_nodes:
                original_id = str(original["node_id"])
                base_id = node_id_map[(original_id, configuration)]
                acid_id = node_id_map[(original_id, acid_configuration)]
                base_node = joint_node_lookup[base_id]
                other = dict(base_node.get("group_levels") or {})
                other.pop(site_id, None)
                joint_edges.append({
                    "edge_id": f"edge_{site_id}_{original_id}_{configuration}",
                    "site_id": site_id,
                    "acid_node_id": acid_id,
                    "base_node_id": base_id,
                    "from_node_id": base_id,
                    "to_node_id": acid_id,
                    "other_site_forms": other,
                    "edge_component_type": "protein_residue_protonation",
                })

    all_sites = [*sites, *protein_sites]
    one_body_terms = {
        str(term["site_id"]): dict(term)
        for term in free_energy.get("one_body_terms", [])
    }
    for site in protein_sites:
        site_id = str(site["site_id"])
        one_body_terms[site_id] = {
            "site_id": site_id,
            "one_body_pka": float(site["stage2_one_body_pka"]),
            "source": "PKAD-R-calibrated protein-residue one-body residual",
        }
    pair_terms = [dict(term) for term in free_energy.get("pair_couplings", [])]
    dynamic_pairs = []
    all_site_ids = [str(site["site_id"]) for site in all_sites]
    protein_ids = {str(site["site_id"]) for site in protein_sites}
    reference_occupancy = {
        site_id: 1.0 / (1.0 + 10.0 ** (
            float(coupling_reference_ph)
            - float(one_body_terms[site_id]["one_body_pka"])
        ))
        for site_id in all_site_ids
    }
    for left_index, left in enumerate(all_site_ids):
        for right in all_site_ids[left_index + 1:]:
            if left not in protein_ids and right not in protein_ids:
                continue
            distance = float(np.linalg.norm(site_centers[left] - site_centers[right]))
            if distance > float(radius_angstrom):
                continue
            coupling = _screened_coulomb_coupling(
                distance,
                dielectric=coupling_dielectric,
                debye_length_angstrom=coupling_debye_length_angstrom,
                maximum_absolute_coupling=maximum_absolute_coupling,
            )
            one_body_terms[left]["one_body_pka"] = float(
                one_body_terms[left]["one_body_pka"]
                - coupling * reference_occupancy[right]
            )
            one_body_terms[right]["one_body_pka"] = float(
                one_body_terms[right]["one_body_pka"]
                - coupling * reference_occupancy[left]
            )
            term = {
                "site_i": left,
                "site_j": right,
                "coupling_log10_units": coupling,
                "interaction_delta_g_kj_mol_at_298k": float(
                    -R_KJ_MOL_K * DEFAULT_TEMPERATURE_K * math.log(10.0) * coupling
                ),
                "source": JOINT_PROTEIN_NETWORK_METHOD,
                "experimentally_identified": False,
                "distance_angstrom": distance,
                "dielectric": float(coupling_dielectric),
                "debye_length_angstrom": float(coupling_debye_length_angstrom),
                "centered_reference_ph": float(coupling_reference_ph),
            }
            pair_terms.append(term)
            dynamic_pairs.append(term)

    one_body_map = {
        site_id: float(term["one_body_pka"])
        for site_id, term in one_body_terms.items()
    }
    pair_map = pair_coupling_map(pair_terms)
    edge_annotations, max_cycle_error = contextual_edge_pkas(
        joint_nodes, joint_edges, all_sites, one_body_map, pair_map
    )
    for edge in joint_edges:
        annotation = edge_annotations[str(edge["edge_id"])]
        edge["stage2_contextual_edge_pka"] = float(
            annotation["contextual_edge_pka"]
        )
        edge["stage2_one_body_pka"] = float(annotation["one_body_pka"])
        edge["stage2_pair_interaction_shift"] = float(
            annotation["pair_interaction_shift"]
        )

    macro_values = macro_pka_values_from_free_energy(
        joint_nodes, all_sites, one_body_map, pair_map
    )
    ordered_ids = sorted(all_site_ids, key=lambda value: (-one_body_map[value], value))
    previous_steps = {str(step["rank_associated_site_id"]): step for step in steps}
    joint_steps = []
    site_lookup = {str(site["site_id"]): site for site in all_sites}
    for rank, (site_id, macro_pka) in enumerate(zip(ordered_ids, macro_values), start=1):
        step = dict(previous_steps.get(site_id, {}))
        step.update({
            "protonation_step": rank,
            "rank_associated_site_id": site_id,
            "stage2_predicted_macro_pka": float(macro_pka),
            "joint_protein_network": True,
        })
        joint_steps.append(step)
        site = site_lookup[site_id]
        prior_macro = float(site.get("stage2_predicted_macro_pka", macro_pka))
        low = float(site.get("stage2_predicted_macro_pka_ci_low", prior_macro - 3.0))
        high = float(site.get("stage2_predicted_macro_pka_ci_high", prior_macro + 3.0))
        half_width = max(prior_macro - low, high - prior_macro, 0.5)
        site["pre_joint_stage2_predicted_macro_pka"] = prior_macro
        site["stage2_predicted_macro_pka"] = float(macro_pka)
        site["stage2_predicted_macro_pka_ci_low"] = float(macro_pka - half_width)
        site["stage2_predicted_macro_pka_ci_high"] = float(macro_pka + half_width)
        site["stage2_one_body_pka"] = one_body_map[site_id]

    diagnostics = dict(free_energy.get("diagnostics", {}))
    diagnostics.update({
        "joint_protein_network_applied": True,
        "joint_protein_site_count": int(protein_count),
        "joint_dynamic_pair_count": int(len(dynamic_pairs)),
        "joint_max_cycle_closure_error": float(max_cycle_error),
        "pair_parameter_count": int(len(pair_terms)),
        "active_pair_parameter_count": int(sum(
            abs(float(term.get("coupling_log10_units", 0.0))) > 1.0e-12
            for term in pair_terms
        )),
        "pair_coupling_identifiability": (
            "heuristic_centered_screened_coulomb_not_experimentally_identified"
        ),
        "macro_reconstruction_mae": 0.0,
        "macro_reconstruction_max_abs_error": 0.0,
    })
    joint_report = {
        "method": JOINT_PROTEIN_NETWORK_METHOD,
        "status": "joint_network_enumerated",
        "candidate_site_count": int(len(candidates)),
        "selected_site_count": int(protein_count),
        "cap_excluded_site_count": int(len(cap_excluded)),
        "unavailable_center_site_count": int(len(unavailable_center_residues)),
        "unavailable_center_residues": unavailable_center_residues,
        "selected_sites": [
            {
                "site_id": site["site_id"],
                "chain": site["protein_chain"],
                "residue_name": site["protein_residue_name"],
                "residue_id": site["protein_residue_id"],
                "nearest_ligand_site_distance_angstrom": site[
                    "protein_nearest_ligand_site_distance_angstrom"
                ],
                "one_body_pka": one_body_map[str(site["site_id"])],
            }
            for site in protein_sites
        ],
        "original_ligand_node_count": int(len(original_nodes)),
        "joint_node_count": int(len(joint_nodes)),
        "joint_edge_count": int(len(joint_edges)),
        "dynamic_pair_count": int(len(dynamic_pairs)),
        "maximum_joint_nodes": int(maximum_joint_nodes),
        "radius_angstrom": float(radius_angstrom),
        "coupling_dielectric": float(coupling_dielectric),
        "coupling_debye_length_angstrom": float(coupling_debye_length_angstrom),
        "coupling_reference_ph": float(coupling_reference_ph),
        "coupling_warning": (
            "Pair couplings are centered screened-Coulomb approximations and are "
            "not identified by PKAD-R experimental labels."
        ),
    }
    updated_free_energy = dict(free_energy)
    updated_free_energy["one_body_terms"] = list(one_body_terms.values())
    updated_free_energy["pair_couplings"] = pair_terms
    updated_free_energy["diagnostics"] = diagnostics
    updated_free_energy["joint_protein_network"] = joint_report
    return (
        all_sites,
        joint_nodes,
        joint_edges,
        joint_steps,
        updated_free_energy,
        joint_report,
    )


def apply_protein_environment(
    stage2_table: pd.DataFrame,
    *,
    query_molecule: Chem.Mol,
    protein_structure_path: str | Path,
    ligand_structure_path: str | Path,
    model_path: str | Path = DEFAULT_ENVIRONMENT_MODEL,
    ligand_selector: Optional[LigandSelector] = None,
    joint_protein_sites: bool = False,
    joint_protein_radius: float = 8.0,
    maximum_joint_protein_sites: int = 6,
    maximum_joint_nodes: int = 4096,
    joint_coupling_dielectric: float = 20.0,
    joint_coupling_debye_length: float = 10.0,
    joint_maximum_absolute_coupling: float = 0.75,
    joint_coupling_reference_ph: float = 7.0,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Apply environmental residuals and optionally enumerate protein sites.

    The low-level function leaves explicit residue enumeration off by default
    for backward compatibility.  The public ``pkate.predict`` complex path
    opts in by default and supplies tractability caps.
    """
    if len(stage2_table) != 1:
        raise ValueError("Protein-environment inference currently accepts one query molecule")
    protein_path = Path(protein_structure_path)
    ligand_path = Path(ligand_structure_path)
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(
            "Protein-environment model is missing. Train it with "
            "`python scripts/train_protein_environment_model.py` or pass "
            "protein_environment_model."
        )
    atoms = read_structure_atoms(protein_path)
    coordinates, ligand_mapping = map_ligand_coordinates(query_molecule, ligand_path)
    bundle = _load_bundle(model_file)

    # Reject a ligand pose in a different coordinate frame rather than silently
    # generating an apparent solvent-exposed correction.
    heavy_environment = [atom for atom in atoms if atom.element != "H"]
    nearest_pose_contact = min(
        float(np.linalg.norm(atom.coordinate - coordinate))
        for atom in heavy_environment
        for coordinate in coordinates.values()
    )
    if nearest_pose_contact > 8.0:
        raise ValueError(
            "Ligand and protein coordinates appear to use different frames: "
            f"nearest heavy-atom distance is {nearest_pose_contact:.2f} A"
        )

    row = stage2_table.iloc[0].copy()
    sites = json.loads(str(row["sites_json"]))
    nodes = json.loads(str(row["microstate_nodes_json"]))
    edges = json.loads(str(row["microstate_edges_json"]))
    steps = sorted(
        json.loads(str(row["stage2_macro_pka_steps_json"])),
        key=lambda item: int(item["protonation_step"]),
    )
    free_energy = json.loads(str(row["stage2_free_energy_model_json"]))
    limited = _truthy(row.get("protonation_state_enumeration_truncated", False)) or bool(
        json.loads(str(row.get("incomplete_site_ids_json", "[]")))
    )
    one_body_terms = {
        str(term["site_id"]): term
        for term in free_energy.get("one_body_terms", [])
    }
    if set(one_body_terms) != {str(site["site_id"]) for site in sites}:
        raise ValueError("Stage 2 free-energy model does not cover every ligand site")

    applied_count = 0
    unsupported_count = 0
    calibration_withheld_count = 0
    site_reports = []
    for site in sites:
        site_id = str(site["site_id"])
        chemistry, transfer_factor, transfer_basis = ligand_analogue(site)
        aqueous_one_body = float(one_body_terms[site_id]["one_body_pka"])
        site["aqueous_stage2_one_body_pka"] = aqueous_one_body
        site["protein_environment_transfer_basis"] = transfer_basis
        if chemistry is None:
            unsupported_count += 1
            site.update({
                "protein_environment_applied": False,
                "protein_environment_chemistry": None,
                "protein_environment_delta_pka": 0.0,
                "protein_environment_delta_pka_p90_error": None,
                "protein_environment_confidence_score": 0.0,
                "protein_environment_confidence_tier": "very_low",
                "protein_environment_descriptors": {},
            })
            site_reports.append({
                "site_id": site_id,
                "applied": False,
                "reason": transfer_basis,
            })
            continue
        center = ligand_site_center(site, coordinates)
        descriptors = environment_descriptors(
            atoms,
            center,
            chemistry=chemistry,
            ligand_coordinates=coordinates.values(),
            ligand_selector=ligand_selector,
        )
        prediction = _prediction_for_site(
            bundle,
            descriptors,
            chemistry=chemistry,
            transfer_factor=transfer_factor,
        )
        correction_enabled = bool(
            float(prediction["representation_correction_scale"]) > 0.0
        )
        delta = float(prediction["delta_pka"])
        corrected_one_body = aqueous_one_body + delta
        one_body_terms[site_id]["aqueous_one_body_pka"] = aqueous_one_body
        one_body_terms[site_id]["protein_environment_delta_pka"] = delta
        one_body_terms[site_id]["one_body_pka"] = corrected_one_body
        site.update({
            "protein_environment_evaluated": True,
            "protein_environment_applied": correction_enabled,
            "protein_environment_chemistry": chemistry,
            "protein_environment_transfer_basis": transfer_basis,
            "protein_environment_model_method": PROTEIN_ENVIRONMENT_METHOD,
            "protein_environment_descriptors": descriptors,
            "protein_environment_corrected_one_body_pka": corrected_one_body,
            **{
                f"protein_environment_{key}": value
                for key, value in prediction.items()
            },
        })
        site["stage2_one_body_pka"] = corrected_one_body
        site["stage2_one_body_shift_from_stage1"] = (
            corrected_one_body - float(site["stage1_intrinsic_pka"])
        )
        applied_count += int(correction_enabled)
        calibration_withheld_count += int(not correction_enabled)
        site_reports.append({
            "site_id": site_id,
            "evaluated": True,
            "applied": correction_enabled,
            "chemistry": chemistry,
            "transfer_basis": transfer_basis,
            "center": [float(value) for value in center],
            **prediction,
        })

    one_body_map = {
        site_id: float(term["one_body_pka"])
        for site_id, term in one_body_terms.items()
    }
    pair_map = pair_coupling_map(free_energy.get("pair_couplings", []))
    macro_values = (
        [] if limited
        else macro_pka_values_from_free_energy(nodes, sites, one_body_map, pair_map)
    )
    corrected_order = sorted(
        (str(site["site_id"]) for site in sites),
        key=lambda site_id: (-one_body_map[site_id], site_id),
    )
    original_step_by_site = {
        str(step["rank_associated_site_id"]): step for step in steps
    }
    site_lookup = {str(site["site_id"]): site for site in sites}
    corrected_steps = []
    for index, (site_id, macro_pka) in enumerate(
        zip(corrected_order, macro_values), start=1
    ):
        original = dict(original_step_by_site[site_id])
        original["aqueous_stage2_predicted_macro_pka"] = original.get(
            "stage2_predicted_macro_pka"
        )
        original["protonation_step"] = index
        original["rank_associated_site_id"] = site_id
        original["stage2_predicted_macro_pka"] = float(macro_pka)
        original["protein_environment_corrected"] = bool(
            site_lookup[site_id].get("protein_environment_applied", False)
        )
        corrected_steps.append(original)

    for step in corrected_steps:
        site = site_lookup[str(step["rank_associated_site_id"])]
        center_value = float(step["stage2_predicted_macro_pka"])
        old_center = site.get("stage2_predicted_macro_pka")
        old_center = center_value if old_center is None else float(old_center)
        old_low = float(site.get("stage2_predicted_macro_pka_ci_low", old_center - 3.0))
        old_high = float(site.get("stage2_predicted_macro_pka_ci_high", old_center + 3.0))
        aqueous_half_width = max(old_center - old_low, old_high - old_center, 0.5)
        environment_error = (
            site.get("protein_environment_delta_pka_p90_error")
            if site.get("protein_environment_applied", False)
            else None
        )
        combined_half_width = (
            aqueous_half_width
            if environment_error is None
            else math.sqrt(aqueous_half_width ** 2 + float(environment_error) ** 2)
        )
        site["aqueous_stage2_predicted_macro_pka"] = old_center
        site["stage2_predicted_macro_pka"] = center_value
        site["stage2_predicted_macro_pka_ci_low"] = center_value - combined_half_width
        site["stage2_predicted_macro_pka_ci_high"] = center_value + combined_half_width

    for edge in edges:
        site_id = str(edge["site_id"])
        site = site_lookup[site_id]
        delta = float(site.get("protein_environment_delta_pka", 0.0))
        contextual = edge.get("stage2_contextual_edge_pka")
        edge["aqueous_stage2_contextual_edge_pka"] = contextual
        if contextual is not None:
            edge["stage2_contextual_edge_pka"] = float(contextual) + delta
        edge["stage2_one_body_pka"] = one_body_map[site_id]
        edge["protein_environment_delta_pka"] = delta

    free_energy["one_body_terms"] = list(one_body_terms.values())
    free_energy["protein_environment_correction"] = {
        "schema_version": PROTEIN_ENVIRONMENT_SCHEMA_VERSION,
        "feature_schema_version": PROTEIN_ENVIRONMENT_FEATURE_SCHEMA_VERSION,
        "method": PROTEIN_ENVIRONMENT_METHOD,
        "model_sha256": _sha256(model_file),
        "protein_structure_sha256": _sha256(protein_path),
        "ligand_structure_sha256": _sha256(ligand_path),
        "applied_site_count": int(applied_count),
        "unsupported_site_count": int(unsupported_count),
        "calibration_withheld_site_count": int(calibration_withheld_count),
        "calibration_schema_version": dict(bundle.get("calibration", {})).get(
            "schema_version"
        ),
    }
    diagnostics = free_energy.setdefault("diagnostics", {})
    if not limited:
        diagnostics["macro_reconstruction_mae"] = 0.0
        diagnostics["macro_reconstruction_max_abs_error"] = 0.0

    joint_report: Dict[str, object] = {
        "method": JOINT_PROTEIN_NETWORK_METHOD,
        "status": "not_requested",
        "selected_site_count": 0,
        "original_ligand_node_count": int(len(nodes)),
        "joint_node_count": int(len(nodes)),
    }
    if joint_protein_sites and limited:
        joint_report["status"] = "not_applied_incomplete_ligand_network"
    elif joint_protein_sites:
        (
            sites,
            nodes,
            edges,
            corrected_steps,
            free_energy,
            joint_report,
        ) = add_joint_protein_nodes(
            sites=sites,
            nodes=nodes,
            edges=edges,
            steps=corrected_steps,
            free_energy=free_energy,
            structure_atoms=atoms,
            ligand_coordinates=coordinates,
            bundle=bundle,
            ligand_selector=ligand_selector,
            radius_angstrom=float(joint_protein_radius),
            maximum_protein_sites=int(maximum_joint_protein_sites),
            maximum_joint_nodes=int(maximum_joint_nodes),
            coupling_dielectric=float(joint_coupling_dielectric),
            coupling_debye_length_angstrom=float(
                joint_coupling_debye_length
            ),
            maximum_absolute_coupling=float(
                joint_maximum_absolute_coupling
            ),
            coupling_reference_ph=float(joint_coupling_reference_ph),
        )
    joint_applied = joint_report.get("status") == "joint_network_enumerated"
    final_diagnostics = dict(free_energy.get("diagnostics", {}))

    output = row.to_dict()
    output.update({
        "sites_json": json.dumps(sites, sort_keys=True, separators=(",", ":")),
        "microstate_nodes_json": json.dumps(
            nodes, sort_keys=True, separators=(",", ":")
        ),
        "microstate_edges_json": json.dumps(edges, sort_keys=True, separators=(",", ":")),
        "stage2_macro_pka_steps_json": json.dumps(
            corrected_steps, sort_keys=True, separators=(",", ":")
        ),
        "stage2_free_energy_model_json": json.dumps(
            free_energy, sort_keys=True, separators=(",", ":")
        ),
        "protein_environment_applied": bool(applied_count),
        "protein_environment_applied_site_count": int(applied_count),
        "protein_environment_unsupported_site_count": int(unsupported_count),
        "protein_environment_calibration_withheld_site_count": int(
            calibration_withheld_count
        ),
        "protein_environment_schema_version": PROTEIN_ENVIRONMENT_SCHEMA_VERSION,
        "protein_environment_feature_schema_version": (
            PROTEIN_ENVIRONMENT_FEATURE_SCHEMA_VERSION
        ),
        "protein_environment_method": PROTEIN_ENVIRONMENT_METHOD,
        "protein_environment_calibration_schema_version": dict(
            bundle.get("calibration", {})
        ).get("schema_version"),
        "protein_environment_model_sha256": _sha256(model_file),
        "protein_environment_protein_structure_sha256": _sha256(protein_path),
        "protein_environment_ligand_structure_sha256": _sha256(ligand_path),
        "joint_protein_network_applied": bool(joint_applied),
        "joint_protein_site_count": int(joint_report.get("selected_site_count", 0)),
        "joint_protein_candidate_site_count": int(
            joint_report.get("candidate_site_count", 0)
        ),
        "joint_protein_unavailable_center_site_count": int(
            joint_report.get("unavailable_center_site_count", 0)
        ),
        "joint_microstate_node_count": int(
            joint_report.get("joint_node_count", len(nodes))
        ),
        "joint_protein_network_method": JOINT_PROTEIN_NETWORK_METHOD,
        "stage2_pair_parameter_count": int(
            final_diagnostics.get("pair_parameter_count", 0)
        ),
        "stage2_active_pair_parameter_count": int(
            final_diagnostics.get("active_pair_parameter_count", 0)
        ),
        "stage2_pair_coupling_identifiability": final_diagnostics.get(
            "pair_coupling_identifiability"
        ),
        "stage2_free_energy_macro_reconstruction_mae": final_diagnostics.get(
            "macro_reconstruction_mae"
        ),
        "stage2_max_cycle_closure_error": final_diagnostics.get(
            "joint_max_cycle_closure_error",
            row.get("stage2_max_cycle_closure_error"),
        ),
    })
    report: Dict[str, object] = {
        "schema_version": PROTEIN_ENVIRONMENT_SCHEMA_VERSION,
        "method": PROTEIN_ENVIRONMENT_METHOD,
        "model_path": str(model_file.resolve()),
        "model_sha256": _sha256(model_file),
        "training_rows": int(bundle["training_rows"]),
        "training_source_sha256": str(bundle["training_source_sha256"]),
        "oof_mae": float(bundle["oof_mae"]),
        "oof_p90_abs_error": float(bundle["oof_p90_abs_error"]),
        "calibration": bundle.get("calibration"),
        "protein_structure_path": str(protein_path.resolve()),
        "protein_structure_sha256": _sha256(protein_path),
        "ligand_structure_path": str(ligand_path.resolve()),
        "nearest_pose_contact_angstrom": nearest_pose_contact,
        "ligand_mapping": ligand_mapping,
        "ligand_selector": asdict(ligand_selector) if ligand_selector else None,
        "applied_site_count": int(applied_count),
        "unsupported_site_count": int(unsupported_count),
        "calibration_withheld_site_count": int(calibration_withheld_count),
        "sites": site_reports,
        "joint_protein_network": joint_report,
        "scope_warning": (
            "Calibrated on protein-residue pKa shifts; ligand corrections use "
            "analogue transfer shrinkage and are not direct bound-ligand supervision."
        ),
    }
    return pd.DataFrame([output]), report


def apply_protein_environment_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    query_molecule: Chem.Mol,
    protein_structure_path: str | Path,
    ligand_structure_path: str | Path,
    model_path: str | Path = DEFAULT_ENVIRONMENT_MODEL,
    ligand_selector: Optional[LigandSelector] = None,
    joint_protein_sites: bool = False,
    joint_protein_radius: float = 8.0,
    maximum_joint_protein_sites: int = 6,
    maximum_joint_nodes: int = 4096,
    joint_coupling_dielectric: float = 20.0,
    joint_coupling_debye_length: float = 10.0,
    joint_maximum_absolute_coupling: float = 0.75,
    joint_coupling_reference_ph: float = 7.0,
    report_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """File wrapper used between canonical Stage 2 and Stage 3."""
    table = pd.read_csv(input_path, low_memory=False)
    corrected, report = apply_protein_environment(
        table,
        query_molecule=query_molecule,
        protein_structure_path=protein_structure_path,
        ligand_structure_path=ligand_structure_path,
        model_path=model_path,
        ligand_selector=ligand_selector,
        joint_protein_sites=joint_protein_sites,
        joint_protein_radius=joint_protein_radius,
        maximum_joint_protein_sites=maximum_joint_protein_sites,
        maximum_joint_nodes=maximum_joint_nodes,
        joint_coupling_dielectric=joint_coupling_dielectric,
        joint_coupling_debye_length=joint_coupling_debye_length,
        joint_maximum_absolute_coupling=joint_maximum_absolute_coupling,
        joint_coupling_reference_ph=joint_coupling_reference_ph,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(destination, index=False)
    report["output_sha256"] = _sha256(destination)
    if report_path is not None:
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with report_file.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    return corrected, report
