from typing import Dict, Iterable, Optional, Set

import numpy as np
from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem, rdMolDescriptors  # type: ignore


EMBED_SEED = 17

_MOL_GEOM_KEYS = (
    "embed_success",
    "radius_of_gyration",
    "asphericity",
    "eccentricity",
    "inertial_shape_factor",
    "npr1",
    "npr2",
    "spherocity_index",
)

_SITE_GEOM_KEYS = (
    "embed_success",
    "source_centroid_to_mol_centroid",
    "local_centroid_to_mol_centroid",
    "source_radius_of_gyration",
    "local_radius_of_gyration",
    "source_span",
    "local_span",
    "mean_source_to_local",
    "min_source_to_local",
    "max_source_to_local",
)


def zero_molecule_geometry_features(prefix: str = "mol_geom") -> Dict[str, float]:
    return {f"{prefix}_{key}": 0.0 for key in _MOL_GEOM_KEYS}


def zero_site_geometry_features(prefix: str = "site_geom") -> Dict[str, float]:
    return {f"{prefix}_{key}": 0.0 for key in _SITE_GEOM_KEYS}


def embed_molecule_3d(mol: Optional[Chem.Mol]) -> Optional[Chem.Mol]:
    if mol is None:
        return None

    mol_3d = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = EMBED_SEED
    params.useRandomCoords = False

    try:
        status = AllChem.EmbedMolecule(mol_3d, params)
    except Exception:
        return None
    if status != 0:
        return None

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_3d):
            AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(mol_3d, maxIters=200)
    except Exception:
        pass

    return mol_3d


def molecule_geometry_features(
    mol_3d: Optional[Chem.Mol],
    prefix: str = "mol_geom",
) -> Dict[str, float]:
    features = zero_molecule_geometry_features(prefix=prefix)
    if mol_3d is None or mol_3d.GetNumConformers() == 0:
        return features

    try:
        features.update(
            {
                f"{prefix}_embed_success": 1.0,
                f"{prefix}_radius_of_gyration": float(rdMolDescriptors.CalcRadiusOfGyration(mol_3d)),
                f"{prefix}_asphericity": float(rdMolDescriptors.CalcAsphericity(mol_3d)),
                f"{prefix}_eccentricity": float(rdMolDescriptors.CalcEccentricity(mol_3d)),
                f"{prefix}_inertial_shape_factor": float(rdMolDescriptors.CalcInertialShapeFactor(mol_3d)),
                f"{prefix}_npr1": float(rdMolDescriptors.CalcNPR1(mol_3d)),
                f"{prefix}_npr2": float(rdMolDescriptors.CalcNPR2(mol_3d)),
                f"{prefix}_spherocity_index": float(rdMolDescriptors.CalcSpherocityIndex(mol_3d)),
            }
        )
    except Exception:
        return features

    return features


def _coords_for_atoms(mol_3d: Chem.Mol, atom_indices: Iterable[int]) -> np.ndarray:
    conformer = mol_3d.GetConformer()
    coords = [conformer.GetAtomPosition(int(atom_idx)) for atom_idx in atom_indices]
    if not coords:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray([[pos.x, pos.y, pos.z] for pos in coords], dtype=np.float32)


def _radius_of_gyration(coords: np.ndarray) -> float:
    if coords.size == 0:
        return 0.0
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def _span(coords: np.ndarray) -> float:
    if coords.shape[0] <= 1:
        return 0.0
    deltas = coords[:, None, :] - coords[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    return float(np.max(distances))


def site_geometry_features(
    mol_3d: Optional[Chem.Mol],
    source_atoms: Set[int],
    local_atoms: Set[int],
    prefix: str = "site_geom",
) -> Dict[str, float]:
    features = zero_site_geometry_features(prefix=prefix)
    if mol_3d is None or mol_3d.GetNumConformers() == 0 or not source_atoms:
        return features

    heavy_atom_indices = [atom.GetIdx() for atom in mol_3d.GetAtoms() if atom.GetAtomicNum() > 1]
    source_coords = _coords_for_atoms(mol_3d, sorted(source_atoms))
    local_coords = _coords_for_atoms(mol_3d, sorted(local_atoms)) if local_atoms else source_coords
    mol_coords = _coords_for_atoms(mol_3d, heavy_atom_indices)
    if source_coords.size == 0 or local_coords.size == 0 or mol_coords.size == 0:
        return features

    mol_centroid = mol_coords.mean(axis=0)
    source_centroid = source_coords.mean(axis=0)
    local_centroid = local_coords.mean(axis=0)
    source_to_local = np.sqrt(np.sum((source_coords[:, None, :] - local_coords[None, :, :]) ** 2, axis=2))

    features.update(
        {
            f"{prefix}_embed_success": 1.0,
            f"{prefix}_source_centroid_to_mol_centroid": float(np.linalg.norm(source_centroid - mol_centroid)),
            f"{prefix}_local_centroid_to_mol_centroid": float(np.linalg.norm(local_centroid - mol_centroid)),
            f"{prefix}_source_radius_of_gyration": _radius_of_gyration(source_coords),
            f"{prefix}_local_radius_of_gyration": _radius_of_gyration(local_coords),
            f"{prefix}_source_span": _span(source_coords),
            f"{prefix}_local_span": _span(local_coords),
            f"{prefix}_mean_source_to_local": float(np.mean(source_to_local)),
            f"{prefix}_min_source_to_local": float(np.min(source_to_local)),
            f"{prefix}_max_source_to_local": float(np.max(source_to_local)),
        }
    )
    return features