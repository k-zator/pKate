"""Unit tests for deterministic molecular and site-local 3D features."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from scripts import geometry_features as geometry


def _linear_molecule(coordinates: list[tuple[float, float, float]]) -> Chem.Mol:
    """Return a three-carbon molecule with a controlled conformer."""
    mol = Chem.MolFromSmiles("CCC")
    assert mol is not None
    assert len(coordinates) == mol.GetNumAtoms()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom_idx, coordinate in enumerate(coordinates):
        conformer.SetAtomPosition(atom_idx, coordinate)
    mol.AddConformer(conformer)
    return mol


def test_zero_feature_factories_have_stable_prefixed_schema() -> None:
    molecule = geometry.zero_molecule_geometry_features(prefix="m")
    site = geometry.zero_site_geometry_features(prefix="s")

    assert set(molecule) == {f"m_{key}" for key in geometry._MOL_GEOM_KEYS}
    assert set(site) == {f"s_{key}" for key in geometry._SITE_GEOM_KEYS}
    assert not any(molecule.values())
    assert not any(site.values())


def test_embedding_none_and_failed_embedding_return_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert geometry.embed_molecule_3d(None) is None
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    monkeypatch.setattr(geometry.AllChem, "EmbedMolecule", lambda *_args, **_kwargs: -1)
    assert geometry.embed_molecule_3d(mol) is None


def test_embedding_exception_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    def fail_embedding(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("forced embedding failure")

    monkeypatch.setattr(geometry.AllChem, "EmbedMolecule", fail_embedding)
    assert geometry.embed_molecule_3d(mol) is None


def test_embedding_is_reproducible_and_does_not_mutate_input() -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    atom_count = mol.GetNumAtoms()

    first = geometry.embed_molecule_3d(mol)
    second = geometry.embed_molecule_3d(mol)

    assert first is not None and second is not None
    assert first.GetNumConformers() == second.GetNumConformers() == 1
    assert first.GetNumAtoms() > atom_count
    assert mol.GetNumAtoms() == atom_count
    assert mol.GetNumConformers() == 0
    np.testing.assert_allclose(
        Chem.Get3DDistanceMatrix(first),
        Chem.Get3DDistanceMatrix(second),
        rtol=1e-6,
        atol=1e-6,
    )


def test_embedding_uses_uff_fallback_and_ignores_optimization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = Chem.MolFromSmiles("CC")
    assert mol is not None
    calls: list[str] = []

    def fake_embed(target: Chem.Mol, _params: object) -> int:
        target.AddConformer(Chem.Conformer(target.GetNumAtoms()))
        return 0

    def fail_uff(*_args: object, **_kwargs: object) -> int:
        calls.append("uff")
        raise RuntimeError("forced optimizer failure")

    monkeypatch.setattr(geometry.AllChem, "EmbedMolecule", fake_embed)
    monkeypatch.setattr(geometry.AllChem, "MMFFHasAllMoleculeParams", lambda _mol: False)
    monkeypatch.setattr(geometry.AllChem, "UFFOptimizeMolecule", fail_uff)

    embedded = geometry.embed_molecule_3d(mol)
    assert embedded is not None
    assert embedded.GetNumConformers() == 1
    assert calls == ["uff"]


def test_embedding_uses_mmff_when_parameters_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = Chem.MolFromSmiles("CC")
    assert mol is not None
    calls: list[tuple[str, int]] = []

    def fake_embed(target: Chem.Mol, _params: object) -> int:
        target.AddConformer(Chem.Conformer(target.GetNumAtoms()))
        return 0

    def fake_mmff(_mol: Chem.Mol, *, maxIters: int) -> int:
        calls.append(("mmff", maxIters))
        return 0

    monkeypatch.setattr(geometry.AllChem, "EmbedMolecule", fake_embed)
    monkeypatch.setattr(geometry.AllChem, "MMFFHasAllMoleculeParams", lambda _mol: True)
    monkeypatch.setattr(geometry.AllChem, "MMFFOptimizeMolecule", fake_mmff)

    assert geometry.embed_molecule_3d(mol) is not None
    assert calls == [("mmff", 200)]


def test_molecule_features_fall_back_without_a_conformer() -> None:
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None

    assert geometry.molecule_geometry_features(None) == geometry.zero_molecule_geometry_features()
    assert geometry.molecule_geometry_features(mol) == geometry.zero_molecule_geometry_features()


def test_molecule_features_are_finite_for_embedded_molecule() -> None:
    mol = geometry.embed_molecule_3d(Chem.MolFromSmiles("CCO"))
    features = geometry.molecule_geometry_features(mol, prefix="shape")

    assert features["shape_embed_success"] == 1.0
    assert features["shape_radius_of_gyration"] > 0.0
    assert 0.0 <= features["shape_npr1"] <= features["shape_npr2"] <= 1.0
    assert all(np.isfinite(value) for value in features.values())


def test_molecule_descriptor_error_returns_complete_zero_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mol = _linear_molecule([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0)])

    def fail_descriptor(_mol: Chem.Mol) -> float:
        raise RuntimeError("forced descriptor failure")

    monkeypatch.setattr(geometry.rdMolDescriptors, "CalcRadiusOfGyration", fail_descriptor)
    assert geometry.molecule_geometry_features(mol) == geometry.zero_molecule_geometry_features()


def test_coordinate_helpers_have_known_values_and_empty_boundaries() -> None:
    mol = _linear_molecule([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0)])
    assert geometry._coords_for_atoms(mol, []).shape == (0, 3)
    np.testing.assert_allclose(geometry._coords_for_atoms(mol, [0, 2]), [[0, 0, 0], [3, 0, 0]])

    symmetric = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    assert geometry._radius_of_gyration(np.zeros((0, 3), dtype=np.float32)) == 0.0
    assert geometry._radius_of_gyration(symmetric) == pytest.approx(1.0)
    assert geometry._span(symmetric) == pytest.approx(2.0)
    assert geometry._span(symmetric[:1]) == 0.0


def test_invalid_atom_indices_return_empty_coordinates_instead_of_crashing() -> None:
    mol = _linear_molecule([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0)])

    assert geometry._coords_for_atoms(mol, [-1]).shape == (0, 3)
    assert geometry._coords_for_atoms(mol, [mol.GetNumAtoms()]).shape == (0, 3)
    assert geometry._coords_for_atoms(mol, [None]).shape == (0, 3)


def test_site_features_match_controlled_geometry() -> None:
    mol = _linear_molecule([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0)])
    features = geometry.site_geometry_features(mol, source_atoms={0}, local_atoms={0, 1})

    assert features["site_geom_embed_success"] == 1.0
    assert features["site_geom_source_centroid_to_mol_centroid"] == pytest.approx(4.0 / 3.0)
    assert features["site_geom_local_centroid_to_mol_centroid"] == pytest.approx(5.0 / 6.0)
    assert features["site_geom_source_radius_of_gyration"] == 0.0
    assert features["site_geom_local_radius_of_gyration"] == pytest.approx(0.5)
    assert features["site_geom_source_span"] == 0.0
    assert features["site_geom_local_span"] == pytest.approx(1.0)
    assert features["site_geom_mean_source_to_local"] == pytest.approx(0.5)
    assert features["site_geom_min_source_to_local"] == 0.0
    assert features["site_geom_max_source_to_local"] == pytest.approx(1.0)


def test_empty_local_set_uses_source_atoms_as_local_environment() -> None:
    mol = _linear_molecule([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0)])
    features = geometry.site_geometry_features(mol, source_atoms={1}, local_atoms=set())

    assert features["site_geom_embed_success"] == 1.0
    assert features["site_geom_source_centroid_to_mol_centroid"] == pytest.approx(
        features["site_geom_local_centroid_to_mol_centroid"]
    )
    assert features["site_geom_mean_source_to_local"] == 0.0
    assert features["site_geom_min_source_to_local"] == 0.0
    assert features["site_geom_max_source_to_local"] == 0.0


@pytest.mark.parametrize(
    ("mol", "source_atoms", "local_atoms"),
    [
        (None, {0}, {0}),
        (Chem.MolFromSmiles("CC"), {0}, {0}),
        (_linear_molecule([(0, 0, 0), (1, 0, 0), (2, 0, 0)]), set(), {0}),
        (_linear_molecule([(0, 0, 0), (1, 0, 0), (2, 0, 0)]), {99}, {0}),
        (_linear_molecule([(0, 0, 0), (1, 0, 0), (2, 0, 0)]), {0}, {99}),
    ],
)
def test_site_features_use_zero_fallback_for_unavailable_geometry(
    mol: Chem.Mol | None,
    source_atoms: set[int],
    local_atoms: set[int],
) -> None:
    assert geometry.site_geometry_features(mol, source_atoms, local_atoms) == (
        geometry.zero_site_geometry_features()
    )


def test_site_features_are_translation_and_rotation_invariant() -> None:
    coordinates = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [3.0, 1.0, 1.0]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = coordinates @ rotation.T + np.asarray([7.0, -4.0, 2.0])
    first = _linear_molecule([tuple(row) for row in coordinates])
    second = _linear_molecule([tuple(row) for row in transformed])

    first_features = geometry.site_geometry_features(first, {0, 1}, {0, 1, 2})
    second_features = geometry.site_geometry_features(second, {0, 1}, {0, 1, 2})
    assert first_features == pytest.approx(second_features, abs=1e-6)


def test_rdkit_distance_matrix_agrees_with_site_span() -> None:
    """Guard the intended Euclidean interpretation against feature drift."""
    mol = _linear_molecule([(0.0, 0.0, 0.0), (3.0, 4.0, 0.0), (3.0, 4.0, 12.0)])
    features = geometry.site_geometry_features(mol, {0, 1}, {0, 1, 2})

    assert features["site_geom_source_span"] == pytest.approx(5.0)
    assert features["site_geom_local_span"] == pytest.approx(13.0)
