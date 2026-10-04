"""
Shared pytest fixtures for the protonator test suite.

Manages sys.path so scripts/ imports work, and provides session-scoped
data loaders and model fixtures used across multiple test modules.
"""

import os
import sys
import pickle

import pytest
import pandas as pd
from rdkit import Chem

# ---------------------------------------------------------------------------
# Path setup — make scripts/ and sdf_fixer/ importable
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROCESSED_DIR, "ml_models")

for _path in (PROJECT_ROOT, SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def mol_from_smiles(smi: str) -> Chem.Mol:
    """Parse SMILES into an RDKit Mol. Raises on failure."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smi!r}")
    return mol


# ---------------------------------------------------------------------------
# Session-scoped data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def assignments_df() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "functional_group_assignments.csv")
    if not os.path.exists(path):
        pytest.skip(f"Assignments CSV not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def summary_df() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "functional_group_pka_summary.csv")
    if not os.path.exists(path):
        pytest.skip(f"Summary CSV not found: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def candidate_dataset():
    """Build the full candidate dataset (session-scoped, computed once)."""
    from ml_dataset_builder import DatasetBuildConfig, build_candidate_dataset

    config = DatasetBuildConfig(
        assignments_path=os.path.join(PROCESSED_DIR, "functional_group_assignments.csv"),
        summary_path=os.path.join(PROCESSED_DIR, "functional_group_pka_summary.csv"),
    )
    if not os.path.exists(config.assignments_path):
        pytest.skip("Assignments CSV not found")
    return build_candidate_dataset(config)


# ---------------------------------------------------------------------------
# Model bundle fixtures
# ---------------------------------------------------------------------------

def _load_pkl(relpath: str):
    """Load a pickle file from the models directory, skip if missing."""
    path = os.path.join(MODELS_DIR, relpath)
    if not os.path.exists(path):
        pytest.skip(f"Model file not found: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


@pytest.fixture(scope="session")
def stage1_bundle():
    return _load_pkl("stage1_intrinsic/stage1_intrinsic_model.pkl")


@pytest.fixture(scope="session")
def stage2_bundle():
    return _load_pkl("stage2_delta/stage2_delta_model.pkl")


@pytest.fixture(scope="session")
def site_model_bundle():
    return _load_pkl("site_state_baseline/site_state_random_model.pkl")


@pytest.fixture(scope="session")
def carboxyl_form_bundle():
    return _load_pkl("carboxyl_form_head/carboxyl_form_head.pkl")


@pytest.fixture(scope="session")
def pair_form_heads_bundle():
    return _load_pkl("pair_form_heads/pair_form_heads.pkl")
