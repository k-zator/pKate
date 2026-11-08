#!/usr/bin/env python3
"""Run the fast, data-independent pKₐte unit-test and coverage suite.

The repository also contains slow degradation, retraining, and historical
pipeline tests. Those need local data/model artifacts and are intentionally
separate from the clean-clone CI gate defined here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

UNIT_TEST_FILES = (
    "tests/test_audit_and_provenance_workflows.py",
    "tests/test_canonical_stage3_visualization.py",
    "tests/test_conversion_evaluation_and_thermo_utilities.py",
    "tests/test_dataset_construction_and_splitting.py",
    "tests/test_evidence_quality_audit.py",
    "tests/test_external_proton_linkage_and_direct_ligand.py",
    "tests/test_full_pka_range_information_feeder.py",
    "tests/test_form_head_training_utilities.py",
    "tests/test_functional_group_and_microstate_dataset.py",
    "tests/test_functional_group_preparation_edges.py",
    "tests/test_geometry_features.py",
    "tests/test_intrinsic_model_roundtrip.py",
    "tests/test_interactive_fixer_orchestration.py",
    "tests/test_legacy_visualization_and_toy_checks.py",
    "tests/test_ionization_context.py",
    "tests/test_manual_review_and_sdf_rendering.py",
    "tests/test_manual_quality_control_workflows.py",
    "tests/test_metalloprotein_pka_benchmark.py",
    "tests/test_microstate_enumerator.py",
    "tests/test_ml_features.py",
    "tests/test_pka_evidence_policy.py",
    "tests/test_pka_structure_consistency_review.py",
    "tests/test_pkahub_transition_audit.py",
    "tests/test_pkahub_training_integration.py",
    "tests/test_casp16_training_integration.py",
    "tests/test_pkate_api.py",
    "tests/test_protein_environment_benchmark.py",
    "tests/test_protein_environment_correction.py",
    "tests/test_pkad3_training_integration.py",
    "tests/test_raw_data_ingestion.py",
    "tests/test_resolver_and_builder_cli_edges.py",
    "tests/test_run_canonical_protonation_pipeline.py",
    "tests/test_sdf_fixer.py",
    "tests/test_smarts_patterns.py",
    "tests/test_stage1_applicability.py",
    "tests/test_stage1_canonical_training.py",
    "tests/test_stage1_outlier_review.py",
    "tests/test_stage1_stage2_prediction.py",
    "tests/test_stage1_stage2_training.py",
    "tests/test_stage2_free_energy_coupling.py",
    "tests/test_stage2_network_context.py",
    "tests/test_stage3_microstate_thermodynamics.py",
    "tests/test_stage3_tautomer_ranking_and_calibration.py",
    "tests/test_stage3_uncertainty_confidence.py",
    "tests/test_substructure_match.py",
    "tests/test_unit_test_runner.py",
    "tests/test_visualization_and_cli_utilities.py",
)


def pytest_arguments(
    *,
    coverage: bool,
    html: bool,
    extra: Sequence[str] = (),
) -> list[str]:
    """Build stable pytest arguments while allowing caller-provided additions."""
    arguments = ["-q", *UNIT_TEST_FILES]
    if coverage:
        arguments.extend([
            "--cov=pkate",
            "--cov=scripts",
            "--cov=sdf_fixer",
            "--cov-fail-under=80",
            "--cov-report=term-missing",
            "--cov-report=xml:coverage.xml",
        ])
        if html:
            arguments.append("--cov-report=html:htmlcov")
    arguments.extend(extra)
    return arguments


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse wrapper options and retain unknown arguments for pytest itself."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Write terminal and XML coverage reports.",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Also write the browsable htmlcov report (implies --coverage).",
    )
    return parser.parse_known_args()


def main() -> int:
    """Run the unit suite from the repository root and return pytest's status."""
    args, extra = parse_args()
    coverage = bool(args.coverage or args.html)
    # Fixed relative paths stay valid when launched outside the repository.
    os.chdir(PROJECT_ROOT)
    return int(pytest.main(pytest_arguments(
        coverage=coverage,
        html=bool(args.html),
        extra=extra,
    )))


if __name__ == "__main__":
    raise SystemExit(main())
