#!/usr/bin/env python3
"""Run the fast, data-independent protonator unit-test and coverage suite.

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
    "tests/test_canonical_stage3_visualization.py",
    "tests/test_evidence_quality_audit.py",
    "tests/test_full_pka_range_information_feeder.py",
    "tests/test_ionization_context.py",
    "tests/test_microstate_enumerator.py",
    "tests/test_ml_features.py",
    "tests/test_pka_evidence_policy.py",
    "tests/test_pka_structure_consistency_review.py",
    "tests/test_pkate_api.py",
    "tests/test_run_canonical_protonation_pipeline.py",
    "tests/test_sdf_fixer.py",
    "tests/test_smarts_patterns.py",
    "tests/test_stage1_applicability.py",
    "tests/test_stage1_canonical_training.py",
    "tests/test_stage1_outlier_review.py",
    "tests/test_stage2_free_energy_coupling.py",
    "tests/test_stage2_network_context.py",
    "tests/test_stage3_microstate_thermodynamics.py",
    "tests/test_stage3_tautomer_ranking_and_calibration.py",
    "tests/test_stage3_uncertainty_confidence.py",
    "tests/test_substructure_match.py",
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
