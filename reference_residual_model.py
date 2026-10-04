"""Compatibility export for loading canonical Stage 1 model bundles.

Pipeline entrypoints place ``scripts/`` on ``sys.path`` and pickle the
estimator as ``reference_residual_model.ReferenceResidualRegressor``. Keeping
this root-level export makes the same bundle importable from an ordinary Python
session launched at the repository root.
"""

from scripts.reference_residual_model import ReferenceResidualRegressor

__all__ = ["ReferenceResidualRegressor"]
