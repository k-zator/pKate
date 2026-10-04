"""Import-stable reference-prior residual estimator used by canonical Stage 1."""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator, RegressorMixin, clone  # type: ignore


class ReferenceResidualRegressor(BaseEstimator, RegressorMixin):
    """Predict absolute pKa as a chemistry reference plus a learned residual.

    Rows without an enabled reference prior use a separately fitted absolute
    model. This module must remain importable because fitted bundles pickle the
    estimator by its fully qualified module/class name.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator

    def fit(self, X, y, sample_weight=None):
        X_frame = pd.DataFrame(X).copy()
        y_values = np.asarray(y, dtype=float)
        self.absolute_model_ = clone(self.base_estimator)
        self.absolute_model_.fit(X_frame, y_values, sample_weight=sample_weight)
        present = X_frame["stage1_reference_prior_present"].to_numpy(dtype=float) > 0.5
        self.reference_model_ = None
        if np.any(present):
            reference = X_frame.loc[present, "stage1_reference_pka"].to_numpy(dtype=float)
            residual = y_values[present] - reference
            residual_weight = (
                None if sample_weight is None else np.asarray(sample_weight, dtype=float)[present]
            )
            self.reference_model_ = clone(self.base_estimator)
            self.reference_model_.fit(
                X_frame.loc[present], residual, sample_weight=residual_weight
            )
        self.n_features_in_ = X_frame.shape[1]
        self.feature_names_in_ = np.asarray(X_frame.columns, dtype=object)
        return self

    def predict(self, X):
        X_frame = pd.DataFrame(X).copy()
        result = np.asarray(self.absolute_model_.predict(X_frame), dtype=float)
        present = X_frame["stage1_reference_prior_present"].to_numpy(dtype=float) > 0.5
        if self.reference_model_ is not None and np.any(present):
            residual = self.reference_model_.predict(X_frame.loc[present])
            reference = X_frame.loc[present, "stage1_reference_pka"].to_numpy(dtype=float)
            result[present] = reference + np.asarray(residual, dtype=float)
        return result
