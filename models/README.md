# Fitted model distribution

The public `pkate.predict()` API requires six fitted Stage 1/2 and held-out
calibration artifacts. The complete local training directory is approximately
824 MB because it also contains large diagnostic tables and Stage 3 galleries;
those files are not needed for prediction.

A minimal release candidate has been assembled locally at:

```text
dist/pkate-models-experimental-only.tar.gz
```

It is 52,362,906 bytes (approximately 50 MiB) and contains only:

```text
stage1_intrinsic/stage1_intrinsic_model.pkl
stage1_intrinsic/metrics.json
stage1_intrinsic/scaffold_oof_predictions.csv
stage2_network_context/stage2_network_context_model.pkl
stage2_network_context/scaffold_eval_predictions.csv
stage2_network_context/training_site_quarantine.csv
```

The archive is ignored by Git and is intended to be uploaded as a versioned
GitHub Release asset. It has not yet been uploaded. The exact release-candidate
and member hashes are recorded in
[`MODEL_ARTIFACT_MANIFEST.json`](MODEL_ARTIFACT_MANIFEST.json).

After downloading the release asset from the future GitHub Releases page,
install it from the repository root with:

```bash
mkdir -p data/processed/ml_models_experimental_only
tar -xzf pkate-models-experimental-only.tar.gz \
  -C data/processed/ml_models_experimental_only
```

The resulting paths match `ModelArtifacts.defaults()` in `pkate.py`; no path
configuration is then required. Keep the fitted bundle aligned with the Python
environment in `environment.yml`, because pickle-based scikit-learn artifacts
are not guaranteed to load across arbitrary dependency versions.

The bundle contains fitted models and calibration/provenance tables, not the
raw training SDFs and not the Epik pretraining collection.
