# Fitted model distribution

The public `pkate.predict()` API requires six fitted Stage 1/2 and held-out
calibration artifacts. The complete local training directory is approximately
756 MB because it also contains large diagnostic tables and Stage 3 galleries;
those files are not needed for prediction.

A minimal release candidate has been assembled locally at:

```text
dist/pkate-models-experimental-only.tar.gz
```

It is 3,455,915 bytes (approximately 3.3 MiB) and contains only:

```text
stage1_intrinsic/stage1_intrinsic_model.pkl
stage1_intrinsic/metrics.json
stage1_intrinsic/scaffold_oof_predictions.csv
stage2_network_context/stage2_network_context_model.pkl
stage2_network_context/scaffold_eval_predictions.csv
stage2_network_context/training_site_quarantine.csv
```

The archive is ignored by Git and has not been uploaded. The current Stage 1
model includes 4,801 capped-weight pKahub transitions and 32 low-weight
experimental CASP16 transitions. Stage 2 deploys 14,247 pKahub macro-site rows
after its locked exact-site gate; the five complete CASP16 macro-site rows are
audited but rejected because they worsen held-out performance. Review the
redistribution/model-release terms of all contributing
sources before publishing this candidate as a GitHub Release asset. The exact candidate
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
configuration is then required. The bundle is serialized with scikit-learn
1.8.0, pinned by both `requirements.txt` and `environment.yml`. Pickle-based
scikit-learn artifacts are not guaranteed to load across arbitrary dependency
versions.

The bundle contains fitted models and calibration/provenance tables, not the
raw training SDFs and not the Epik pretraining collection.

## Optional protein-environment artifact

The binding-environment extension uses one additional local artifact:

```text
data/processed/ml_models_experimental_only/protein_environment/
  protein_environment_model.pkl
  metrics.json
  calibration.json
  strict_648_site_calibration_cohort.csv
  training_features_and_oof.csv
  excluded_rows.json
```

It is intentionally not part of the current six-file aqueous deployment
bundle. Rebuild it with `python scripts/train_protein_environment_model.py`
after acquiring the audited PKAD-R label/structure snapshot described in
[`docs/PROTEIN_ENVIRONMENT_DATA_AUDIT.md`](../docs/PROTEIN_ENVIRONMENT_DATA_AUDIT.md).
The locally fitted model is about 2.9 MB, but neither it nor derivative label
tables should be uploaded until PKAD-R redistribution/model-release rights are
clarified. Ordinary aqueous `pkate.predict()` calls do not require it.
