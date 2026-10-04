# ChEMBL pKa Truth Dataset Builder

Use `build_chembl_pka_truth_dataset.py` to create a fresh pKa dataset from ChEMBL activity data.

## What this script does

- Pulls activity records from ChEMBL where `standard_type = pKa`.
- Uses ChEMBL API molecule metadata for canonical SMILES.
- Enriches rows with assay and target metadata.
- Writes three tables:
  - raw activity table
  - cleaned observation table with `keep_observation` and `drop_reason`
  - per-molecule prevalent pKa summary with conflict flags

## Example (smoke run)

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/build_chembl_pka_truth_dataset.py \
  --max-pages 2 \
  --out-raw /tmp/chembl_pka_activity_raw_smoke.csv \
  --out-clean /tmp/chembl_pka_observations_clean_smoke.csv \
  --out-summary /tmp/chembl_pka_molecule_summary_smoke.csv \
  --checkpoint /tmp/chembl_pka_pull_checkpoint_smoke.json
```

## Default output paths

- `data/raw/chembl_pka_activity_raw.csv`
- `data/processed/chembl_pka_observations_clean.csv`
- `data/processed/chembl_pka_molecule_summary.csv`

## Resume behavior

Use `--resume` to continue from the saved checkpoint (`--checkpoint` path).

## Notes on trust and provenance

- `canonical_smiles` and `canonical_smiles_api` are sourced from ChEMBL molecule API metadata.
- `keep_observation` marks rows suitable for downstream use.
- `drop_reason` explains exclusions (`missing_numeric_value`, `unsupported_relation`, `outside_reasonable_range`, `missing_canonical_smiles`, `missing_parent_molecule`).
- `conflict_flag` and `conflict_note` in the summary indicate disagreement patterns (for example, wide spread or exact/censored mismatch).
