# Data layout and provenance

The canonical pipeline uses three data classes:

- `raw/`: external source material and locally protonation-corrected SDFs;
- `curation/`: small, version-controlled human review decisions and reference
  priors; and
- `processed/`: generated datasets, models, predictions, audits, and figures.

`raw/` and generated `processed/` artifacts are ignored by Git. This keeps the
multi-gigabyte development-only Epik material and regenerable outputs out of
the repository. It does **not** mean that the training-data origin is unknown:
the sources, revisions, mappings, and hashes are committed in
[`TRAINING_DATA_MANIFEST.csv`](TRAINING_DATA_MANIFEST.csv).

## Canonical five-file training pool

[`scripts/training_data_resolver.py`](../scripts/training_data_resolver.py)
requires these exact filenames under `data/raw/`:

| pKₐte filename | Records | Upstream file | Role in pKₐte |
|---|---:|---|---|
| `experimental_training_datasets.sdf` | 5,994 | `combined_training_datasets_unique.sdf` | Training pool; prepared ChEMBL25 and DataWarrior measurements |
| `AvLiLuMoVe_testdata.sdf` | 123 | `AvLiLuMoVe_cleaned_mono_unique_notraindata.sdf` | Included in the current training pool |
| `novartis_testdata.sdf` | 280 | `novartis_cleaned_mono_unique_notraindata.sdf` | Included in the current training pool |
| `FIXED_chembl26.sdf` | 8,503 | `chembl26.sdf.gz` | Training pool after local protonation repairs |
| `literature_compilation.sdf` | 1,765 | `literature_compilation.sdf.gz` | Training pool after local protonation repairs |

The `testdata` names are inherited from their original use as external test
sets. Despite those names, the present pKₐte configuration includes all five
files in its training pool and relies on its own grouped/scaffold splitting and
provenance controls; those two files are therefore not independent external
tests of pKₐte.

### Immediate bundle: pKₐsolver-data

- Repository:
  [wiederm/pkasolver-data](https://github.com/wiederm/pkasolver-data)
- Revision present in the development workspace:
  `96298ff7ffb9af6fcd9949ed93555ae8c11f155a`
- Relevant files:
  `Baltruschat/00_experimental_training_datasets.sdf`,
  `Baltruschat/00_AvLiLuMoVe_testdata.sdf`, and
  `Baltruschat/00_novartis_testdata.sdf`.

This is most likely the single download location from which the three named
training/test SDFs were obtained. Their SHA-256 hashes match the archived
pre-repair copies in the development workspace. The pKₐsolver repository also
contains an Epik-labelled ChEMBL pretraining collection; that separate file is
**not** one of pKₐte's five canonical experimental inputs.

### Primary provenance: Machine learning meets pKₐ

- Repository:
  [czodrowskilab/Machine-learning-meets-pKa](https://github.com/czodrowskilab/Machine-learning-meets-pKa)
- Dataset archive:
  [Zenodo DOI 10.5281/zenodo.7884512](https://doi.org/10.5281/zenodo.7884512)
- Publication:
  [Machine learning meets pKₐ](https://doi.org/10.12688/f1000research.22090.2)
- Revision inspected for the checked scientific provenance:
  `45e5300db2890d3de08a281e4253be2f2db69590`.
- The current upstream dataset release declares CC BY 4.0 in
  `datasets/LICENSE.txt`; the pKₐsolver bundle also carries its own
  `Baltruschat/LICENSE.md`. Preserve both attribution trails when publishing a
  derivative bundle.

This is the primary scientific source for the 5,994-record combined
ChEMBL25/DataWarrior set, the 123-record AvLiLuMoVe set, and the 280-record
Novartis set bundled by pKₐsolver-data. The combined SDF's `original_dataset`
property confirms the ChEMBL25/DataWarrior composition.

### Source B: Multiprotic-pKₐ-Processing

- Repository:
  [czodrowskilab/Multiprotic-pKa-Processing](https://github.com/czodrowskilab/Multiprotic-pKa-Processing)
- Revision used for the checked provenance:
  `1dc5f5f97fc44307add374c766dcb103b4bffc36`
- Upstream files: `datasets/chembl26.sdf.gz` and
  `datasets/literature_compilation.sdf.gz`.

The repository declares an MIT license for the project, identifies ChEMBL26 as
the origin of `chembl26.sdf`, and provides the publication list behind
`literature_compilation.sdf`. It does not provide a separate blanket dataset
license for those two files. Their underlying source terms and citations must
therefore be retained and checked before redistributing a derived bundle.

## Upstream files are not the final pKₐte snapshots

The upstream repositories recover the measurements and original structures,
but **not** the exact five SDFs used in the current model. The local canonical
files contain protonation-form corrections made during this project. Every
canonical file consequently has a different SHA-256 hash from its upstream
counterpart; both hashes are recorded in `TRAINING_DATA_MANIFEST.csv`.

This distinction matters:

1. Downloading the upstream files is sufficient to audit the original data.
2. Renaming them to the pKₐte filenames is not sufficient to reproduce the
   fitted model.
3. A fully reproducible public training release still needs the five corrected
   snapshots deposited as a versioned release asset, or a deterministic replay
   tool that regenerates them from the upstream originals and committed
   curation decisions.

The corrected five-file set is only about 33 MB uncompressed and approximately
1.9 MB with gzip compression. File size is therefore not the main obstacle;
source attribution, derivative-data licensing, and versioned distribution are.

## What is not canonical training data

- `chembl_epik_site_charged.csv` and the large Epik-annotated SDF are excluded
  from the experimental-only training path.
- `marvin_pKa`, `marvin_atom`, and related upstream properties may remain in an
  SDF for provenance, but the canonical builders reject Marvin and Epik values
  as labels.
- SAMPL7 and SAMPL8 clones under `raw/external_datasets/` are external
  evaluation resources, not members of the five-file training pool.
- Files prefixed `INCORRECT_` or stored under `raw/handled/` are retained local
  originals/intermediates and are not selected as canonical inputs.

## Rebuilding and inference

For a full rebuild, restore the five corrected snapshots under `data/raw/`,
verify their hashes against `TRAINING_DATA_MANIFEST.csv`, and follow the
[canonical rebuild guide](../scripts/CANONICAL_PIPELINE_GUIDE.md#10-rebuild-and-inference-workflows).
For inference only, restore the released fitted model artifacts instead.

The two manually authored site-resolution override parts remain tracked under
`processed/` for compatibility with the current release builder. The fast unit
tests use synthetic temporary fixtures and do not require raw data or trained
models.
