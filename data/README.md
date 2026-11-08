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

## Protein-environment research data

Experimental protein-residue pKa tables and corresponding structure snapshots
downloaded for the optional binding-environment correction are stored locally at
`raw/external_datasets/protein_environment/`. They are ignored by Git and are
not part of the aqueous pKate training pool. They train a separate, explicitly
lower-confidence residual artifact when
`scripts/train_protein_environment_model.py` is invoked. The acquisition
record, quality findings, licensing cautions, and implementation boundary are in the
[protein-environment data audit](../docs/PROTEIN_ENVIRONMENT_DATA_AUDIT.md).
The trainer also writes an ignored 648-row calibration ledger: 643
homology-grouped OOF predictions and five explicit missing-site-atom failures.

The small committed
[`benchmarks/binding_induced_pka_shifts.csv`](benchmarks/binding_induced_pka_shifts.csv)
manifest is different: it contains 13 literature-derived experimental challenge
targets, is explicitly evaluation-only, and is never selected by a trainer.
Its committed
[`structural mapping`](benchmarks/binding_induced_pka_structure_mapping.json)
locks target chains/sites, ligand-removal controls, coupled-site semantics, and
RCSB coordinate hashes. Acquire those ignored coordinate files with
`python scripts/acquire_binding_shift_structures.py`.
The PKAD-2 leakage audit and benchmark protocol are documented in the
[protein-environment benchmark report](../docs/PROTEIN_ENVIRONMENT_BENCHMARK.md).

PKAD-3 split-0 candidates are prepared with
`scripts/prepare_pkad3_training_data.py`, and their 172 unique RCSB entries are
acquired with `scripts/acquire_pkad3_structures.py`. All KaML computed features
and `model_pka` are ignored. Exact PKAD-R sites are removed and pKate
descriptors are recomputed. The current 990-row candidate fails the
group-balanced deployment gate, so no PKAD-3 label enters the fitted model.

The committed
[`benchmarks/metalloprotein_pka_2026.csv`](benchmarks/metalloprotein_pka_2026.csv)
contains 12 experimental, evaluation-only transitions from the 2026
metalloprotein paper. It is never read by a trainer. The author SI can be
checksum-acquired with
`python scripts/acquire_metalloprotein_pka_supporting_information.py`; its
CC BY-NC 4.0 coordinate archive and calculated-pKₐ tables remain ignored.
`python scripts/benchmark_metalloprotein_pka.py --published-qm` scores the
paper's external predictions without exposing them to fitting.

## PLBD proton-linkage evidence

Acquire and prepare the CC BY 4.0 PLBD snapshot with:

```bash
python scripts/acquire_plbd_data.py
python scripts/prepare_plbd_proton_linkage.py
```

The pinned archive yields 225 experimental compound-pKₐ rows, 50 protein-pKₐ
rows, 1,661 intrinsic-parameter rows, and 1,649 joined linkage constraints.
Intrinsic binding parameters have already been corrected for linked
protonation; they are suitable for consistency checks or weak evaluation, but
are not direct measurements of a bound ligand's site pKₐ. Accordingly, PLBD
contributes zero direct bound-ligand calibration rows.

## Direct free/bound ligand calibration candidates

[`curation/direct_bound_ligand_pka_candidates.csv`](curation/direct_bound_ligand_pka_candidates.csv)
is a 15-case, row-level experimental literature registry. Acquire the seven
exact mapped complexes and evaluate the fail-closed calibration gate with:

```bash
python scripts/acquire_direct_bound_ligand_structures.py
python scripts/prepare_direct_bound_ligand_calibration.py
python scripts/train_protein_environment_model.py
```

Methotrexate–DHFR (`3DFR`), trimethoprim–DHFR (`2HM9`), and five human
GST–glutathione structures (`1PKW`, `1TDI`, `1XW6`, `1XW5`, and `5GSS`)
currently meet the exact-label, exact-ligand, exact-structure, and mapped-atom
criteria. Homologous GSTs share groups, so these seven rows represent four
protein groups and two chemistries, below the required 12 rows, six protein
groups, and three chemistries. The direct calibration gate is therefore closed
and the protein-environment bundle records zero deployed direct rows.

## pKahub candidate expansion

The pKahub aggregate is an optional, provenance-gated supplement. Its
experimental macroscopic values are useful, but almost all published
charge-transition assignments were generated with Epik and never serve as
independent pKate labels. Download `all_datasets.tsv` from the
[pKahub datasets page](https://pkahub.ttk.hu/datasets) into
`raw/external_datasets/pkahub/`, then run:

```bash
python scripts/audit_pkahub_transition_assignability.py
python scripts/prepare_pkahub_training_data.py
python scripts/stage1_train_intrinsic_pka.py
python scripts/build_molecule_microstate_network_dataset.py
python scripts/stage2_train_network_context.py
```

The raw and derived files are ignored and remain subject to their source-level
terms. Stage 1 and Stage 2 each have a canonical-only performance gate; a
supplement is fitted only when the corresponding exact-label holdout does not
worsen. The current build deploys 4,801 single-transition Stage 1 rows and
14,247 complete macro-site Stage 2 rows at capped weights; 8,467 weak rows with
incomplete state spaces remain blocked. The methodology, snapshot counts,
transition-certainty classes, and safe integration boundary are documented in the
[pKahub transition-assignment survey](../docs/PKAHUB_TRANSITION_ASSIGNMENT_SURVEY.md).

## CASP16 ligand pKa and complex provenance

Acquire and prepare the CASP16 pharmaceutical ligand release with:

```bash
python scripts/acquire_casp16_pka_data.py
python scripts/prepare_casp16_training_data.py
python scripts/acquire_casp16_complex_structures.py
```

Only numeric experimental `PKA1`/`PKA2` fields are eligible. ADMET Predictor
columns are always ignored. Thirty-two single-transition rows pass the joint
Stage 1 supplemental gate. The five macro values on three multisite molecules
now have complete four-state networks after plain indole was classified as an
extreme-range context rather than an unresolved basic site; the Stage 2 gate
nevertheless rejects those five rows because canonical scaffold MAE worsens.
The 18 associated complexes preserve mapping provenance but are ineligible for
protein-environment fitting because the reported ligand pKas are not identified
as bound-state measurements.
