# pKahub transition-assignment survey

## Why the supplied transitions are not labels

The pKahub flat file contains experimental macroscopic pKa values, but its
charge-state transitions are predominantly computational assignments.  The
[pKahub methodology](https://pkahub.ttk.hu/about) states that experimental
values are matched to Epik predictions.  In the downloaded 2026-10-22 flat
file, 97,603 of 97,614 rows use an Epik/Epikx assignment method; only 11 are
marked manual.  Those transition fields must therefore not supervise pKate,
because doing so would reintroduce a proprietary computational teacher.

The experimental pKa values themselves remain potentially useful.  Their safe
use depends on how much the molecular structure establishes without the Epik
assignment.

## Reproducible structural audit

Download `all_datasets.tsv` from the
[pKahub dataset page](https://pkahub.ttk.hu/datasets) to
`data/raw/external_datasets/pkahub/all_datasets.tsv`, then run:

```bash
python scripts/audit_pkahub_transition_assignability.py
```

The script deliberately ignores `charge_state_pre`, `charge_state_post`, and
the microspecies fields.  It applies pKate's open SMARTS library and writes a
molecule-level ledger, source summary, and JSON summary under
`data/processed/dataset_audit/pkahub_transition_assignment/`.

The 2026-10-22 snapshot contains 97,614 measurements, 31,272 molecule IDs, and
31,250 unique SMILES.  The structural audit found:

| Structural result | Molecules | Fraction |
|---|---:|---:|
| exactly one supported transition, no unresolved context | 10,807 | 34.6% |
| exactly one transition plus unresolved context | 82 | 0.3% |
| multiple candidate transitions | 17,268 | 55.2% |
| no currently supported transition | 3,115 | 10.0% |

All 31,272 SMILES parsed.  There were 276 molecules with at least one
unresolved context.  The supplied computational assignment error has median
0.291 and p90 1.384 pKa units; that is a teacher-matching diagnostic, not an
experimental site-label uncertainty.

Exact canonical-SMILES comparison against the current intrinsic master found
7,700 structures already present and 23,550 structurally new entries.  This is
only a first overlap screen: source lineage, reference, conditions, salt form,
tautomer, and repeated measurements still require deduplication.

## Implemented integration

Run the source preparation with:

```bash
python scripts/prepare_pkahub_training_data.py
```

The preparation excludes SAMPL6/7/8 and euroSAMPL1, removes molecule-identity
overlap with canonical pKate data, collapses exact duplicate measurements,
screens references/evidence, and rejects replicate spans above 2 pKa units. It
never reads the supplied charge-state or microspecies assignments as labels.

1. The current preparation identifies 2,450 non-overlapping one-transition
   candidate molecule keys. No row passes the complete current Stage 1 policy:
   1,984 fail evidence eligibility and 466 exceed the two-unit replicate-range
   threshold. The resulting site-supplement CSV is intentionally empty.
2. Multi-transition molecules remain unordered macroscopic examples. Current
   source/reference and tractability controls retain 48 macro sets represented
   by 54 transition placeholders; 736 candidates are quarantined.
3. pKahub macro evidence may consume only weight left under a total weak-label
   cap of 25% of exact-site weight, after preserving the pre-existing weak
   cohort. In the current rebuilt network every retained pKahub macro value is
   blocked by the complete-candidate-space rule, so effective weight is zero.
4. Exact-site rows alone select the model, establish family support, and define
   validation truth. Weak pKahub rows cannot do any of those things.
5. Quarantine unresolved or incomplete candidate networks. Guessing a site
   from an incomplete candidate set would convert ambiguity into a false exact
   label.
6. Preserve source/reference groups and keep declared external benchmarks out
   of fitting even when they occur in the aggregate.
7. Treat the prepared tables as policy-versioned derived data. An older local
   snapshot contained thousands of accepted rows, but the current preparation
   code cannot reproduce that cohort and it must not be reused for training.
8. Do not interpret any macroscopic pKa fit as observed microstate accuracy.
   Rows blocked by the completeness rule are useful audit hypotheses, not new
   labels.

The prepared files, combined network, model, and audit tables are ignored Git
artifacts. Redistribution and model-release decisions still require checking
the terms of every contributing pKahub source.

## General boundary

The original structural conclusions remain useful:

1. The 10,807 clean single-transition molecules can receive deterministic
   structural site labels, but still require provenance and quality controls.
2. The 17,268 multi-transition molecules must remain macroscopic examples. Do
   not select a microscopic site merely because it minimizes training error.
3. Quarantine the 82 unresolved single-transition cases and inspect the 3,115
   zero-transition cases.  Some will expose detector gaps; others may be
   nonionizable structures, mixtures, or implausible imported measurements.
4. Split by canonical molecular identity before expanding measurements, and
   preserve source/reference groups across folds.  Aggregated pKahub sources
   overlap one another, so rows are not independent observations.
5. Keep SAMPL challenge sets and any other declared external benchmarks out of
   fitting even when they appear in the pKahub aggregate.

Thus transition assignment is structurally determined for roughly one third of the
collection.  For the majority it is not a missing SMARTS problem: a
macroscopic titration value does not uniquely identify a microscopic site in a
multi-site molecule.  Those data are still valuable, but at the Stage 3
macroscopic-network level rather than as forced Stage 1 labels.
