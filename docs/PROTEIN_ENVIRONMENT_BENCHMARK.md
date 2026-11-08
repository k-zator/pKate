# Protein-environment calibration and benchmark policy

Benchmark snapshot: 2026-10-26

## Outcome

The current protein-residue environment model passes a useful **PKAD-2
portability check**, but PKAD-2 cannot provide a statistically meaningful
independent test of a model trained on PKAD-R. PKAD-R is a curated successor
to PKAD-2, and the two resources share measurements, structures, references,
and near-identical proteins.

After filtering the revised PKAD-2 wild-type sheet, the audit found:

| Cohort step | Rows |
|---|---:|
| PKAD-2 wild-type rows | 1,450 |
| exact numeric Asp/Glu/His/Lys/Cys/Tyr labels | 1,300 |
| exact site matches to strict PKAD-R training rows | 507 |
| rows sharing a training reference | 1,147 |
| source rows left after site/PDB/reference/supplied-homology filtering | 104 |
| unique experimental measurements after deduplication | 73 |
| structure-mapped portability predictions | 72 |
| rows disjoint at the conservative 0.80 near-duplicate proxy | 1 |

The overlap counts are not mutually exclusive. The one nominally disjoint
case is `P03230 A ASP150`, with experimental pKₐ 7.37. pKate predicts 4.39
(90% interval 2.39–6.39), an absolute error of 2.98 pKₐ units. One observation
is not an external benchmark; it is a warning case. The similarity proxy is
not a substitute for a formal remote-homology analysis.

## PKAD-2 portability result

On the 72 mapped, measurement-deduplicated portability rows:

| Predictor | MAE | RMSE | p90 absolute error |
|---|---:|---:|---:|
| aqueous residue reference only | 0.676 | 0.946 | 1.600 |
| ungated environment model | **0.482** | **0.716** | 1.062 |
| deployed chemistry-gated model | 0.496 | 0.743 | 1.108 |

The gated 90% intervals cover 97.2% of these rows. This suggests conservative
uncertainty on this related-data cohort. It does not prove 90% coverage on new
protein families. Carboxyl MAE improves from 0.771 to 0.518 over 39 rows;
imidazole improves from 0.575 to 0.478 over 32 rows. The only phenol row stays
at the reference prediction because the phenol correction gate is withheld.
No independent Lys or Cys evaluation survives the provenance filters.

These metrics must not be used to tune hyperparameters, chemistry gates,
reference pKₐ values, or interval widths. Doing so would turn the portability
set into development data and make even this limited check circular.

## Binding-induced pKₐ shift panel

[`binding_induced_pka_shifts.csv`](../data/benchmarks/binding_induced_pka_shifts.csv)
transcribes Table S1 of Aguilar *et al.*, *Biophysical Journal* 98, 872–880
(2010), DOI `10.1016/j.bpj.2009.11.016`, and links every row to its primary
experimental paper.

The panel contains 13 documented shifts:

- seven exact numeric targets and six one-sided bounds;
- eight individual residue targets and five coupled-site titration midpoints;
- eight rows with distinct apo/holo PDB entries; and
- five comparisons in which “apo” means computational ligand removal from
  the holo coordinates, not an independently measured apo structure.

The panel is intentionally a challenge set, not training data. Coupled targets
may only be scored against a combined titration midpoint prediction. Bounds are
scored by inequality satisfaction and violation, not by treating the boundary
as an exact value.

All 13 cases now have explicit, checksum-locked structural mappings in
[`binding_induced_pka_structure_mapping.json`](../data/benchmarks/binding_induced_pka_structure_mapping.json).
The mapping records repeated crystallographic copies, ligand-removal controls,
individual versus coupled targets, and three rows that require a homodimer or
missing-symmetry-mate convention. It never silently substitutes a different
site.

### Frozen-model challenge result

The model was retrained on the 643 descriptor-ready experimental PKAD-R main
sites plus 990 de-overlapped PKAD-3 rows at effective weight 160.75. The 95
mutant/state endpoints were audited in the same homology folds but not deployed
because paired-delta OOF error worsened. The frozen model was then evaluated on
this untouched panel; all 13 cases mapped and none were excluded.

| Challenge metric | Result |
|---|---:|
| exact shifts scored | 7 / 7 |
| exact-shift MAE | 2.941 pKₐ units |
| exact-shift RMSE | 3.158 pKₐ units |
| exact-shift 90% interval coverage | 42.9% |
| censored bounds satisfied | 3 / 6 (50.0%) |

This is a clear negative result. The current static local-count and
charge-distance representation transfers modestly on PKAD-2-like protein
residue pKₐs but does not reproduce large binding-induced shifts. In
particular, ligand-removal controls barely change the descriptors in several
cases where the experiment reports a 2.5–5.5 unit shift. The challenge panel
must remain untouched evaluation data; fitting its 13 labels would conceal the
representation failure rather than solve it.

## Teacher-leakage controls

The benchmark runner fails closed unless every evaluation row states:

```text
label_origin = experimental
computational_teacher = ""
evaluation_only = true
eligible_for_training = false
```

It also verifies the model's recorded PKAD-R training-source SHA-256, rejects
byte-identical evaluation/training sources, writes every PKAD-2 overlap reason,
and keeps exact, censored, individual-site, and coupled-site metrics separate.
Computational methods such as PROPKA, Poisson–Boltzmann calculations, or
constant-pH simulations may be recorded as comparator predictions, but their
outputs may not be labels in either benchmark.

## Reproducing the audit

Install the environment, acquire the ignored PKAD-R and revised PKAD-2
snapshots described in the [data audit](PROTEIN_ENVIRONMENT_DATA_AUDIT.md),
download the checksum-locked challenge structures, train the local residual
artifact, then run:

```bash
python scripts/acquire_binding_shift_structures.py
python scripts/train_protein_environment_model.py
python scripts/benchmark_protein_environment.py
```

The ignored output directory contains:

```text
data/processed/ml_models_experimental_only/protein_environment/benchmark/
  benchmark_summary.json
  pkad2_overlap_ledger.csv
  pkad2_portability_predictions.csv
  pkad2_prediction_exclusions.json
  pkad2_metrics.json
  binding_shift_model_predictions.csv
  binding_shift_scored_predictions.csv
  binding_shift_prediction_exclusions.json
```

To score predictions for the binding panel, supply a CSV with `case_id`,
`predicted_delta_pka`, and `prediction_semantics`:

```bash
python scripts/benchmark_protein_environment.py \
  --binding-predictions binding_shift_predictions.csv
```

The source workbook, prediction model, binding manifest, and structural mapping
hashes are written to `benchmark_summary.json` for each run.

## PKAD-3 augmentation gate

PKAD-3 is not scored as an independent benchmark: it overlaps PKAD-R heavily
and its published split files omit row-level references. The fitting
integration removes 610 exact PKAD-R sites, recomputes project descriptors for
990 of 994 remaining structure rows, connects UniProt and PKAD-R homology
components, and evaluates augmentation only on held-out canonical PKAD-R
sites. At its 25%-of-canonical weight cap, pooled MAE improves from 0.9023 to
0.8701 and equal-group MAE improves from 1.1516 to 1.1137; the 990-row
augmentation is therefore deployed. Its effective total weight is capped at
160.75, exactly 25% of the 643 canonical rows.

## Direct bound-ligand calibration gate

The literature registry has 15 experimental free/bound ligand cases. Seven
currently satisfy every strict requirement: an exact point label, exact ligand
and protein identity, exact complex coordinates, and a mapped site atom. They
comprise two DHFR heteroaromatic ligands (`3DFR` and `2HM9`) and five human
GST–glutathione thiol transitions (`1PKW`, `1TDI`, `1XW6`, `1XW5`, and
`5GSS`). Homologous Alpha- and Mu-class GSTs share groups, so the strict set
spans four protein groups and two chemistries rather than seven independent
proteins.

Run the checksum-locked structure acquisition and gate with:

```bash
python scripts/acquire_direct_bound_ligand_structures.py
python scripts/prepare_direct_bound_ligand_calibration.py
python scripts/train_protein_environment_model.py
```

Deployment requires at least 12 rows, six protein groups, and three
chemistries before grouped OOF performance is eligible for evaluation. The
current gate is therefore closed and the fitted protein bundle records
`direct_bound_ligand_calibration_rows: 0`. This is a data-insufficiency result,
not a zero-shift experimental conclusion.

PLBD is kept separate from this gate: its protonation-corrected intrinsic
binding parameters support linkage consistency checks but are not direct
site-resolved free/bound ligand pKₐ pairs.

## 2026 metalloprotein specialist panel

[`metalloprotein_pka_2026.csv`](../data/benchmarks/metalloprotein_pka_2026.csv)
transcribes the 12 experimental Table 1 transitions in DOI
`10.1021/acs.jctc.6c00884`. It is a locked evaluation-only panel outside the
current six-residue-class calibration. Validate it, or score an external
prediction table containing `case_id`, `predicted_pka`, and
`prediction_semantics`, with:

```bash
python scripts/acquire_metalloprotein_pka_supporting_information.py
python scripts/benchmark_metalloprotein_pka.py
python scripts/benchmark_metalloprotein_pka.py --published-qm
python scripts/benchmark_metalloprotein_pka.py \
  --predictions metalloprotein_predictions.csv
```

Exact, approximate, and upper-bound targets are reported separately. No row
can be consumed by a trainer. The author SI is CC BY-NC 4.0 and remains
ignored. The paper's published external QM table obtains exact MAE 1.525 on
eight cases and approximate MAE 4.20 on three; its single upper-bound case is
violated by 1.60 pKₐ units. These are external published-method scores rather
than pKate predictions.
