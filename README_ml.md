# ML Scaffold: Site/State Learning from Dull SMILES

This scaffold keeps SMARTS as candidate enumeration and trains ML to rank protonation site/state in complex molecules.

![Canonical protonator pipeline overview](docs/canonical_pipeline_overview.svg)

See [`scripts/CANONICAL_PIPELINE_GUIDE.md`](scripts/CANONICAL_PIPELINE_GUIDE.md)
for the detailed scientific and operational explanation.

## Files

- `scripts/ml_dataset_builder.py`: builds candidate-level training rows from processed assignments.
- `scripts/ml_features.py`: builds RDKit descriptor + Morgan fingerprint + candidate prior features.
- `scripts/ml_splits.py`: random and Bemis-Murcko scaffold split utilities.
- `scripts/train_site_state_model.py`: legacy candidate-ranking classifier retained for reproducibility.
- `scripts/eval_site_state_pka.py`: legacy candidate-table evaluation.
- `scripts/stage1_train_intrinsic_pka.py`: canonical Stage 1 experimental-only training entrypoint.
- `scripts/pka_evidence_policy.py`: evidence tiers, chemistry blockers, and
  transition-specific reference priors.
- `scripts/reference_residual_model.py`: import-stable Stage 1 prior-residual estimator.
- `scripts/stage1_network_inference.py`: blind Stage 1 prediction for mapped network sites.
- `scripts/stage2_train_network_context.py`: canonical Stage 2 network-residual model.
- `scripts/stage2_apply_network_context.py`: canonical Stage 2 application and thermodynamic ordering.
- `scripts/stage2_free_energy_coupling.py`: canonical pairwise free-energy closure and populations.
- `scripts/stage3_microstate_thermodynamics.py`: Stage 3 schema and legacy inverse helpers.
- `scripts/stage3_apply_microstate_inference.py`: canonical complete-network Stage 3 application.
- `scripts/run_canonical_protonation_pipeline.py`: preferred provenance-checked Stage 2 → Stage 3 runner.
- `scripts/visualize_canonical_stage3_failures.py`: canonical Stage 3 review galleries and CSV queues.
- `scripts/PIPELINE_STAGE_MAP.md`: authoritative stage-to-file map.
- `scripts/CANONICAL_PIPELINE_GUIDE.md`: detailed scientific and operational
  explanation of all three canonical stages.
- `scripts/render_pipeline_overview.py`: reproducibly generates the SVG/PNG
  overview figure embedded in the canonical guide.
- `scripts/visualize_stage3_failures.py`: legacy candidate-table predicted-vs-true galleries.

## 1) Build candidate dataset

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/ml_dataset_builder.py \
  --assignments data/processed/functional_group_assignments.csv \
  --summary data/processed/functional_group_pka_summary.csv \
  --out data/processed/ml_site_state_candidates.csv
```

## 2) Train baseline site/state model

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/train_site_state_model.py \
  --assignments data/processed/functional_group_assignments.csv \
  --summary data/processed/functional_group_pka_summary.csv \
  --out-dir data/processed/ml_models/site_state_baseline
```

Outputs include:

- `metrics.json`
- `random_eval_candidates.csv`
- `scaffold_eval_candidates.csv`
- split manifests
- serialized models (`.pkl`)

## 3) Evaluate predictions

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/eval_site_state_pka.py \
  --predictions data/processed/ml_models/site_state_baseline/scaffold_eval_candidates.csv \
  --out-json data/processed/ml_models/site_state_baseline/scaffold_eval_metrics.json
```

## 4) Stage 1: Intrinsic pKa model (single-group)

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/stage1_train_intrinsic_pka.py \
  --network-dataset data/processed/pka_molecule_microstate_network_dataset.csv \
  --out-dir data/processed/ml_models_experimental_only/stage1_intrinsic
```

Outputs include:

- `stage1_intrinsic_model.pkl` (model + feature schema)
- `metrics.json`
- `stage1_training_sites.csv` and `stage1_training_site_quarantine.csv`
- `model_selection_summary.csv` and `model_selection_fold_metrics.csv`
- `scaffold_oof_predictions.csv` and `scaffold_split_manifest.csv`
- `label_interpolation_oof_predictions.csv` and
  `training_reconstruction_predictions.csv`
- `reference_anchor_panel_predictions.csv` (internal calibration, not external validation)

## 5) Stage 2: Network macroscopic residual model

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/stage2_train_network_context.py \
  --network-dataset data/processed/pka_molecule_microstate_network_dataset.csv \
  --out-dir data/processed/ml_models_experimental_only/stage2_network_context

/home/kate/anaconda3/envs/prot/bin/python scripts/stage2_apply_network_context.py
```

Outputs include:

- `stage2_network_context_model.pkl`
- `metrics.json` (model selection, provenance, and Stage 1 vs Stage 2 metrics)
- `training_site_table.csv` and `training_site_quarantine.csv`
- `weak_molecule_training_rows.csv` and `weak_molecule_blocked.csv`
- `model_selection_folds.csv`
- `random_eval_predictions.csv`
- `scaffold_eval_predictions.csv`
- `stage2_free_energy_application_metrics.json`
- `pka_molecule_microstate_network_stage2_predictions.csv` (application output)

Application converts the supported, ordered macroscopic prediction into one
cycle-consistent state-energy model. It reports one-body pKa terms, symmetric
pair couplings in log10-equilibrium and kJ/mol units, and a contextual pKa for
every protonation edge. Because the scalar data do not uniquely identify every
pair term, these are conservative regularized closure parameters with explicit
identifiability flags—not claimed experimental interaction energies.

The older `train_delta_context_model.py` and `stage3_protonation_state_inference.py`
belong to the legacy candidate-table pipeline and must not be mixed with the
canonical molecule-network bundles.

## 6) Canonical Stage 3 microstate inference

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/run_canonical_protonation_pipeline.py \
  --ph 7.4
```

Outputs include:

- `pka_molecule_microstate_network_stage3_predictions.csv`
- `stage3_microstates/stage3_site_predictions.csv`
- `stage3_microstates/metrics.json`
- `stage3_microstates/stage3_output_schema.json`
- `stage3_microstates/canonical_pipeline_run_manifest.json`

Stage 3 directly evaluates the Stage 2 pairwise state-energy model on the
enumerated coupled-coordinate graph. It calculates complete-network
populations, site marginals, dominant states, pKa/probability intervals, and
uncertainty-aware confidence. Serial amphoteric charge levels and
background-dependent edge pKas are represented explicitly. Pair-energy
identifiability contributes to confidence because the available evidence is
predominantly scalar macroscopic pKa rather than microstate-resolved data.
The site table separates `stage2_predicted_macro_pka` (the scalar comparable
to experiment), `stage3_one_body_pka` (a free-energy coefficient), and
`stage3_contextual_edge_pka_at_dominant_background` (a microscopic transition
in the predicted dominant background). `stage3_effective_local_pka` is retained
only as a deprecated alias of the one-body coefficient.

## 7) Visualize canonical Stage 3 review cases

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/visualize_canonical_stage3_failures.py \
  --out-dir data/processed/stage3_failure_visualization_canonical \
  --max-examples 24
```

Open `data/processed/stage3_failure_visualization_canonical/index.html` to view
seven explicitly named molecule galleries: held-out pKa errors, post-hoc
secondary-transition assignment suggestions, the deployment view of those
held-out errors, low-confidence state calls at pH 7.4, Stage 2 free-energy
reconstruction gaps, pair-coupling-changed site calls, and incomplete
microstate networks.
Each gallery has a matching CSV review queue and
the directory also contains `stage3_failure_summary.json`. Orange atoms mark
the detected site and red atoms mark its protonation center.

`heldout_pka_errors` is an internal validation category. Its paired
`deployment_view_of_heldout_errors` panel draws the representative input form
on the left and Stage 3's dominant predicted microstate on the right. The input
drawing is not an experimental protonation-state observation. The panel also
reports the dominant site form implied by the experimental pKa at pH 7.4; this
check is approximate for multisite molecules. The remaining categories are
Stage 3 diagnostics, not experimentally proven failures.

`secondary_transition_assignment_guesses` is deliberately not an automatic
fallback prediction. For a held-out error above 3 pKa units it compares the
experimental value with blind Stage 1 network pKas for other enumerated edges,
requiring at least 1 pKa unit improvement and at most 2 pKa units residual
error. An alternative that already has an exact experimental anchor is tested
as a reciprocal two-label swap. Suggestions are written only to the review
gallery/CSV and never alter training labels or deployed predictions.

### Interactive pKa–structure consistency audit

Launch the dedicated reviewer for cases where the experimental pKa implies
the opposite dominant site member from the deployed Stage 3 call:

```bash
/home/kate/anaconda3/envs/prot/bin/python \
  scripts/pka_structure_consistency_review.py
```

Open `http://127.0.0.1:8766`. Each record shows the source drawing, the form
naively implied by the scalar pKa at pH 7.4, the Stage 3 dominant microstate,
and the original SDF/ChEMBL provenance. Decisions are saved immediately to
`data/curation/pka_structure_consistency_review_decisions.csv`; source SDFs are
never edited. The reviewer supports retaining the current assignment,
discarding one structure–pKa pair or the entire source record, supplying a
same-connectivity replacement SMILES, and flagging a wrong transition
definition or direction.

The release builder consumes this overlay. Exclusions are quarantined,
replacement SMILES are revalidated before use, and transition corrections
remain quarantined until the requested conjugate-pair definition is
implemented. This last case matters for heterocycles: a high pKa on an N-H
imidazole can describe neutral-to-anion deprotonation and must not be forced
into the neutral-to-imidazolium pKaH transition merely by changing the drawn
charge.

## 8) Legacy Stage 3 candidate-table inference

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/stage3_protonation_state_inference.py \
  --assignments data/processed/functional_group_assignments_pruned_eval_unpooled_families.csv \
  --summary data/processed/functional_group_pka_summary_pruned_eval_unpooled_families.csv \
  --ph 7.4 \
  --out-csv data/processed/ml_models_pruned_eval_unpooled_families_predicted_pka/stage3_protonation/stage3_candidate_scores.csv \
  --out-json data/processed/ml_models_pruned_eval_unpooled_families_predicted_pka/stage3_protonation/stage3_metrics.json
```

This command uses the legacy candidate-table Stage 1/2 bundles. It is retained
for reproducibility but is not the consumer of the canonical molecule-network
Stage 2 output.

To score a prepared candidate CSV, reuse the same default bundle with:

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/score_candidate_set_with_pipeline.py \
  --candidates data/processed/external_datasets/sampl/combined/candidates_curated_priors.csv \
  --out-csv data/processed/external_datasets/sampl/combined/pipeline_scored_candidates.csv \
  --out-json data/processed/external_datasets/sampl/combined/pipeline_scored_metrics.json
```

Outputs include:

- `stage3_candidate_scores.csv` (candidate-level effective pKa and protonation scores)
- `stage3_metrics.json` (molecule-level top1 site accuracy and pair-form accuracy)
- Additional Stage 3 confidence columns: `selection_confidence`, `pka_eff_confidence`, `pred_effective_pka_ci_low`, `pred_effective_pka_ci_high`, and score-gap diagnostics.

## 9) Visualize legacy candidate-table Stage 3 failures (predicted vs true)

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/visualize_stage3_failures.py \
  --in-csv data/processed/ml_models/stage3_protonation/stage3_candidate_scores.csv \
  --out-dir data/processed/stage3_failure_visualization \
  --max-examples 24 \
  --failure-type all
```

Outputs include:

- `stage3_failures_all.png`
- `stage3_failures_all.csv`
- `stage3_failure_summary.csv`
- By default the visualization overlays all resolved (pruned) groups and lists top candidate `label:pKa_eff` values in legends.
- Add `--hide-all-groups` to disable all-group overlay.

## Notes

- This first baseline is intentionally 2D and light-weight for label-quality iteration.
- `candidate_prior_*` features transfer information from single-group statistics to complex structures.
- For future neutral input inference, score all candidates from `resolved_groups` and select top-k site/state by probability.
- Observed `pka_value` is excluded from site/state features by default to avoid leakage for dull-SMILES deployment.
- If needed for ablation only, enable `--include-observed-pka` during training.
- Stage 1 learns intrinsic site/group behavior from simpler systems; Stage 2 predicts the complex-system macroscopic ladder and emits one cycle-consistent pairwise free-energy model.
- Stage 3 evaluates that single free-energy model over the complete enumerated network to obtain microstate populations and per-site protonation confidence at user-specified pH.

## Functional-group detector and Stage 1 review gate

Detector schema `2.3.0` distinguishes phosphate/phosphonate/phosphinate/
phosphoramidate oxyacids and anions, phosphate-ester context, nitro context,
phenol/phenolate versus ether, and neutral alcohol versus alkoxide. Nitro and
phosphate ester are contextual groups; they are not treated as protonation
transitions themselves. The corresponding oxyacids/anions and
alcohol/alkoxide are conjugate-pair site families and can be enumerated.
Schema 2.1 additionally recognizes hydrazine/hydrazinium,
hydroxylamine/hydroxylammonium, oxonium (`R-OH2+`), protonated thiol
(`R-SH2+`), carbon-substituted primary/secondary/tertiary ammonium across
aliphatic and aromatic attachments, and both `S=O` and charge-separated
`S+-O-` sulfoxide representations.

Schemas 2.2 and 2.3 add the reviewed non-standard heterocycles and
charge-aware 1,2,3-triazole, 1,2,4-triazole, and indazole transition labels.
The latter two are represented as serial cation/neutral/anion coordinates;
1,2,3-triazole exposes only the approved neutral-to-anion transition.

Newly recognized alcohol/phosphorus measurements are not automatically
admitted as Stage 1 truth merely because only one structural site was found.
They require atom-level source evidence, a pre-existing site override, or a
terminal manual review. Generate and merge their review queue with the normal
OOF outlier queue using:

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/stage1_outlier_review.py \
  --detector-expansion-release /path/to/candidate_release.csv \
  --export-only --render-all --no-open
```

Run the click-through reviewer with:

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/stage1_outlier_review.py --web
```

The current decisions are saved immediately in
`data/curation/stage1_outlier_review_decisions.csv`; superseded decisions are
preserved in `data/curation/stage1_outlier_review_decision_history.csv`.

The canonical evidence audit is:

```bash
/home/kate/anaconda3/envs/prot/bin/python scripts/audit_pka_evidence_quality.py
```

The complete ambiguous-evidence queue is a triage backlog, not a requirement
to manually review every weak label: ambiguous rows are already excluded from
exact-site Stage 1/2 supervision.
