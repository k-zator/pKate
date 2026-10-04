# Canonical pKa pipeline stage map

This file is the authoritative map from conceptual stages to executable files.
For a detailed explanation of the scientific logic, training targets,
thermodynamic transformations, confidence fields, and output contracts, see
[`CANONICAL_PIPELINE_GUIDE.md`](CANONICAL_PIPELINE_GUIDE.md).

## Public deployment entrypoint

`pkate.py` is the user-facing API and CLI for a molecule that is not already in
the prepared dataset:

```bash
python -m pkate "NCCCC(=O)O" --ph 7.4
```

It creates a new atom-mapped network, applies Stage 1 to every detected
transition, passes the compatible query through Stage 2, and returns Stage 3's
complete microstate, site table, confidence, and tautomer ranking. It does not
add the query to training data or require an experimental query pKa.

## Stage 0: experimental data and microstate networks

1. `build_release_microstate_dataset.py`
   - Creates the experimental transition table and quarantine artifacts.
   - Assigns gold/silver/ambiguous evidence tiers without confusing a
     structural atom selection with source-level site attribution.
   - Preserves source-original, supplied, canonicalized, and generated
     structures as separate provenance fields.
   - Enforces persistent exact-site confirmations, reassignments and exclusions
     from `data/curation/stage1_outlier_review_decisions.csv`.
2. `build_molecule_microstate_network_dataset.py`
   - Groups transitions by molecule.
   - Enumerates atom-mapped protonation states, edges and tautomers.
   - Runs blind Stage 1 prediction for every detected site.
   - Computes the macroscopic baseline on the enumerated coupled-coordinate
     graph, including serial cation/neutral/anion azole coordinates.
   - Stores exact-site measurements and ambiguous molecule-level weak labels
     separately.

Primary output:

`data/processed/pka_molecule_microstate_network_dataset.csv`

Quality-control loop:

```bash
python scripts/stage1_outlier_review.py --export-only --render-all --no-open
python scripts/stage1_outlier_review.py --web
```

`stage1_outlier_review.py` expands every scaffold-OOF error above 3 pKa units
back to the original raw measurement and atom-indexed structure. Decisions are
record-specific and feed Stage 0 on the next rebuild; therefore they correct
the labels used by both Stage 1 and Stage 2 rather than merely hiding rows from
an evaluation report.

It also writes a complete ambiguous-evidence backlog. In the web interface,
structural confirmation is separate from explicit verification of source-site
attribution, transition identity, and measurement conditions. An ambiguous
row cannot be promoted to exact supervision by an old structural-only click.

## Stage 1: intrinsic local-site pKa

Canonical training command:

```bash
python scripts/stage1_train_intrinsic_pka.py \
  --network-dataset data/processed/pka_molecule_microstate_network_dataset.csv \
  --out-dir data/processed/ml_models_experimental_only/stage1_intrinsic
```

Files:

- `stage1_train_intrinsic_pka.py`: canonical CLI entrypoint.
- `stage1_canonical_data.py`: replicate-aware canonical single-site extraction,
  provenance, quarantine, weights, and scaffold groups.
- `train_intrinsic_single_group_model.py`: shared Stage 1 feature implementation;
  its older assignment-table trainer is not the canonical entrypoint.
- `reference_residual_model.py`: import-stable reference-prior residual estimator.
- `pka_evidence_policy.py`: transition evidence tiers, unenumerated-context
  blockers, and explicit chemistry priors.
- `stage1_network_inference.py`: Stage 1 bundle validation and blind network-site inference.

Model:

`data/processed/ml_models_experimental_only/stage1_intrinsic/stage1_intrinsic_model.pkl`

Stage 1 never receives an experimental pKa as an input feature. The network
dataset stores `stage1_intrinsic_pka` and `experimental_anchor_pka` separately.
Training compares regularized reference-residual candidates and uses the
one-standard-error rule rather than selecting a memorizing leaf-size-1 fit.
It reports training reconstruction, same-label interpolation, and five-fold
grouped-scaffold extrapolation separately, then refits deployment on all
eligible canonical rows. Each prediction carries exact-label count, nearest
same-label similarity, zero-shot status, and a label-specific pKa interval.
If an exact label is absent but an enabled transition-specific reference exists,
deployment uses that reference as a disclosed zero-shot fallback and retains the
raw cross-family model estimate for audit.

Because the current Stage 1 extractor reads the molecule network, a chemistry
definition change requires a topology refresh with the existing Stage 1 bundle,
then Stage 1 retraining, followed by a second network build with the new bundle.
This two-pass refresh avoids training from stale site definitions. A first-ever
bootstrap therefore needs a previously fitted or provisional Stage 1 bundle.
The temporary topology-refresh build must explicitly use
`--allow-stage1-detector-mismatch`; the final network build rejects a detector
version mismatch by default.

## Thermodynamic baseline between Stages 1 and 2

`coupled_network_thermodynamics()` in
`build_molecule_microstate_network_dataset.py` converts all blind Stage 1 local
pKas into an ordered macroscopic baseline using the enumerated protonation
graph. Ordinary binary sites reduce to the familiar independent-site binding
polynomial; serial amphoteric sites share one multi-level coordinate and are
not treated as independent Boolean sites.

This is not Stage 2 and contains no learned context correction.

## Stage 2: complex-network macroscopic correction

Training:

```bash
python scripts/stage2_train_network_context.py
```

Application:

```bash
python scripts/stage2_apply_network_context.py
```

Files:

- `stage2_network_context.py`: deployable context and geometry features.
- `stage2_train_network_context.py`: replicate-aware residual extraction,
  quarantine, model selection, and molecule/scaffold-safe validation.
- `stage2_apply_network_context.py`: provenance validation, family support gate,
  prediction, monotonic macro-pKa projection, and free-energy-model emission.
- `stage2_free_energy_coupling.py`: cycle-consistent one-body and symmetric
  pair-coupling closure of the predicted macroscopic ladder.

Target:

`evidence-tiered exact or marginalized weak macro pKa - blind Stage1/network macro pKa`

Model:

`data/processed/ml_models_experimental_only/stage2_network_context/stage2_network_context_model.pkl`

Prediction output:

`data/processed/pka_molecule_microstate_network_stage2_predictions.csv`

Stage 2 is applied only to multisite molecules. The general family-support gate
requires at least 50 experimental training rows; guanidine-like and
phenol/phenolate sites are explicit chemistry priorities and are enabled with
the available training data below that threshold. Other unsupported families and single-site molecules
fall back exactly to the Stage 1/network baseline. Raw predictions are retained,
while supported-family corrections are projected to a non-increasing macro
ladder with fallback entries held fixed. Validation performs this projection on
all sites in each held-out molecule, including unmeasured sites, matching
deployment. The default application also requires the exact network snapshot
used for training; `--allow-compatible-network` is an explicit inference-only
override after schema, detector, and Stage 1 provenance checks pass.

Only exact-site anchors participate in model selection and reported validation.
For deployment, ambiguous molecule-level values may be marginalized across all
candidate sites at low total weight only when the candidate network is complete.
Unenumerated ionizable contexts and truncated networks are written to
`weak_molecule_blocked.csv`; weak rows never unlock a family support gate.

Stage 2 first predicts site-associated macroscopic corrections, then emits a
single state-energy function
`log10(weight) = sum(b_i*x_i) + sum(J_ij*x_i*x_j) - nH*pH`.
Every microscopic edge pKa is derived from that function, so background-state
effects are symmetric and all thermodynamic cycles close. The pair terms are a
minimum-norm regularized closure, not direct experimental measurements: only
72 molecules have two or more exact anchors and only 19 have every multisite
step observed. Their identifiability status is stored with every molecule.

## Stage 3: complete microstate inference

Application:

```bash
python scripts/run_canonical_protonation_pipeline.py --ph 7.4
```

Files:

- `stage3_microstate_thermodynamics.py`: Stage 3 schema plus retained
  backward-compatible inverse helpers.
- `stage3_apply_microstate_inference.py`: provenance validation, complete
  network annotation, empirical pKa-side calibration, dominant-state output,
  and end-to-end evaluation.
- `stage3_tautomer_ranking.py`: configuration-wise RDKit rule-score ranking of
  every enumerated tautomer, with explicit non-thermodynamic semantics.
- `stage3_empirical_calibration.py`: hierarchical exact-label/family/global
  calibration from scaffold-held-out signed pKa residuals.
- `run_canonical_protonation_pipeline.py`: preferred Stage 2 → Stage 3 runner;
  verifies artifact hand-off and records an end-to-end hash manifest.
- `visualize_canonical_stage3_failures.py`: atom-mapped Stage 3 diagnostic
  galleries with matching CSV review queues.

Output:

`data/processed/pka_molecule_microstate_network_stage3_predictions.csv`

Flat site, ranked-tautomer, and calibration artifacts:

- `data/processed/ml_models_experimental_only/stage3_microstates/stage3_site_predictions.csv`
- `data/processed/ml_models_experimental_only/stage3_microstates/stage3_ranked_tautomers.csv`
- `data/processed/ml_models_experimental_only/stage3_microstates/stage3_empirical_call_calibration.json`

Review gallery:

```bash
python scripts/visualize_canonical_stage3_failures.py
```

Open `data/processed/stage3_failure_visualization_canonical/index.html`. The
views separate scaffold-validation pKa errors, deployment views of those same
cases, post-hoc secondary-transition assignment suggestions, low-confidence
site-state calls, Stage 2 free-energy reconstruction gaps,
pair-coupling-changed calls, and population-unavailable networks.
Only the first is a measured-label error; the other views are uncertainty and
model-applicability diagnostics.

The secondary-transition view is diagnostic only. For held-out errors above 3
pKa units it asks whether another already-enumerated edge has a closer blind
Stage 1 network pKa. Occupied alternatives are tested as reciprocal two-label
swaps. No suggestion replaces a curated site assignment or becomes a training
label without review.

Stage 3 directly evaluates the Stage 2 one-body and pair-coupling energy model
on the enumerated coupled-coordinate topology. It calculates microstate
populations and site protonation probabilities without fitting another inverse.
Truncated/incomplete networks are reported as unavailable rather than falsely
renormalized. Context-specific edge pKas now include symmetric pair terms, but
those terms remain prior-dependent because scalar macro-pKa evidence does not
uniquely determine all pair interactions.
The site-level output deliberately separates the Stage 2 macroscopic pKa used
for comparison with scalar experiments, the Stage 3 one-body coefficient, and
the context-specific microscopic edge pKa. The legacy
`stage3_effective_local_pka` column is retained as a named compatibility alias
of `stage3_one_body_pka`, not as an extra experimental observable.
Stage 2 pKa intervals are propagated through the energy model and into protonation
probability intervals. A state call whose interval crosses 50% protonation is
explicitly non-robust. A separate empirical confidence estimates whether the
observed scalar pKa lies on the side of the requested pH that supports the
Stage 3 site call. It uses hierarchical shrinkage from exact-label to family to
global scaffold-held-out signed residuals. Overall confidence combines this
empirical evidence with population decisiveness, exact-label applicability,
free-energy reconstruction quality, and coupling identifiability. This is a
pKa-side proxy, not direct validation against experimental microstate labels.

Within each populated protonation configuration, Stage 3 ranks the union of all
enumerated tautomers with RDKit's rule score and writes one auditable row per
tautomer. The final drawing is chosen by selecting the physically modeled
dominant protonation configuration first and only then its top-ranked tautomer;
equivalent reference-node seeds are freshly enumerated, deduplicated, and
limited once per configuration rather than inheriting a cap from each node;
the heuristic cannot overturn the protonation result. Normalized score weights
are deliberately labelled heuristic: no tautomer free-energy or population
training labels are available, and a truncated tautomer enumeration is flagged.

## Evidence audit

```bash
python scripts/audit_pka_evidence_quality.py
```

This writes tier-by-label counts, internal-reference plausibility conflicts,
copied source-lineage groups, and a JSON summary under
`data/processed/dataset_audit/evidence_quality/`. The reference panel is an
internal calibration panel, not an independent external benchmark.

## Legacy files

These files belong to the earlier candidate-table pipeline and are not part of
the canonical molecule-network Stage 1/2 path:

- `train_delta_context_model.py`
- `stage3_protonation_state_inference.py` (not canonical Stage 3)
- `score_candidate_set_with_pipeline.py`
- `train_site_state_model.py`
- `train_thermo_pair_form_model.py`
- existing `data/processed/**/stage2_delta/` bundles

The legacy Stage 2 used assignment-table rows, included Marvin labels, learned
`experimental - intrinsic` directly, and did not use the molecule microstate
network baseline. Do not mix legacy bundles with the canonical pipeline.
