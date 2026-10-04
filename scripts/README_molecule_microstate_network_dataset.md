# Molecule-level pKa microstate network dataset

`data/processed/pka_molecule_microstate_network_dataset.csv` is the current
one-row-per-molecule dataset. It groups protonation and tautomer drawings using
the first two blocks of the standard InChIKey: connectivity and stereochemistry
are retained, while the final protonation block is removed.

The transition-level `pka_microstate_dataset.csv` remains the normalized source
table. Its experimental rows are conserved in the molecule file through
`source_transition_row_ids_json`.

## What is experimental and what is predicted

Every site receives a blind `stage1_intrinsic_pka`. Exact-site experimental pKa
values are stored separately inside `sites_json` as `experimental_measurements`
and `experimental_anchor_pka`; ambiguous values are retained separately as
`weak_experimental_measurements`. For multisite molecules exact anchors are described as
`experimental_macroscopic_site_associated`: the reviewed atom mapping identifies
the affected site, but the measured pKa can include the other microstates in the
ensemble.

Stage 1 predictions come from the model in
`data/processed/ml_models_experimental_only/stage1_intrinsic/`, trained on 2,184
chemically single-coordinate transition targets representing 2,861 exact-site
source rows and 2,841 independent measurement lineages, with no Marvin or Epik
labels.
Replicates are aggregated by median, copied source lineages are collapsed, and
evidence/dispersion-aware weights are used. The selected regularized
reference-residual Extra Trees model has five-fold grouped-scaffold OOF MAE
0.601 (RMSE 0.867, R2 0.900), same-label interpolation MAE 0.462, and training
reconstruction MAE 0.079.

The deployment model is refit on all 2,184 eligible transitions. The larger
`stage1_training_site_quarantine.csv` now includes ambiguous labels and
apparently single-site molecules with competing, unenumerated ionizable
contexts, in addition to replicate conflicts. Exact-label training counts,
nearest same-label fingerprint similarity, applicability domain, and
label-specific OOF intervals are attached to each network site. Broad family
support is no longer presented as evidence for a sparse exact label.

Reference pKas are explicit transition-specific priors used to shrink sparse
chemistry, and the model learns residuals from them. When an exact transition
label is absent from training, an enabled reference is used as an explicitly
marked zero-shot fallback; the raw cross-family ML estimate is retained for
audit. Pyrrole, pyrazole, imidazole, and aziridine have transition-appropriate
priors and labels. The 15-row reference anchor panel is project
calibration data, not an independent external benchmark; external performance
therefore remains unestablished.

The blind network baseline uses only `stage1_intrinsic_pka`. Experimental
anchors are labels for Stage 2 and never replace Stage 1 predictions.

Experimental anchors carry structural assignment confidence separately from
`source_site_evidence_confidence`. Most legacy SDF pKas are molecule-level
scalars without an atom annotation; manual overrides do not upgrade that
source-level evidence.

- `stage1_experimental_only_intrinsic_prediction`: deployable Stage 1 input.
- `stage1_reference_prior_zero_shot_exact_label_fallback`: deployable baseline
  only where the exact transition label has no training example.
- `experimental_anchor_pka`: eligible Stage 2 label, subject to the
  site-associated macroscopic interpretation stated in `anchor_caveat`.

## Network representation

- `sites_json`: mapped ionization sites, measurements, Stage-1 priors and the
  rank-associated macroscopic step.
- `microstate_nodes_json`: complete mapped molecular protonation states, their
  tautomer ensembles, formal charges, site forms and pH population estimates.
- `microstate_edges_json`: single-proton acid/base edges. Every edge connects
  states whose total formal charges differ by exactly one.
- `baseline_macro_pka_steps_json`: predicted macroscopic steps from the
  enumerated-coordinate binding polynomial. The legacy
  `independent_site_macro_pka_steps_json` name is retained as a compatibility
  alias.

For ordinary independent binary sites with local pKa values `p_i`, this reduces
to coefficients of

`product_i (1 + 10^p_i * [H+])`.

The predicted macroscopic pKa for protonation step `k` is

`log10(coefficient_k / coefficient_(k-1))`.

This includes the statistical effect of multiple sites. Alternate atom-centered
drawings of the same acid/base configuration share that configuration's
population; merely enumerating more drawings does not create an energetic or
statistical advantage.

Amphoteric 1,2,4-triazole, indazole, pyrazole, and imidazole use one three-level coordinate
(`anion < neutral < cation`) with two serial transition pKas. 1,2,3-triazole
currently exposes only its user-approved neutral-to-anion transition. This
prevents the two edges of one ring from generating a fictitious fourth state.

## Current snapshot

- Dataset schema 2.9.0 contains 6,131 retained molecule networks. All are
  complete and population-eligible.
- 8,343 experimental transition records: 4,169 exact-site candidates and
  4,174 ambiguous molecule-level weak labels.
- 9,764 detected sites. Of these, 2,152 have no exact-label Stage 1 training
  example and are explicitly marked zero-shot.
- 40,787 protonation states and 161,740 one-proton edges. The largest network
  has 4,096 states, equal to the recorded `protonation_state_cap`; no retained
  network is truncated and no retained site lacks a valid edge.
- The current `pka_molecule_microstate_network_quarantine.csv` is empty. Invalid
  transition records are rejected earlier by the release-dataset quarantine and
  are not converted into invented microstates.

The builder still treats a truncated or incomplete network as unavailable, so a
partial graph cannot masquerade as a complete molecule. There are zero such
networks in the current retained snapshot.

## Limitations

The Stage 2 thermodynamic layer now emits one-body terms and symmetric pair
couplings, so a site's edge pKa changes with the other sites' protonation state
while all graph cycles remain closed. These couplings are a regularized closure
of the predicted macroscopic ladder, not uniquely measured site-pair energies:
only 70 molecules have two or more exact anchors, and only 16 have all multisite
steps observed. Stage 3 now ranks tautomers within each protonation
configuration with RDKit's rule score. That score is not a physical energy, so
its normalized weights are explicitly heuristic and cannot be interpreted as
calibrated tautomer populations. The final drawing is selected from the
physically dominant protonation configuration, so the tautomer heuristic cannot
change the protonation call. Consequently the protonation-configuration
populations remain auditable physical-model predictions, while the final
tautomer choice is a disclosed rule-based rank rather than experimental
microstate ground truth.

The canonical Stage 2 output is
`data/processed/pka_molecule_microstate_network_stage2_predictions.csv`. Stage 2
predicts a correction to the blind macroscopic baseline and converts it to a
cycle-consistent pairwise free-energy model. The output contains the one-body
terms, pair couplings, their identifiability diagnostics, and the contextual
pKa of every enumerated edge.

Stage 2 training uses 966 consistent exact-site anchors; weak labels never
enter model selection or reported validation. Deployment additionally uses 334
independent molecule-level measurements, expanded to 865 candidate rows by
marginalizing over complete candidate networks at 0.2 total weight per
measurement. Another 3,840 weak measurements are preserved in `weak_molecule_blocked.csv`
because their candidate-site space remains incomplete or unavailable, making
marginalization unsafe. A six-model scaffold-grouped comparison selected Extra
Trees with leaf size 2. On the exact-only scaffold
split, the deployment-matched pairwise free-energy prediction improves the
Stage 1/network MAE from 1.490 to 1.020
pKa units.
It is enabled for amine and carboxyl under the general 50-anchor rule;
guanidine-like and phenol/phenolate remain explicit priority families with 17
and 32 exact anchors. Sparse non-priority families fall back exactly to the
Stage 1/network baseline. Validation and deployment both project the complete
molecular ladder.

The Stage 2 target remains macroscopic, so arbitrary individual pair energies
are underdetermined. The implemented minimum-norm closure stays near the Stage
1 plus Stage 2 site-wise proposal and penalizes pair terms; it explicitly marks
all inferred couplings as prior-dependent and not experimentally identified.
The canonical `stage3_apply_microstate_inference.py` directly evaluates that
energy function rather than performing another macro-to-local inverse. The
existing `stage3_protonation_state_inference.py` remains a legacy
candidate-table script.

At pH 7.4, canonical Stage 3 produces normalized populations for all 6,131
retained molecule networks and marks zero networks unavailable. The pairwise
free-energy macro-ladder reconstruction MAE is 0.022 pKa units. The aggregate
internal held-out pKa summary is MAE 0.635 across 2,376 anchors: Stage 1
five-fold scaffold OOF single-coordinate MAE is 0.601 and Stage 2 exact-only scaffold
holdout pairwise free-energy MAE is 1.020. These are different internal validation protocols and are
reported separately; neither is an external benchmark.

Stage 3 schema 1.6.0 ranks 176,876 tautomer drawings across 37,026 populated
protonation configurations. It always chooses the dominant protonation
configuration before selecting that configuration's rank-1 tautomer, so the
rule score cannot overturn the pKa/free-energy result. The selected tautomer
differs from the configuration's original reference drawing in 369 molecules.
Stage 3 now enumerates from every equivalent reference-node seed, deduplicates
the candidates, and applies one limit of 16 per protonation configuration.
This resolves 10,719 of the 14,585 configurations previously flagged by the
four-per-node source cap: 33,160 configurations are complete and 3,866 remain
truncated, including 194 dominant configurations. There are 16,893 tied top
rule scores; the increase reflects newly exposed alternatives rather than a
loss of protonation-state certainty. These cases remain explicitly flagged.

Stage 2 pKa intervals are propagated through the Stage 3 state-energy model.
At the current conservative calibration, 3,607 site-state calls remain on one
side of 50% protonation over their full interval, while 6,157 intervals cross
the decision boundary. The
latter retain zero interval-robust confidence as a separate diagnostic. Stage 3
also calibrates the probability that the scalar observed pKa would support the
predicted acid/base-side call at the requested pH, using hierarchical
exact-label/family/global shrinkage of scaffold-held-out signed residuals.
Overall confidence incorporates that empirical evidence, population
decisiveness, exact-label applicability, free-energy reconstruction error, and
pair-coupling identifiability. The empirical value is a pKa-side proxy, not
validation against measured site-state populations.
The median empirical pKa-side call confidence is 0.956 over all 9,764 sites.
There are 132 sites in 112 molecules below 50% support; these are explicit
scalar-macro-pKa versus coupled-state conflicts requiring review of the
macro-step/site mapping or prior-dependent pair closure.

## Rebuild order

```bash
python scripts/build_release_microstate_dataset.py
# Refresh the network topology/anchors with the currently installed Stage 1
# bundle before extracting the new Stage 1 training table.
python scripts/build_molecule_microstate_network_dataset.py \
  --allow-stage1-detector-mismatch
python scripts/stage1_train_intrinsic_pka.py \
  --network-dataset data/processed/pka_molecule_microstate_network_dataset.csv \
  --out-dir data/processed/ml_models_experimental_only/stage1_intrinsic
# Re-score every network site with the newly fitted Stage 1 bundle.
python scripts/build_molecule_microstate_network_dataset.py
python scripts/stage2_train_network_context.py
python scripts/run_canonical_protonation_pipeline.py --ph 7.4
python scripts/visualize_canonical_stage3_failures.py
python scripts/audit_pka_evidence_quality.py
```

The visualization command writes a browsable canonical Stage 3 review index to
`data/processed/stage3_failure_visualization_canonical/index.html`, alongside
PNG galleries and complete CSV queues named `heldout_pka_errors`,
`secondary_transition_assignment_guesses`, `deployment_view_of_heldout_errors`,
`low_confidence_state_calls_at_ph_7_4`,
`empirical_pka_side_conflicts`,
`stage2_free_energy_reconstruction_gaps`, `pair_coupling_changed_site_calls`, and
`incomplete_microstate_networks`. The deployment panel keeps the held-out
failure cases but shows the representative input drawing beside Stage 3's
dominant predicted microstate. It explicitly labels the input drawing as not
being an experimental state observation and compares the Stage 3 site form to
the form implied by the experimental pKa at pH 7.4. That comparison is only an
approximation for coupled multisite systems. The HTML section contains the
full table; the paired composite PNG is limited to the top examples.

The secondary-transition queue is a post-hoc curation aid for held-out errors
above 3 pKa units. It ranks other enumerated transitions by their blind Stage 1
network pKa and reports an alternative only when it is at least 1 pKa unit
better and within 2 pKa units of experiment. If the alternative already owns
an exact anchor, the queue evaluates the two labels as a reciprocal swap. The
experimental value is used only to create this review queue: the primary label
is never changed automatically and the suggestion never enters training or
deployment.

The canonical runner executes Stage 2 and Stage 3 together and writes
`canonical_pipeline_run_manifest.json`, recording the exact network, model,
intermediate, final-output, and schema hashes used in that run. Stage 3 also
writes `stage3_output_schema.json`, `stage3_ranked_tautomers.csv`, and the
reproducible `stage3_empirical_call_calibration.json` artifact. In the flat site table,
`stage2_predicted_macro_pka` is the scalar comparable to experimental pKa;
`stage3_one_body_pka` is an energy-model coefficient; and
`stage3_contextual_edge_pka_at_dominant_background` is the microscopic
transition pKa with the other coordinates fixed to the predicted dominant
configuration. The older `stage3_effective_local_pka` name remains only as a
deprecated alias of the one-body coefficient.

For a Stage-3-only code change, the canonical runner accepts
`--reuse-stage2-output`. Reuse is refused unless the existing Stage 2 CSV,
network dataset, and Stage 2 model hashes exactly match the prior canonical
manifest; the manifest records that reuse occurred.
