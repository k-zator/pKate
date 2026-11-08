# Changelog

## 0.2.0 — 2026-11-08

### Added

- Public `pkate.predict()` / `calculate_pkas()` API and single-molecule CLI.
- Explicit Stage 1 intrinsic-transition, Stage 2 network-coupling, and Stage 3
  complete-microstate workflow with end-to-end provenance manifests.
- Atom-mapped protonation/tautomer enumeration, coupled site probabilities,
  applicability diagnostics, and uncertainty-aware confidence tiers.
- Optional protein-environment correction with nearby titratable protein sites,
  PKAD-R calibration, PKAD-2 portability evaluation, and PKAD-3 augmentation.
- Audited pKahub, CASP16, PLBD, metalloprotein, and direct bound-ligand data
  integrations with explicit training/evaluation eligibility boundaries.
- Interactive review, failure galleries, pipeline figures, release-model
  manifest, and clean-clone coverage workflow.

### Changed

- Stage 2 now emits a cycle-consistent one-body/pair free-energy model rather
  than an unconstrained contextual delta alone.
- Stage 3 consumes that energy model directly, reports macro/microscopic pKₐ
  quantities separately, and ranks tautomers only within the selected
  protonation configuration.
- Functional-group coverage and conjugate handling were expanded for phosphate,
  nitro/N-oxide, phenol/phenolate, alcohol/alkoxide, sulfur, hydrazine,
  ammonium, oxonium, thiol, and heterocyclic contexts.
- The fitted deployment runtime is pinned to scikit-learn 1.8.0.

### Validation

- 1,246 clean-clone tests pass.
- Coverage: 89.06% lines, 76.68% branches, 86.10% combined.
- Canonical Stage 3 accounts for all 20,613 molecular networks with zero
  incomplete population networks and zero free-energy optimizer failures.

### Known limitations

- The direct bound-ligand calibration gate remains closed: seven strict rows
  cover only four protein groups and two chemistries, below the preregistered
  12-row / six-group / three-chemistry minimum.
- The optional protein-environment model is not included in the aqueous release
  archive pending clarification of PKAD-R redistribution/model-release rights.
- The separate fitted aqueous model asset remains subject to the source-license
  review recorded in `models/MODEL_ARTIFACT_MANIFEST.json`.
