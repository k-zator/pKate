# Experimental pKa microstate dataset

This transition-level file is now the normalized input to the molecule-level
network dataset documented in `README_molecule_microstate_network_dataset.md`.

`data/processed/pka_microstate_dataset.csv` is the release-oriented dataset.
It has one row per enumerable experimental pKa transition and contains complete,
atom-mapped molecular graphs for the conjugate acid and conjugate base, plus
charge-preserving RDKit tautomer ensembles. Each row also has an evidence tier:
`gold` or `silver` may supervise an exact site, while `ambiguous` is retained
only as a weak molecule-level label. The current release schema is `1.8.0`.

## Inclusion contract

A row is included only when all of the following are true:

1. The measurement method is `experimental`. Marvin and Epik values are not
   read as labels or used to choose a site.
2. The raw SDF record is readable.
3. The measurement is not covered by an audited assay/record exclusion in
   `data/curation/pka_measurement_exclusions.csv`.
4. The measurement has not been manually quarantined by the Stage 1 outlier
   review in `data/curation/stage1_outlier_review_decisions.csv`. Exact
   reviewed site assignments in that file take precedence over the older
   label-only override tables.
5. An unannotated amine-family assignment does not exceed the conservative
   pKa 14 plausibility guard; larger molecule-level values are quarantined
   because they may describe another acidic site.
6. The drawn transition has either one structurally unique supported pair site
   or a manual group override that can be revalidated on the current raw
   structure. This establishes a structural mapping, not necessarily that the
   experimental source identified that site.
7. A protonation center can be identified and both conjugates sanitize.
8. The conjugate acid has total formal charge exactly one unit above the
   conjugate base.

Rows that fail enumeration are written to
`data/processed/pka_microstate_dataset_quarantine.csv`. The original PART1 and
PART2 override files remain source review artifacts. Only overrides validated
against current records are copied to
`data/processed/site_resolution_overrides.csv`; rejected and stale override
rows are in `data/processed/site_resolution_overrides_quarantine.csv`.

An enumerable row is not automatically exact training truth. Evidence policy
`1.1.0` separately scores structural mapping, experimental site attribution,
transition identity, and measurement conditions. Potential ionizable contexts
without a complete conjugate-pair enumerator (for example thioamides and
sulfonamides) prevent the claim that an otherwise unique enumerable site must
own the scalar pKa. Reference-scale conflicts and source drawings whose formal
charge changed during transformation are also retained as `ambiguous`, not
silently assigned to a local site.

## Important columns

- `experimental_pka`: experimental transition pKa.
- `site_label`, `site_family`: selected conjugate-pair chemotype.
- `site_detector_label`: structural SMARTS label before expansion into a
  transition-specific chemotype.
- `site_coupling_group`, `site_acid_level`, `site_base_level`: shared
  protonation coordinate and the two charge-adjacent levels for this edge.
- `site_atom_maps_json`: all mapped atoms in the matched site.
- `protonation_center_atom_maps_json`: atom or atoms at which the enumerated
  proton is added or removed.
- `override_atom_index_raw`: optional reviewer-selected zero-based atom index
  used to distinguish repeated instances of the same functional-group label.
  It records a curation decision and remains separate from `atom_index_raw`,
  which describes atom-level evidence supplied by the original source. The
  site-resolution reviewer fills this field when a numbered candidate is
  selected.
- `reference_acid_atom_mapped_smiles`,
  `reference_base_atom_mapped_smiles`: deterministic representatives.
- `acid_microstates_json`, `base_microstates_json`: JSON arrays of mapped
  tautomeric microstates for the two protonation levels.
- `site_assignment_basis`, `structural_site_assignment_confidence`: auditable
  structural site-assignment rubric. The legacy `site_assignment_confidence`
  column is retained as an alias for compatibility.
- `source_site_evidence`, `source_site_evidence_confidence`: evidence actually
  present in the experimental record. Structural uniqueness and a manual
  override do not become source-level atom annotations.
- `source_assay_id`, `source_document_id`, `source_molecule_id`: original
  source identifiers carried into the release for assay-level audits.
- `evidence_tier`, `evidence_tier_reason`: `gold`, `silver`, or `ambiguous`
  supervision policy and its machine-readable reason.
- `experimental_site_attribution_confidence`,
  `transition_identity_confidence`, `measurement_conditions_confidence`:
  distinct evidence axes. Confirming highlighted atoms does not promote the
  other axes.
- `reference_pka`, `reference_uncertainty`, `reference_z_distance`, and
  `reference_transition_definition`: internal chemistry plausibility prior and
  review signal. It is not an external test label.
- `all_resolved_detector_labels_json` and
  `unresolved_ionizable_contexts_json`: all detected chemistry, including
  plausible ionizable contexts not yet fully enumerable.
- `source_original_smiles`, `raw_structure_smiles`,
  `standardized_structure_smiles`, and `structure_provenance`: separate source,
  supplied-drawing, canonicalized-drawing, and transformation provenance.
- `outlier_review_decision`, `outlier_review_note`, `outlier_reviewed_at`:
  persistent manual review provenance when a row has been confirmed or
  reassigned. Manually rejected rows carry the same fields in quarantine.
- `acid_fraction_at_ph`, `base_fraction_at_ph`: Henderson-Hasselbalch
  fractions conditional on this isolated transition.
- `acid_tautomer_enumeration_truncated`,
  `base_tautomer_enumeration_truncated`: true when the configured tautomer cap
  was reached.
- `other_pair_sites_held_as_drawn`: number of non-target ionizable sites whose
  input protonation was not changed for this transition.

Atom maps are stable one-based identifiers derived from the raw molecule. All
heavy atoms retain the same maps in every generated state. Explicit hydrogen
atoms, if present in a source SDF, are not assigned persistent maps.

### Heteroaromatic conventions

- `quinoline` and `isoquinoline` are supported base forms. Their conjugate
  acids are `quinolinium` and `isoquinolinium`, formed at the pyridine-like
  ring nitrogen.
- `pyrrolium` denotes the thermodynamic C2/C5-protonated conjugate acid of
  pyrrole. The serialized non-aromatic resonance form places the formal
  positive charge on nitrogen, but the added proton and
  `protonation_center_atom_maps_json` are on an alpha carbon.
- N-protonated pyrrole is detected separately as `pyrrolium_n`; it is not
  assigned the textbook C2-protonation pKaH and is not currently a calibrated
  conjugate-pair target.

### Functional-group detector 2.5

Detector schema `2.0.0` adds explicit one-proton conjugate families for:

- phosphate, phosphonate, phosphinate and phosphoramidate P-OH/P-O- sites;
- primary, secondary and tertiary alcohol/alkoxide sites, including methanol
  and saturated carbons bearing heteroatom substituents;
- phenol/phenolate oxygen sites protected from generic ether assignment.

Each proton-bearing phosphorus oxygen is a separate site, so partially
deprotonated polyprotic phosphate species can contain both P-OH and P-O- sites.
`phosphate_ester` is retained as a non-ionizing C-O-P context descriptor.
Likewise, `nitro` recognizes both charge-separated `N+(=O)O-` and neutral
`N(=O)=O` encodings, but is deliberately not a protonation site: ordinary
nitro groups do not receive a fabricated aqueous pKa. Charge-aware amine and
aniline SMARTS prevent the nitro nitrogen from being mislabeled as a base.

Schema `2.1.0` also recognizes hydrazine/hydrazinium,
hydroxylamine/hydroxylammonium, ammonium substitution states, oxonium,
protonated thiol, and neutral or charge-separated sulfoxide drawings. The
transition dataset records `functional_group_detector_schema_version`; Stage 1
training schema `3.1.0` propagates detector and evidence-policy provenance from
the molecule-network input.

Schema `2.2.0` separates structural recognition from aqueous ionization policy.
It adds mapped tetrazole/tetrazolate and aromatic/aliphatic N-oxide pairs,
pyridazine, benzimidazole, oxazole/isoxazole and thiazole/isothiazole conjugate
ensembles, common oxadiazole/thiadiazole descriptors, and permanent quaternary
ammonium. Any nitrogen not covered by a specific rule receives an explicit
`unclassified_*_N` fallback and is reported as an unresolved ionizable context;
it cannot silently become training truth. The active model scope is aqueous pKa
0–14. See `README_full_pka_range.md` and the accompanying curation template for
the information required to activate extreme-range or amphoteric transitions.

Schema `2.3.0` adds charge-aware 1,2,3-triazole, 1,2,4-triazole and indazole
forms. The neutral 1,2,4-triazole and indazole detector sites expand into two
separate labels (pKaH and N-H acidity) while sharing one serial protonation
coordinate. The approved transition values and literature provenance live in
`full_pka_range_transition_inputs.csv` and
`stage1_reference_pka_priors.csv`; they are reference priors, not synthetic
experimental rows.

Schema `2.4.0` applies the same serial-coordinate representation to pyrazole
and imidazole. Neutral forms now expose distinct cation-to-neutral basicity
and neutral-to-anion N-H-acidity edges; their charged and anionic drawings
expose only the adjacent edge. A historical site-only review of either neutral
ring is no longer accepted as transition-level evidence. The review must name
the transition-specific label (for example `imidazole_acidity`) or provide an
acidic/basic measurement type. Untyped pKa-based edge selection remains
provisional and is ineligible for exact Stage 1 supervision.

Schema `2.5.0` separates three-membered-ring aziridine/aziridinium from generic
secondary amine/ammonium. The pair remains the ordinary `+1 -> 0` conjugate-acid
transition, but uses an aziridine-specific aqueous reference because ring strain
makes it substantially less basic than an acyclic secondary amine. The raw
2,2-dimethylaziridine pKa 5.37 record is retained as weak conflicting evidence;
it is not an exact Stage 1 target.

## Scope and limitations

Each SMILES is a complete molecular graph, but this schema represents a local
pKa transition. In polyprotic molecules, non-target sites are held in their raw
input forms. Therefore the pH fractions are an isolated-site approximation,
not a coupled macroscopic microstate distribution. Tautomers are rule-based
RDKit enumerations and are not energy-ranked. Rows with multiple plausible
protonation centers retain the alternatives as an ensemble and receive medium
site-assignment confidence.

The 2026-09-15 evidence-audited snapshot contains 8,453 enumerable experimental
transitions: 4,228 `silver` exact-site candidates and 4,225 `ambiguous` weak
labels. There are no source-verified `gold` rows in the current raw metadata;
that absence is reported explicitly. The audit finds 2,885 rows with an
unenumerated competing context, 540 with a changed source/dataset formal charge,
and 453 copied-lineage groups containing 1,200 rows. Copied values count as one
independent lineage in Stage 1 rather than as independent replication.

## Manual review of Stage 1 outliers

The review is driven by grouped-scaffold out-of-fold predictions, not fitted
training predictions. By default every Stage 1 error greater than 3 pKa units
is expanded back to its original experimental SDF measurement. This prevents
an aggregated training target from hiding one bad source record.

Generate the queue, all atom-indexed structure images, and the static visual
index:

```bash
python scripts/stage1_outlier_review.py --export-only --render-all --no-open
```

Review the pending measurements in the local browser interface:

```bash
python scripts/stage1_outlier_review.py --web
```

Then open `http://127.0.0.1:8765`. The page saves each decision immediately,
advances to the next unresolved measurement, displays a persistent progress
counter, and resumes safely after a restart. It uses only the Python standard
library and binds to loopback by default. The terminal reviewer remains
available with `python scripts/stage1_outlier_review.py`.

The reviewer displays the experimental and predicted pKa, raw provenance,
current site, all detected candidate sites, and zero-based atom indices. The
web reviewer has separate attestations for source-site attribution, transition
identity, and measurement conditions. For an `ambiguous` record, confirming the
structural selection requires explicit source-site and transition attestations;
the old structural-only confirmation remains pending for evidence purposes. Its
commands are:

- `c [note]`: confirm the current exact site and experimental value.
- `r <candidate-index> [note]`: reassign to that exact detected atom set.
- `w [note]`: site is wrong and the correct site is unknown; quarantine.
- `e [note]`: wrong or non-molecular endpoint; quarantine.
- `v [note]`: implausible/incorrect experimental pKa; quarantine.
- `u <label> <atom-index>`: required site chemistry is unsupported; quarantine
  for a later SMARTS/microstate-rule addition.
- `d [note]`: defer and keep the record in later review sessions.
- `s`: skip without recording a decision; `q`: save and quit.

Every decision is saved immediately to
`data/curation/stage1_outlier_review_decisions.csv`, keyed to the original SDF
record and measurement. Re-running the reviewer skips completed decisions.
The generated queue is
`data/processed/dataset_audit/stage1_outlier_review_queue.csv`; the browsable
image index is
`data/processed/dataset_audit/stage1_outlier_review_index.html`.

`data/processed/dataset_audit/stage1_evidence_quality_review_queue.csv` is a
complete evidence backlog, not a requirement to review every ambiguous row.
Ambiguous rows are already prevented from exact-site training. Prioritize
reference conflicts, altered charge-state drawings, and chemistry important to
the intended deployment domain.

Generate the machine-readable evidence audit with:

```bash
python scripts/audit_pka_evidence_quality.py
```

Its summary, label/tier table, plausibility-review table, and copied-lineage
report are written under `data/processed/dataset_audit/evidence_quality/`.

After reviewing, run the full rebuild order in
`README_molecule_microstate_network_dataset.md`. The release builder applies
the decisions before training data are reconstructed. A later model may expose
new errors over the threshold, but already reviewed source measurements retain
their decisions and are not presented again unless `--include-reviewed` is
requested.

The separate pKa–structure consistency pass is launched with
`python scripts/pka_structure_consistency_review.py`. It stores non-destructive
structure/transition decisions in
`data/curation/pka_structure_consistency_review_decisions.csv`. On rebuild,
discard decisions enter quarantine, validated same-connectivity replacement
SMILES replace only the curated input overlay, and requested transition/site
corrections remain quarantined until their conjugate-pair chemistry is
implemented.

When a detector upgrade makes an old review newly actionable, run:

```bash
python scripts/stage1_outlier_review.py --reopen-detector-gaps --export-only --render-all --no-open
```

Only invalidated exact assignments and previously rejected measurements with
a newly supported conjugate site are reopened. Superseded decisions are kept
in `data/curation/stage1_outlier_review_decision_history.csv`, and the affected
records are listed in
`data/processed/dataset_audit/stage1_detector_change_rereview.csv`.

## Rebuild

From the repository root, with the project RDKit environment active:

```bash
python scripts/build_release_microstate_dataset.py
```

The defaults use `functional_group_assignments.csv`, both manual override
parts, pH 7.4, an overlap threshold of 0.5, and at most 16 tautomers per
protonation level. See `--help` for output and enumeration controls.
