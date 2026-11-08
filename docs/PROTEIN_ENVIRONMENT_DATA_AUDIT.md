# Protein-environment pKa data audit

Audit date: 2026-10-26

## Decision

The available public data are sufficient to build and validate a **first,
explicitly conditional protein-environment extension** to pKate. They are not
sufficient to train a general, experimentally grounded predictor of bound
small-molecule pKa values.

The defensible first model is therefore:

1. retain the present pKate aqueous ligand prediction as the reference state;
2. calculate a structure-dependent environmental free-energy correction from
   a supplied protein-ligand complex;
3. calibrate and test the environmental representation on experimental
   protein-residue pKa shifts;
4. expose transfer to ligand sites as a separate, lower-confidence evidence
   tier; and
5. include nearby titratable protein residues as explicit joint Stage 3 nodes
   when the local binary network remains tractable, while labelling their
   screened-Coulomb pair terms as heuristic rather than learned evidence.

It must not describe PKAD-R or PKAD-2 as a bound-ligand pKa training set. Their
experimental labels belong predominantly to protein side chains, not to
drug-like ligand sites.

## Material acquired locally

The files are under the ignored directory
`data/raw/external_datasets/protein_environment/`; they are intentionally not
committed to Git.

| Resource | Local content | Source snapshot | Purpose |
|---|---|---|---|
| PKAD-R | official JSON, no-PDB CSV, page snapshot, and 188 RCSB mmCIF files | `PKAD-R-250211.json`, downloaded 2026-10-19 | primary curated experimental labels |
| revised PKAD-2 | four workbooks, raw/fixed structures, scripts, sequences, and published splits | `xu-shi-jie/revPKAD2` commit `39a10ec44aebedd7ea3091cf394cfa94889800ca` | provenance comparison and failure audit |
| pKALM | processed PKAD-2 workbook, sequence datasets, README, and license | commit `1ef612114cb9009fd64221cba947a3987778db6a` | independent processed-table and licensing check |
| PKAD-3/KaMLs | 20 public train/test table pairs, README, and 172 checksum-recorded RCSB structures for de-overlapped split-0 candidates | commit `04c8499121de5647d8d69a97849df7de2d245fbf` | experimental augmentation candidate behind a PKAD-R-only gate |
| PLBD | CC BY 4.0 JSON archive with compound/protein pKₐ and protonation-corrected intrinsic-binding tables | Zenodo record `7482058` | proton-linkage consistency evidence; never a direct bound-ligand pKₐ label |
| direct free/bound ligand registry | 15 curated literature cases and seven checksum-pinned exact complex structures | curated 2026-10-31 | direct ligand calibration candidate behind count/diversity and grouped-OOF gates |
| 2026 metalloprotein SI | author coordinate archive and calculated-pKₐ PDF | DOI `10.1021/acs.jctc.6c00884` SI | specialist locked benchmark and external-method comparison only |

Important hashes:

| File | SHA-256 |
|---|---|
| PKAD-R JSON | `9ee3cfb885306ca944e177c9b21fa700dd19b8cc985e4bdb08f42513ae269937` |
| PKAD-R no-PDB CSV | `92e7d0ae4608fc8317b9883ff64c096fd3b9745d140cc00bea83b26244ff3279` |
| original PKAD-2 workbook | `f04495bc23d1e31b9e37a2e81bcaa895951864a2b64ea2095418f1a50feaf8a0` |
| processed revised PKAD-2 workbook | `4194628cedb1f481cad5e6fcbcab0426006a8ea955626251a05980cb68134e77` |
| PKAD-R structure checksum manifest | `b6bb6af6702650d3fd0f7dda1efe2f78e0303e3000d2fe48d1a684661dc522e3` |
| PLBD `main.zip` | `48e120d05ef55b34b6efc5bb3cf30091ecb4a76fe4786fdabbbde30ba3cd959e` |
| direct ligand registry | `af300ca5c7aebf630d5aae9b838b282a530960094b20719ae01839755ecbf2d0` |
| metalloprotein coordinate SI | `c32a0c2047ddd87f25c365975bd87f1fe0bbeaf540017132aab2c3ab7a6efae6` |
| metalloprotein calculated-pKₐ SI | `e128ff1fe63e99fc1c2479840adec7cc7a5a79bc1c62c37d7d8a6a2a912a4d7e` |

The RCSB coordinate snapshots are CC0. The PKAD-R website does not state an
explicit data license in the downloaded material, so redistribution of the
label table or a derivative data bundle should wait for written clarification.
The pKALM repository applies CC BY-NC-SA 4.0. The KaMLs repository snapshot has
no `LICENSE` file. These resources can be inspected locally, but their labels
must not silently become a redistributable pKate training asset.

## Quality findings

### PKAD-R: recommended label source

PKAD-R has 1,024 rows over 188 PDB entries:

- 778 `Main` rows;
- 116 alternative pKa measurements;
- 76 alternatives measured in mutants; and
- 54 alternatives measured in another protein state.

Its strongest feature is not merely its size. It preserves alternatives and
warnings rather than flattening them into apparent independent ground truth.
After requiring `Main`, an exact numeric pKa, no warning, and one of the six
side-chain classes, 651 rows remain:

| Site class | Strict rows before coordinate validation |
|---|---:|
| Asp | 193 |
| Glu | 215 |
| His | 106 |
| Lys | 82 |
| Cys | 36 |
| Tyr | 19 |

Coordinate validation against freshly downloaded mmCIF files found 25 missing
targets among all 778 `Main` rows. Twenty-two were correctly marked
`ResID NOT exist`; three warning-free Cys rows were also absent from their
specified structure (`1LJU A 10`, `4QYT A 111`, and `4XLL A 104`). The strict
structure-mapped set is therefore **648 rows**, not 651.

Five of those 648 mapped residues lack the complete terminal ionization-center
atoms needed for deterministic descriptors (four Lys side chains in `1XSN`
and one Asp in `3FT7`). The fitted correction artifact uses **643 rows** plus
990 de-overlapped PKAD-3 rows at a capped effective weight of 160.75. After
extending validation components with PKAD-3 UniProt links, the main rows occupy
84 groups. Homology-grouped out-of-fold validation gives MAE 0.870 pKₐ unit
and a p90 absolute error of 2.174 (finite-sample 90% calibration radius 2.390),
versus MAE 0.987 for applying no environmental shift and 0.960 for a
fold-fitted chemistry-only median. These modest results support a
low-confidence correction layer, not a claim of solved protein pKa prediction.

PKAD-R's alternatives are represented explicitly without pretending that they
are independent sites. Ninety-five warning-free mutant/state alternatives
have a distinct experimental PDB structure and a resolvable target site: 63
mutant pairs and 32 state pairs. Each alternative remains in its main site's
homology group, and the total alternative weight for one main site is capped at
0.5. The augmentation candidate improves canonical MAE from 0.9023 to 0.8945
but worsens paired-delta MAE from 0.3978 to 0.4086, so it is **not deployed**.
The 112 usable alternative measurements instead provide a measurement-noise
summary (median absolute repeat difference 0.105; p90 0.509) and are not added
as duplicate training observations.

The calibration now preserves all 648 strict sites in one ledger. It reports
643 grouped OOF predictions and five unavailable descriptor rows rather than
imputing a false site center. Homology-group bootstrapping gives the following
representation decisions:

| Chemistry | Rows | OOF MAE | Zero MAE | Chemistry-only MAE | Bootstrap p05 gain vs zero / chemistry | Decision |
|---|---:|---:|---:|---:|---:|---|
| carboxyl | 407 | 0.738 | 0.881 | 0.867 | +0.050 / +0.027 | validated, scale 1.0 |
| imidazole | 106 | 0.806 | 0.862 | 0.850 | -0.006 / -0.027 | provisional, scale 0.5 |
| amine | 78 | 1.135 | 1.040 | 1.176 | -0.232 / -0.126 | correction withheld |
| thiol | 33 | 1.940 | 2.675 | 2.005 | +0.227 / -0.317 | provisional, scale 0.5 |
| phenol | 19 | 1.114 | 0.816 | 0.858 | -0.451 / -0.497 | correction withheld |

The gate compares each chemistry with both applying no protein correction and
a fold-fitted residue-chemistry median. The second control tests whether the
spatial representation adds information beyond merely learning a residue-class
average. The gate prevents a representation that failed grouped validation
from degrading the existing aqueous model. Ligand-analogue shrinkage is
applied only after this protein-residue calibration gate. These are
cross-fitted development-set
calibration results, not a locked external test and not validation on bound
drug-like ligand pKas.

Experimental metadata remain incomplete even in that strict subset:

- uncertainty missing for 22.4%;
- salt concentration missing for 24.4%;
- experimental pH range missing for 9.1%;
- temperature missing for 6.0%; and
- reference missing for 2.0%.

The 188 coordinate snapshots comprise 161 X-ray, 24 solution-NMR, two
electron-microscopy, and one unclassified structures. The 161 structures with
reported resolution have a median of 1.90 A and a range of 0.93--3.60 A.

Among `Main` sites, 89 are within 6 A of a non-water heterogen, 53 are within
6 A of an organic heterogen, 16 are within 4 A of a metal, and 523 are within
4 A of a crystallographic water. These counts show that useful nonuniform
environments are present, but do **not** establish that the pKa experiment was
performed in the exact ligand-bound or metallated state represented by the
deposited structure.

### Revised PKAD-2: useful provenance, unsafe default training table

The processed revised workbook contains 1,712 rows (1,450 wild type and 262
mutant), but only 1,609 pKa fields are directly numeric. The other 103 are
censored, approximate, or multi-valued strings. It also contains:

- 31 repeated site-key groups covering 70 rows;
- 15 repeated sites with conflicting numeric pKas;
- a maximum repeated-site spread of 1.99 pKa units; and
- 17.1% missing SASA values.

All table contexts have a corresponding fixed structure and all stated target
residues map in those fixed structures. However, those structures were made by
deleting all heterogens and waters and adding hydrogens at pH 7. They are not
valid environmental ground truth for a binding-site model. The original PDB
files retain the relevant context, but 11 wild-type rows fail direct residue
mapping there.

The published split must not be reused:

- all 75 validation rows also occur in training;
- train and test share 54 PDB IDs and 55 protein/mutant environments; and
- validation and test share 40 PDB IDs/environments.

A pKate model needs connected-component splitting by protein sequence identity,
with all alternative structures, measurements, states, and mutants for the
same biological site kept on one side of the split.

### PKAD-3/KaMLs: implemented augmentation, rejected by the locked gate

One public split reconstructs 1,707 structure rows over 372 PDB entries, but
those rows represent only 879 unique biological site identifiers because the
same experimental pKa is paired with alternative structures. In split 0,
training and test share 90 PDB IDs and 11 biological identifiers. Several
metal-distance feature columns are the constant missing-value sentinel `999`
for every row.

The integration uses only split 0's experimental `Expt_pKa` and structural
identifiers. It explicitly ignores `model_pka` and every published computed
descriptor. Removing 610 exact PKAD-R structure-sites and 103 rows without a
four-character experimental PDB leaves 994 structure candidates representing
656 measurement groups. Fresh RCSB coordinates resolve 990 rows; four have
incomplete terminal side-chain atoms.

Alternative structures for one measurement share half a canonical row of raw
weight, and total PKAD-3 weight is capped at 25% of the canonical cohort.
PKAD-R sequence links and PKAD-3 UniProt contexts are collapsed transitively
before folding. At the maximum cap, PKAD-3 improves canonical pooled OOF MAE
from 0.9023 to 0.8701, RMSE from 1.3948 to 1.3266, p90 error from 2.3551 to
2.1740, and equal-homology-group MAE from 1.1516 to 1.1137. The candidate is
therefore **deployed**. Its 990 rows enter with total effective weight 160.75,
exactly 25% of the 643 canonical rows. Recomputed features, exclusions, hashes,
and gate metrics remain auditable beside the model.

The published split files carry no row-level references, so exact-reference
de-overlap cannot be proven. That limitation is another reason not to weaken
the gate.

### CASP16 complexes: useful ligand evidence, not bound-state supervision

The CASP16 pharmaceutical-ligand release contributes 72 numeric experimental
pKa values across 63 ligands. Only the chymase/cathepsin table contains pKa
columns; the accompanying autotaxin file contains none despite the inventory
description. All ADMET Predictor fields are ignored. Acid/base class uniquely
selects 32 supported single transitions. In the current joint supplemental
fit, those rows contribute effective Stage 1 weight 5.40 alongside 2,252
canonical rows. Plain indole is treated as an extreme-range context rather
than a basic site; all three multisite molecules therefore have complete
four-state networks and all five macro values reach the Stage 2 gate. The
gate rejects them because exact-site scaffold MAE worsens from 0.9307 to
0.9444, so they contribute zero deployed Stage 2 weight.

Eighteen associated RCSB complexes (`7H60`--`7H6I`, with the deposited gaps)
are checksum-recorded locally. They preserve mapping provenance only: the
release does not say that the ligand pKa values were measured in the bound
complex, so they are ineligible for protein-environment fitting.

### PLBD: linkage constraints, not bound-ligand pKₐ labels

The checksum-pinned CC BY 4.0 PLBD archive contains 226 compound-pKₐ rows
(225 experimental), 50 protein-pKₐ rows, 1,661 intrinsic-parameter rows, and
127 structure records. Joining the tables yields 1,649 proton-linkage
constraints; 12 intrinsic rows cannot be joined cleanly and are reported.

PLBD's intrinsic binding parameters are derived after correcting observed
binding for linked protonation. They constrain thermodynamic consistency, but
they do not report which atom in a bound ligand changed protonation or a
site-resolved bound-versus-free ligand pKₐ. The preparer therefore marks every
row ineligible for direct environment fitting and reports
`direct_bound_ligand_calibration_rows: 0`.

### Direct free/bound ligand evidence: implemented gate, still insufficient

The committed registry contains 15 experimental literature cases. Seven have
an exact point label, exact protein and ligand identity, exact complex
coordinates, and a mapped site atom: methotrexate and trimethoprim in the
*L. casei* DHFR group, plus glutathione bound to human GSTA1, GSTA3, GSTM1,
GSTM2, and GSTP1. The GST rows use the glutathione sulfur atom and
checksum-pinned structures `1PKW`, `1TDI`, `1XW6`, `1XW5`, and `5GSS`.
The other eight rows are approximate or one-sided labels, lack exact
coordinates, or use a homologous/nonidentical-ligand structure and remain
excluded from fitting.

The deployment rule requires at least 12 strict rows across six protein groups
and three chemistries, followed by grouped OOF improvement over a zero-shift
baseline both pooled and group-balanced. Homologous GST isoforms are grouped
by Alpha or Mu class rather than counted as independent proteins. The present
seven rows therefore span four protein groups and two chemistries. Consequently
the count/diversity gate is still closed, no direct model is serialized, and
the main protein bundle records seven candidates but zero deployed
direct-ligand rows. If a future gate passes, the direct model takes precedence
over residue-analogue transfer for ligand sites; artifact schema, feature
columns, and row counts are validated at load time.

### 2026 metalloprotein panel: evaluation only

Twelve Table 1 transitions from DOI `10.1021/acs.jctc.6c00884` are committed as
a locked specialist benchmark. They cover metal-bound water, alcohol, serine,
and hydroxide chemistry; eight labels are exact, three approximate, and one is
an upper bound. The present six-residue-class model cannot claim calibrated
coverage of those transitions. The panel is therefore explicitly experimental,
evaluation-only, and ineligible for training; bounds are never treated as
point targets. The author SI is acquired separately under CC BY-NC 4.0 and is
not committed. Its published external QM predictions score exact-label MAE
1.525 over eight cases and approximate-label MAE 4.20 over three; the sole
upper bound is violated by 1.60 pKₐ units. These numbers describe the paper's
method, not pKate, and no calculated value enters training.

## What can reasonably be built now

### A useful first release

1. **Complex input and audit**
   - accept mmCIF/PDB plus an atom-mapped ligand and identify its chain/residue;
   - retain explicit waters, ions, cofactors, and metals;
   - report missing atoms, alternate locations, model number, resolution,
     mutation/sequence mismatch, ligand mapping, and pose provenance.

2. **Transparent environment descriptors**
   - burial/SASA and local heavy-atom density;
   - electrostatic potential or a documented charge-based proxy;
   - hydrogen-bond donors/acceptors and salt bridges;
   - water accessibility;
   - nearby titratable residues and site-site distances;
   - ligand, cofactor, and metal coordination;
   - per-feature ensemble variation when several structures/poses are given.

3. **Protein-residue residual model**
   - predict a delta from the aqueous/intrinsic reference rather than a new
     absolute pKa from scratch;
   - use the 643 descriptor-ready rows from the strict PKAD-R set;
   - use distinct-structure mutant/state alternatives as down-weighted paired
     perturbations, while treating repeat measurements only as noise evidence;
   - use homology-connected grouped splits and publish per-residue results;
   - compare against a simple null model and optional PROPKA/PB backends.

4. **Cautious transfer to ligand sites**
   - keep pKate Stage 1 as the free-ligand reference;
   - apply the structure/physics correction as a separate evidence layer;
   - increase uncertainty for functional groups outside the six residue-like
     chemistries, metals, buried waters, covalent ligands, and pose ambiguity;
   - never present the protein-residue fit as direct ligand supervision.

5. **Local joint Stage 3 network**
   - add nearby Asp/Glu/His/Lys/Cys/Tyr/termini as titratable nodes;
   - couple them to ligand nodes using cycle-consistent free-energy terms;
   - enumerate exactly only while the local state space is tractable, then use
     a declared approximation;
   - return conditional microstate populations, site marginals, and separate
     aqueous-model, environment, pose, and network confidence components.

This would be scientifically useful for hypothesis generation and structure
preparation. It would not yet justify a claim of generally accurate bound
ligand pKa prediction.

### What the current data cannot support

- a direct supervised model for arbitrary drug-like ligand sites;
- a well-calibrated bound-ligand error bar across chemistry space;
- reliable treatment of binding-induced conformational change from one pose;
- a learned description of explicit proton transfer through water networks;
- general metal-site, covalent-ligand, or redox-linked protonation chemistry;
- experimentally validated full-complex microstate populations; or
- causal attribution of a pKa shift to the ligand merely because a heterogen
  is close in a deposited structure.

The published experimental literature contains only a small validation-scale
collection. Aguilar *et al.* tabulate 13 experimental binding-induced pKa-shift
rows: seven exact values, six one-sided bounds, and five coupled-site targets.
These are valuable named challenge cases, not enough labels for machine
learning. They are now preserved in a committed evaluation-only manifest.

## Ideal data and the gap to it

The ideal dataset would contain matched apo and holo structural ensembles,
the bound pose, all relevant waters/metals/cofactors, solution conditions, and
site-resolved microscopic pKas for both ligand and protein in free and bound
states. It would cover repeated chemotypes across diverse pockets and repeated
pockets across diverse chemotypes, with uncertainty and negative/no-shift
examples. Thousands of such pairs would be desirable for a broadly learned
model; no public dataset of that form was found.

Until that exists, the most defensible route is a hybrid:

- experimental PKAD-R calibration for protein environmental shifts;
- pKate experimental aqueous training for ligand intrinsic terms;
- continuum-electrostatics or constant-pH calculations as explicitly labelled
  computational evidence;
- the very small experimental binding-shift collection as a locked external
  challenge set; and
- targeted new measurements or active learning for the ligand chemistries and
  pocket types where pKate is intended to be trusted.

## Implementation status and remaining gates

The local artifact completes strict row filtering, machine-readable exclusions,
homology-connected splitting, null-model comparison, provenance-rich model
serialization, explicit local protein nodes, a PKAD-2 overlap/portability
audit, a PKAD-3 augmentation gate, PLBD linkage preparation, a fail-closed
direct bound-ligand calibration path, and the mapped locked challenge panels.
The PKAD-2 result improves MAE from 0.676 to 0.496 over 72 mapped unique
measurements, but only
one row survives the conservative 0.80 near-duplicate proxy, so this is not an
independent external performance claim. The 13-case challenge maps completely,
but its seven exact shifts have MAE 2.941 pKₐ units and only three of six
censored bounds are satisfied. The remaining release/scientific gates are:

1. obtain written redistribution/model-training clarification for PKAD-R labels;
2. verify that each coordinate structure represents the experimental state,
   ligand/cofactor content, construct, and mutation closely enough for the
   proposed feature set; and
3. develop a stronger electrostatic/solvation and conformational representation,
   then evaluate it on additional experimental cases without tuning on this
   small locked challenge panel.
