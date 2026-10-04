# Inputs required for full-pKa-range microstate enumeration

The default active scope is aqueous pKa 0–14. Structural descriptors and
permanent formal charges are retained outside this range, but their extreme-
range proton-transfer edges are not silently activated.

To extend the model, complete one row per distinct charge transition in
`data/curation/full_pka_range_transition_inputs.csv`. The required inputs are:

You do not need to edit that CSV manually. Start the guided local feeder:

```bash
python scripts/full_pka_range_information_feeder.py
```

It opens `http://127.0.0.1:8771`, presents one plain-language question at a
time, autosaves after every answer, and resumes where you stopped. Choosing
"I don't know—ask Codex to research this" is a valid submitted response. The
answers are written separately to
`data/curation/full_pka_range_information_responses.csv`. That response log is
never rewritten. After validation, approved values are copied into the
machine-readable transition table and only those rows are set `enable=true`.

The currently approved additions are the 1,2,3-triazole neutral-to-anion edge,
both 1,2,4-triazole edges, and both indazole edges. All other feeder candidates
remain disabled. Searchable literature metadata was completed during code
validation; the curator is asked only for intended scope or genuinely
ambiguous chemical conventions.

1. The lower and upper pKa boundary to support. "All pKa" is not a single
   aqueous condition: solvent, temperature and activity conventions must be stated.
2. Solvent or solvent mixture, temperature and (where available) ionic
   strength for each reference value.
3. Whether each value is a macroscopic charge-level pKa or an atom-specific
   microscopic pKa. Molecule-level values must not be assigned to one atom.
4. A transition definition specifying acid/base charge, the atoms allowed to
   gain or lose H, and whether the chemistry is protonation, N-H/O-H/S-H loss,
   or carbon deprotonation such as enolate formation.
5. All allowed tautomers for both charge levels. If a macroscopic pKa links two
   ensembles, relative tautomer populations or free energies are needed to
   infer microscopic pKas; otherwise the output remains an explicitly
   underdetermined ensemble.
6. A traceable source citation, uncertainty and applicability domain for each
   anchor. Separate rows are required when ring substitution materially shifts
   the reference distribution.
7. A policy for salts, counterions, metal coordination and covalent adducts.
   These can change the relevant transition and cannot be inferred from a bare
   pKa number.

Amphoteric rings require more than one row. For example, neutral triazole is
the base member of cation→neutral protonation and the acid member of
neutral→anion deprotonation. Tetrazole N-H positions are not manually selected:
the enumerator generates all sanitizable atom-mapped tautomers, and a measured
macroscopic pKa constrains the transition between the two charge ensembles.

Formal `N+–O-` also requires context. Pyridine/amine N-oxide O-protonation is
implemented in the aqueous model. Nitro `R-N+(=O)O-` remains a resonance-form
descriptor; nitroalkane alpha-C-H loss would be a separate carbon-acidity row.
