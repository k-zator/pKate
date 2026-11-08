#!/usr/bin/env python3
"""Audit which pKahub macroscopic pKa records can receive structural sites.

pKahub's published charge transitions are computational assignments rather
than experimental site labels.  This audit therefore ignores those labels and
asks only what pKate's open, deterministic functional-group rules can infer
from each unique input structure.  Multi-site molecules remain macroscopic
training candidates, but must not be converted into invented site labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem

from functional_group_pka_analysis import expand_pair_site_transitions
from substructure_match import find_sites_with_metadata, resolve_overlapping_sites


DEFAULT_INPUT = Path(
    "data/raw/external_datasets/pkahub/all_datasets.tsv"
)
DEFAULT_OUTPUT = Path(
    "data/processed/dataset_audit/pkahub_transition_assignment"
)
REQUIRED_COLUMNS = {
    "pka_value", "data_source", "molid", "smiles", "assignment_method"
}


def _molecule_assignment(molid: str, smiles: str) -> Dict[str, object]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {
            "molid": str(molid),
            "input_smiles": str(smiles),
            "canonical_smiles": "",
            "status": "invalid_smiles",
            "candidate_transition_count": 0,
            "resolved_site_count": 0,
            "unresolved_context_count": 0,
            "transition_families": "",
            "resolved_labels": "",
        }

    resolved, _ = resolve_overlapping_sites(find_sites_with_metadata(mol), 0.75)
    transitions = [
        transition
        for site in resolved
        for transition in expand_pair_site_transitions(site)
    ]
    unresolved = [
        site for site in resolved
        if str(site.get("type", "")) == "unresolved"
        or "quarantine" in str(site.get("ionization_role", ""))
    ]
    count = len(transitions)
    if count == 0:
        status = "no_supported_transition"
    elif count == 1 and unresolved:
        status = "single_transition_with_unresolved_context"
    elif count == 1:
        status = "clean_single_transition"
    else:
        status = "multiple_candidate_transitions"
    return {
        "molid": str(molid),
        "input_smiles": str(smiles),
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "status": status,
        "candidate_transition_count": int(count),
        "resolved_site_count": int(len(resolved)),
        "unresolved_context_count": int(len(unresolved)),
        "transition_families": ";".join(sorted({
            str(transition.get("pair_family", "")) for transition in transitions
            if str(transition.get("pair_family", ""))
        })),
        "resolved_labels": ";".join(sorted({
            str(site.get("label", "")) for site in resolved
            if str(site.get("label", ""))
        })),
    }


def audit(input_path: str | Path, output_dir: str | Path) -> Dict[str, object]:
    """Write molecule-level assignability and source/provenance summaries."""
    source = Path(input_path)
    output = Path(output_dir)
    frame = pd.read_csv(source, sep="\t", low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"pKahub table is missing required columns: {missing}")

    conflicting = (
        frame.groupby("molid")["smiles"].nunique(dropna=False).loc[lambda x: x > 1]
    )
    if len(conflicting):
        raise ValueError(
            f"{len(conflicting)} molid values map to more than one SMILES"
        )
    molecules = frame[["molid", "smiles"]].drop_duplicates("molid")
    assignments = pd.DataFrame([
        _molecule_assignment(row.molid, row.smiles)
        for row in molecules.itertuples(index=False)
    ])

    source_rows = []
    for name, group in frame.groupby("data_source", dropna=False):
        source_rows.append({
            "data_source": str(name),
            "rows": int(len(group)),
            "unique_molids": int(group["molid"].nunique()),
            "exact_pka_rows": int(pd.to_numeric(
                group["pka_value"], errors="coerce"
            ).notna().sum()),
        })
    source_summary = pd.DataFrame(source_rows).sort_values("data_source")

    errors = pd.to_numeric(frame.get("assignment_error"), errors="coerce").dropna()
    statuses = Counter(assignments["status"])
    transition_counts = Counter(assignments["candidate_transition_count"])
    assignment_methods = {
        str(key): int(value)
        for key, value in frame["assignment_method"].fillna("").value_counts().items()
    }
    epik_rows = int(frame["assignment_method"].fillna("").str.lower().str.contains(
        "epik"
    ).sum())
    summary: Dict[str, object] = {
        "input_rows": int(len(frame)),
        "unique_molids": int(frame["molid"].nunique()),
        "unique_smiles": int(frame["smiles"].nunique()),
        "status_counts": dict(sorted(statuses.items())),
        "candidate_transition_count_distribution": {
            str(key): int(value) for key, value in sorted(transition_counts.items())
        },
        "molecules_with_unresolved_context": int(
            (assignments["unresolved_context_count"] > 0).sum()
        ),
        "assignment_provenance": {
            "epik_derived_rows": epik_rows,
            "non_epik_rows": int(len(frame) - epik_rows),
            "assignment_methods": assignment_methods,
            "assignment_error_median": float(np.median(errors)) if len(errors) else None,
            "assignment_error_p90": float(np.quantile(errors, 0.90)) if len(errors) else None,
            "policy": (
                "ignore supplied computational charge transitions as labels; "
                "use clean single-transition structures for direct site labels "
                "and fit multi-site rows through a macroscopic set loss"
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output / "molecule_assignability.csv", index=False)
    source_summary.to_csv(output / "source_summary.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(audit(args.input, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
