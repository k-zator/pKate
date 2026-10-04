#!/usr/bin/env python
"""
cli.py
======
Command-line entry point for the SDF reader + group identifier +
protonation evaluation.

Usage (from the project root)::

    python -m sdf_fixer.cli data/raw/INCORRECT_literature_compilation.sdf
    python -m sdf_fixer.cli data/raw/INCORRECT_literature_compilation.sdf -n 30 -v

This will parse the file, identify ionizable functional groups, evaluate
whether each SMILES correctly encodes the protonation state implied by
the reported pKa at pH 7.4, and print a per-record assessment.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from typing import List

from sdf_fixer.sdf_reader import SDFRecord, read_sdf
from sdf_fixer.group_identifier import (
    GroupIdentification,
    IonizableSite,
    identify_groups,
    classify_protonation_expectation,
    rank_sites_by_pka_plausibility,
    infer_pka_label,
)
from sdf_fixer.protonation_evaluator import (
    ProtonationAssessment,
    Verdict,
    evaluate_record,
    _current_form_of,
)


# ---------------------------------------------------------------------------
# ANSI colour helpers (for terminal readability)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def _green(t: str) -> str: return _c("32", t)
def _red(t: str) -> str:   return _c("31", t)
def _yellow(t: str) -> str: return _c("33", t)
def _cyan(t: str) -> str:  return _c("36", t)
def _bold(t: str) -> str:  return _c("1", t)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _form_tag(form: str | None) -> str:
    if form == "acid_form":
        return "protonated/acid-form"
    elif form == "base_form":
        return "deprotonated/base-form"
    return "?"


def _verdict_banner(v: Verdict) -> str:
    if v == Verdict.CORRECT:
        return _green("CORRECT")
    elif v == Verdict.NEEDS_DEPROTONATION:
        return _red("NEEDS DEPROTONATION")
    elif v == Verdict.NEEDS_PROTONATION:
        return _red("NEEDS PROTONATION")
    elif v == Verdict.BORDERLINE:
        return _yellow("BORDERLINE (pKa ≈ pH)")
    elif v == Verdict.AMBIGUOUS:
        return _yellow("AMBIGUOUS (multiple groups)")
    elif v == Verdict.NO_IONIZABLE:
        return _cyan("NO IONIZABLE GROUP")
    return "UNKNOWN"


# Wrapper aliases for consistent naming.
_infer_pka_label = infer_pka_label


def print_evaluation(
    rec: SDFRecord,
    gid: GroupIdentification,
    assessment: ProtonationAssessment,
    *,
    verbose: bool = False,
) -> None:
    """Pretty-print one record's full evaluation."""
    print(f"\n{'='*76}")
    header = f"Record #{rec.record_index}"
    if rec.name:
        header += f"  ({rec.name})"
    print(_bold(header))
    print(f"  Original  SMILES: {rec.original_smiles}")
    print(f"  Canonical SMILES: {rec.canonical_smiles}")
    # Determine correct label: pKaH for bases, pKa for acids.
    _pka_tag = _infer_pka_label(gid)
    print(f"  {_pka_tag} = {rec.pka_value}   temp = {rec.temperature}   "
          f"mol_charge = {rec.formal_charge}")

    # -- ionizable groups --
    if gid.ionizable_sites:
        print(f"  Ionizable groups ({gid.n_ionizable}):")
        for idx, site in enumerate(gid.ionizable_sites, 1):
            conj = site.conjugate_label or "n/a"
            eff_ref = site.effective_reference_pka
            ref_tag = f"ref {site.pka_label}≈{eff_ref:.0f}" if eff_ref is not None else "ref=?"
            cur  = _current_form_of(site.label) or "?"
            print(
                f"    {idx}. [{site.label}]  family={site.family}  "
                f"current={_form_tag(cur)}  conj={conj}  {ref_tag}  "
                f"atoms={site.atom_indices}"
            )

        if rec.pka_value is not None and gid.n_ionizable > 1:
            ranking = rank_sites_by_pka_plausibility(gid.ionizable_sites, rec.pka_value)
            print(f"  Plausibility ranking ({_pka_tag}={rec.pka_value}):")
            for rank, (site, diff) in enumerate(ranking, 1):
                marker = " ← best match" if rank == 1 else ""
                print(f"    {rank}. [{site.label}]  |{_pka_tag} − ref| = {diff:.1f}{marker}")
    else:
        if verbose and gid.non_ionizable_sites:
            print(f"  Non-ionizable groups ({len(gid.non_ionizable_sites)}):")
            for site in gid.non_ionizable_sites:
                print(f"    - {site['label']}  ({site['type']})")

    # -- verdict --
    print(f"  VERDICT: {_verdict_banner(assessment.verdict)}")
    print(f"    {assessment.explanation}")

    if assessment.assigned_site:
        cur_tag = _form_tag(assessment.current_form)
        exp_tag = _form_tag(assessment.expected_form)
        print(f"    Assigned site : [{assessment.assigned_site.label}]")
        print(f"    Current form  : {cur_tag}")
        print(f"    Expected form : {exp_tag}")

    if assessment.corrected_smiles:
        print(f"    CORRECTED SMILES: {_red(assessment.corrected_smiles)}")
    elif assessment.needs_fix:
        print(f"    (could not compute corrected SMILES automatically)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read an SDF, identify ionizable groups, evaluate protonation correctness.",
    )
    parser.add_argument("sdf", help="Path to the .sdf file to analyse.")
    parser.add_argument("-n", "--max-records", type=int, default=0,
                        help="Process at most N records (0 = all).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show non-ionizable groups too.")
    parser.add_argument("--ph", type=float, default=7.4,
                        help="Reference pH for protonation evaluation (default 7.4).")
    parser.add_argument("--overlap-threshold", type=float, default=0.5,
                        help="Overlap threshold for site resolution.")
    parser.add_argument("--only-fixes", action="store_true",
                        help="Only print records that need a fix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading {args.sdf} …")
    records = read_sdf(args.sdf)
    print(f"Parsed {len(records)} records.\n")

    if args.max_records > 0:
        records = records[: args.max_records]

    verdict_counts: Counter = Counter()
    fix_count = 0
    corrected_count = 0

    for rec in records:
        gid = identify_groups(rec.mol, overlap_threshold=args.overlap_threshold)
        assessment = evaluate_record(rec, gid, ph=args.ph)
        verdict_counts[assessment.verdict] += 1

        if assessment.needs_fix:
            fix_count += 1
            if assessment.corrected_smiles:
                corrected_count += 1

        if args.only_fixes and not assessment.needs_fix:
            continue

        print_evaluation(rec, gid, assessment, verbose=args.verbose)

    # -- summary --
    print(f"\n{'='*76}")
    print(_bold("SUMMARY"))
    print(f"  Total records          : {len(records)}")
    for v in Verdict:
        cnt = verdict_counts.get(v, 0)
        if cnt:
            print(f"  {v.value:<24s}: {cnt}")
    print(f"  Need amendment         : {fix_count}")
    print(f"  Auto-corrected SMILES  : {corrected_count}")
    unfixable = fix_count - corrected_count
    if unfixable > 0:
        print(f"  Failed auto-correction : {unfixable}")


if __name__ == "__main__":
    main()
