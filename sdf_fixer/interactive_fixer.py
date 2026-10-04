"""
interactive_fixer.py
====================
Iterate over every record in an SDF file, auto-fix obvious protonation
errors, and pause to ask the user when the assignment is ambiguous.

The user sees:
  * a PNG image of the molecule with numbered atoms and highlighted groups
  * an ASCII atom table
  * all ionizable groups with pKa plausibility ranking
  * a prompt asking which atom to protonate/deprotonate (or skip)

Usage (from project root)::

    python -m sdf_fixer.fix  data/raw/INCORRECT_literature_compilation.sdf
    python -m sdf_fixer.fix  data/raw/INCORRECT_literature_compilation.sdf --ph 7.4
"""

from __future__ import annotations

import copy
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem  # type: ignore

from sdf_fixer.sdf_reader import SDFRecord, read_sdf
from sdf_fixer.group_identifier import (
    GroupIdentification,
    IonizableSite,
    identify_groups,
    rank_sites_by_pka_plausibility,
    infer_pka_label,
)
from sdf_fixer.protonation_evaluator import (
    ProtonationAssessment,
    Verdict,
    evaluate_record,
    _current_form_of,
    _deprotonate_at,
    _protonate_at,
)
from sdf_fixer.mol_visualizer import (
    atom_table_str,
    render_for_user,
    get_vis_dir,
)


# ---------------------------------------------------------------------------
# Fix log entry
# ---------------------------------------------------------------------------

@dataclass
class FixLogEntry:
    record_index: int
    name: Optional[str]
    original_smiles: str
    fixed_smiles: str
    pka_value: Optional[float]
    verdict: str             # Verdict.value
    action: str              # "auto" | "user" | "skip" | "kept"
    site_label: Optional[str]
    atom_index: Optional[int]
    user_command: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "record_index": self.record_index,
            "name": self.name or "",
            "original_smiles": self.original_smiles,
            "fixed_smiles": self.fixed_smiles,
            "pka_value": self.pka_value,
            "verdict": self.verdict,
            "action": self.action,
            "site_label": self.site_label or "",
            "atom_index": self.atom_index if self.atom_index is not None else "",
            "user_command": self.user_command or "",
        }


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and sys.stdin.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def _green(t: str) -> str:  return _c("32", t)
def _red(t: str) -> str:    return _c("31", t)
def _yellow(t: str) -> str: return _c("33", t)
def _cyan(t: str) -> str:   return _c("36", t)
def _bold(t: str) -> str:   return _c("1", t)


# ---------------------------------------------------------------------------
# Try to open an image in the user's viewer
# ---------------------------------------------------------------------------

def _try_open_image(path: str) -> bool:
    """Best-effort attempt to open *path* in a GUI image viewer."""
    openers = []
    system = platform.system()
    if system == "Darwin":
        openers = ["open"]
    elif system == "Linux":
        openers = ["xdg-open", "eog", "feh", "display"]
    elif system == "Windows":
        openers = ["start"]

    for cmd in openers:
        if shutil.which(cmd):
            try:
                subprocess.Popen(
                    [cmd, path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception:
                continue
    return False


# Wrapper alias.
_infer_pka_label = infer_pka_label


# ---------------------------------------------------------------------------
# User interaction for ambiguous / borderline / no-ionizable records
# ---------------------------------------------------------------------------

def _form_tag(form: Optional[str]) -> str:
    if form == "acid_form":
        return "protonated/acid-form"
    elif form == "base_form":
        return "deprotonated/base-form"
    return "?"


def _print_record_header(rec: SDFRecord, gid: GroupIdentification) -> None:
    print(f"\n{'='*76}")
    header = f"Record #{rec.record_index}"
    if rec.name:
        header += f"  ({rec.name})"
    print(_bold(header))
    print(f"  Original  SMILES : {rec.original_smiles}")
    print(f"  Canonical SMILES : {rec.canonical_smiles}")
    _pka_tag = _infer_pka_label(gid)
    print(f"  {_pka_tag} = {rec.pka_value}   mol_charge = {rec.formal_charge}")


def _print_ionizable_groups(
    gid: GroupIdentification,
    pka_value: Optional[float],
) -> None:
    if not gid.ionizable_sites:
        print("  No ionizable groups detected.")
        if gid.non_ionizable_sites:
            labels = ", ".join(s["label"] for s in gid.non_ionizable_sites)
            print(f"  Non-ionizable groups present: {labels}")
        return

    _pka_tag = _infer_pka_label(gid)
    print(f"  Ionizable groups ({gid.n_ionizable}):")
    for idx, site in enumerate(gid.ionizable_sites, 1):
        conj = site.conjugate_label or "n/a"
        eff_ref = site.effective_reference_pka
        ref_tag = f"ref {site.pka_label}≈{eff_ref:.0f}" if eff_ref is not None else "ref=?"
        cur = _current_form_of(site.label) or "?"
        print(
            f"    {idx}. [{site.label}]  family={site.family}  "
            f"current={_form_tag(cur)}  conj={conj}  {ref_tag}  "
            f"atoms={sorted(site.atom_set)}"
        )

    if pka_value is not None and gid.n_ionizable > 1:
        ranking = rank_sites_by_pka_plausibility(gid.ionizable_sites, pka_value)
        print(f"  Plausibility ranking ({_pka_tag} = {pka_value}):")
        for rank, (site, diff) in enumerate(ranking, 1):
            marker = " ← best match" if rank == 1 else ""
            print(f"    {rank}. [{site.label}]  |{_pka_tag} − ref| = {diff:.1f}{marker}")


def _show_atom_table(mol: Chem.Mol, gid: GroupIdentification) -> None:
    print()
    print(atom_table_str(mol, gid))
    print()


def _show_visualization(
    mol: Chem.Mol,
    gid: GroupIdentification,
    rec: SDFRecord,
) -> None:
    """Render a PNG and tell the user where it is / try to open it."""
    png_path = render_for_user(mol, gid, rec.record_index, rec.pka_value)
    opened = _try_open_image(png_path)
    if opened:
        print(f"  [Image opened: {png_path}]")
    else:
        print(f"  [Image saved: {png_path}]")
        print(f"   (Open manually to view the labelled structure)")


def _apply_user_edit(mol: Chem.Mol, command: str) -> Tuple[Optional[str], Optional[int], str]:
    """
    Parse a user command and apply it to *mol*.

    Supported commands
    ------------------
    ``p <atom_idx>``   — protonate atom (add H, +1 charge)
    ``d <atom_idx>``   — deprotonate atom (remove H, -1 charge)
    ``s``              — skip this record (keep original SMILES)
    ``k``              — keep current SMILES as-is
    ``q``              — quit the interactive session

    Returns ``(new_smiles_or_None, atom_idx, action_tag)``.
    """
    parts = command.strip().lower().split()
    if not parts:
        return None, None, "invalid"

    cmd = parts[0]

    if cmd in ("s", "skip"):
        return None, None, "skip"

    if cmd in ("k", "keep"):
        return Chem.MolToSmiles(mol, canonical=True), None, "kept"

    if cmd in ("q", "quit", "exit"):
        return None, None, "quit"

    if cmd in ("p", "prot", "protonate") and len(parts) >= 2:
        try:
            atom_idx = int(parts[1])
        except ValueError:
            return None, None, "invalid"
        new_smiles = _protonate_at(mol, atom_idx)
        return new_smiles, atom_idx, "user_protonate"

    if cmd in ("d", "deprot", "deprotonate") and len(parts) >= 2:
        try:
            atom_idx = int(parts[1])
        except ValueError:
            return None, None, "invalid"
        new_smiles = _deprotonate_at(mol, atom_idx)
        return new_smiles, atom_idx, "user_deprotonate"

    return None, None, "invalid"


def ask_user(
    rec: SDFRecord,
    gid: GroupIdentification,
    assessment: ProtonationAssessment,
) -> FixLogEntry:
    """
    Display the molecule, groups, and pKa to the user, then ask them
    what to do.  Returns a `FixLogEntry` recording the decision.
    """
    _print_record_header(rec, gid)
    _print_ionizable_groups(gid, rec.pka_value)
    _show_atom_table(rec.mol, gid)
    _show_visualization(rec.mol, gid, rec)

    print(_yellow(f"  VERDICT: {assessment.verdict.value}"))
    print(f"  {assessment.explanation}")
    print()
    print(_bold("  Commands:"))
    print("    p <atom#>   protonate atom (add H⁺, +1 charge)")
    print("    d <atom#>   deprotonate atom (remove H, −1 charge)")
    print("    k           keep current SMILES unchanged")
    print("    s           skip (leave original, flag in log)")
    print("    q           quit interactive session")
    print()

    while True:
        try:
            raw = input(_cyan("  > "))
        except (EOFError, KeyboardInterrupt):
            print()
            return FixLogEntry(
                record_index=rec.record_index,
                name=rec.name,
                original_smiles=rec.original_smiles,
                fixed_smiles=rec.canonical_smiles,
                pka_value=rec.pka_value,
                verdict=assessment.verdict.value,
                action="skip",
                site_label=None,
                atom_index=None,
                user_command="EOF",
            )

        new_smiles, atom_idx, action_tag = _apply_user_edit(rec.mol, raw)

        if action_tag == "quit":
            raise _QuitSignal()

        if action_tag == "skip":
            return FixLogEntry(
                record_index=rec.record_index,
                name=rec.name,
                original_smiles=rec.original_smiles,
                fixed_smiles=rec.canonical_smiles,
                pka_value=rec.pka_value,
                verdict=assessment.verdict.value,
                action="skip",
                site_label=None,
                atom_index=None,
                user_command=raw.strip(),
            )

        if action_tag == "kept":
            print(_green(f"  Kept: {new_smiles}"))
            return FixLogEntry(
                record_index=rec.record_index,
                name=rec.name,
                original_smiles=rec.original_smiles,
                fixed_smiles=new_smiles or rec.canonical_smiles,
                pka_value=rec.pka_value,
                verdict=assessment.verdict.value,
                action="kept",
                site_label=None,
                atom_index=None,
                user_command=raw.strip(),
            )

        if action_tag == "invalid":
            print(_red("  Invalid command. Try: p <atom#>, d <atom#>, k, s, q"))
            continue

        if new_smiles is None:
            print(_red(f"  RDKit could not apply that change to atom {atom_idx}. Try another."))
            continue

        # Success.
        print(_green(f"  Applied: {new_smiles}"))
        # Find site label for logging.
        site_label = None
        if atom_idx is not None:
            for site in gid.ionizable_sites:
                if atom_idx in site.atom_set:
                    site_label = site.label
                    break

        return FixLogEntry(
            record_index=rec.record_index,
            name=rec.name,
            original_smiles=rec.original_smiles,
            fixed_smiles=new_smiles,
            pka_value=rec.pka_value,
            verdict=assessment.verdict.value,
            action="user",
            site_label=site_label,
            atom_index=atom_idx,
            user_command=raw.strip(),
        )


class _QuitSignal(Exception):
    """Raised when the user types ``q`` to abort the session."""
    pass


# ---------------------------------------------------------------------------
# SDF writer
# ---------------------------------------------------------------------------

def _write_fixed_sdf(
    original_path: str,
    out_path: str,
    fixes: Dict[int, str],
) -> int:
    """
    Re-read *original_path* and write *out_path*, replacing the SMILES
    property for every record whose index appears in *fixes*.

    Returns the number of records written.
    """
    count = 0
    with open(original_path, "rb") as fh_in:
        supplier = Chem.ForwardSDMolSupplier(fh_in, removeHs=False)
        writer = Chem.SDWriter(out_path)

        for record_idx, mol in enumerate(supplier):
            if mol is None:
                continue

            if record_idx in fixes:
                new_smiles = fixes[record_idx]
                orig_smi = mol.GetProp("SMILES") if mol.HasProp("SMILES") else ""
                mol.SetProp("SMILES_original", orig_smi)
                mol.SetProp("SMILES", new_smiles)
                # Try to also update the mol object itself to reflect the
                # new protonation, but keep the original if parsing fails.
                new_mol = Chem.MolFromSmiles(new_smiles)
                if new_mol is not None:
                    # Carry over all properties.
                    for prop_name in mol.GetPropNames():
                        if not new_mol.HasProp(prop_name):
                            new_mol.SetProp(prop_name, mol.GetProp(prop_name))
                    new_mol.SetProp("SMILES", new_smiles)
                    writer.write(new_mol)
                else:
                    writer.write(mol)
            else:
                writer.write(mol)
            count += 1

        writer.close()
    return count


# ---------------------------------------------------------------------------
# Main fixer loop
# ---------------------------------------------------------------------------

def fix_sdf(
    sdf_path: str,
    out_path: Optional[str] = None,
    log_path: Optional[str] = None,
    *,
    ph: float = 7.4,
    max_records: int = 0,
    auto_only: bool = False,
) -> Tuple[str, str, List[FixLogEntry]]:
    """
    Main entry point: iterate over records, auto-fix the obvious ones,
    pause for user input on ambiguous cases, and write the corrected SDF.

    Parameters
    ----------
    sdf_path : str
        Input SDF file.
    out_path : str, optional
        Output SDF.  Default: ``FIXED_<basename>``.
    log_path : str, optional
        CSV log of all changes.  Default: ``FIXLOG_<basename>.csv``.
    ph : float
        Reference pH.
    max_records : int
        0 = process all.
    auto_only : bool
        If True, skip interactive prompts (only auto-fix clear cases).

    Returns
    -------
    (out_path, log_path, log_entries)
    """
    basename = os.path.basename(sdf_path)
    dirname = os.path.dirname(sdf_path) or "."

    if out_path is None:
        out_path = os.path.join(dirname, f"FIXED_{basename}")
    if log_path is None:
        stem = os.path.splitext(basename)[0]
        log_path = os.path.join(dirname, f"FIXLOG_{stem}.csv")

    print(f"Reading {sdf_path} …")
    records = read_sdf(sdf_path)
    total = len(records)
    print(f"Parsed {total} records.")
    if max_records > 0:
        records = records[:max_records]
        print(f"Processing first {max_records} records.")
    print(f"Visualizations directory: {get_vis_dir()}")
    print()

    log: List[FixLogEntry] = []
    fixes: Dict[int, str] = {}  # record_index → corrected SMILES
    counts: Counter = Counter()

    try:
        for i, rec in enumerate(records):
            progress = f"[{i+1}/{len(records)}]"
            gid = identify_groups(rec.mol)
            assessment = evaluate_record(rec, gid, ph=ph)
            verdict = assessment.verdict

            # ============================================================
            # CASE 1: correct — no change needed
            # ============================================================
            if verdict == Verdict.CORRECT:
                counts["correct"] += 1
                log.append(FixLogEntry(
                    record_index=rec.record_index,
                    name=rec.name,
                    original_smiles=rec.original_smiles,
                    fixed_smiles=rec.canonical_smiles,
                    pka_value=rec.pka_value,
                    verdict=verdict.value,
                    action="kept",
                    site_label=assessment.assigned_site.label if assessment.assigned_site else None,
                    atom_index=None,
                ))
                continue

            # ============================================================
            # CASE 2: clear auto-fix (single ionizable group, clear pKa)
            # ============================================================
            if verdict in (Verdict.NEEDS_PROTONATION, Verdict.NEEDS_DEPROTONATION):
                if assessment.corrected_smiles:
                    fixes[rec.record_index] = assessment.corrected_smiles
                    counts["auto_fixed"] += 1
                    _site = assessment.assigned_site
                    _ptag = _site.pka_label if _site else "pKa"
                    print(
                        f"  {progress} #{rec.record_index:>4d}  "
                        f"{_green('AUTO-FIX')}  "
                        f"{rec.original_smiles}  →  {assessment.corrected_smiles}  "
                        f"({_site.label if _site else '?'}, "
                        f"{_ptag}={rec.pka_value})"
                    )
                    log.append(FixLogEntry(
                        record_index=rec.record_index,
                        name=rec.name,
                        original_smiles=rec.original_smiles,
                        fixed_smiles=assessment.corrected_smiles,
                        pka_value=rec.pka_value,
                        verdict=verdict.value,
                        action="auto",
                        site_label=assessment.assigned_site.label if assessment.assigned_site else None,
                        atom_index=None,
                    ))
                    continue
                else:
                    # Needs fix but auto-correction failed — fall through to interactive.
                    pass

            # ============================================================
            # CASE 2b: no ionizable groups — nothing to change, skip silently
            # ============================================================
            if verdict == Verdict.NO_IONIZABLE:
                counts["no_ionizable"] += 1
                log.append(FixLogEntry(
                    record_index=rec.record_index,
                    name=rec.name,
                    original_smiles=rec.original_smiles,
                    fixed_smiles=rec.canonical_smiles,
                    pka_value=rec.pka_value,
                    verdict=verdict.value,
                    action="kept",
                    site_label=None,
                    atom_index=None,
                ))
                continue

            # ============================================================
            # CASE 3: needs user input (ambiguous / borderline / auto-fail)
            # ============================================================
            if auto_only:
                counts["skipped"] += 1
                print(
                    f"  {progress} #{rec.record_index:>4d}  "
                    f"{_yellow('SKIPPED')}  {verdict.value}  {rec.original_smiles}"
                )
                log.append(FixLogEntry(
                    record_index=rec.record_index,
                    name=rec.name,
                    original_smiles=rec.original_smiles,
                    fixed_smiles=rec.canonical_smiles,
                    pka_value=rec.pka_value,
                    verdict=verdict.value,
                    action="skip",
                    site_label=None,
                    atom_index=None,
                ))
                continue

            # Interactive prompt.
            counts["interactive"] += 1
            entry = ask_user(rec, gid, assessment)
            log.append(entry)
            if entry.action == "user" and entry.fixed_smiles:
                fixes[rec.record_index] = entry.fixed_smiles
            elif entry.action == "kept" and entry.fixed_smiles:
                # User said keep → no change
                pass

    except _QuitSignal:
        print(_yellow("\nUser quit. Saving progress so far …"))

    # ---- write outputs ----
    n_written = _write_fixed_sdf(sdf_path, out_path, fixes)
    _write_log(log, log_path)

    # ---- summary ----
    print(f"\n{'='*76}")
    print(_bold("FIX SUMMARY"))
    print(f"  Records processed : {len(records)}")
    print(f"  Correct (no fix)  : {counts['correct']}")
    print(f"  No ionizable grps : {counts['no_ionizable']}")
    print(f"  Auto-fixed        : {counts['auto_fixed']}")
    print(f"  Interactive fixes  : {counts['interactive']}")
    print(f"  Skipped           : {counts['skipped']}")
    print(f"  Output SDF        : {out_path}  ({n_written} records)")
    print(f"  Fix log           : {log_path}")
    print(f"  Visualizations    : {get_vis_dir()}")

    return out_path, log_path, log


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------

def _write_log(log: List[FixLogEntry], log_path: str) -> None:
    if not log:
        return
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    fieldnames = [
        "record_index", "name", "original_smiles", "fixed_smiles",
        "pka_value", "verdict", "action", "site_label", "atom_index",
        "user_command",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in log:
            writer.writerow(entry.as_dict())
