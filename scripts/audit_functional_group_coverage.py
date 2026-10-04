#!/usr/bin/env python3
"""Audit functional-group and conjugate-pair coverage atom-by-atom.

This does not pretend that a finite SMARTS library is chemically exhaustive.
It establishes measurable coverage on the project corpora and separates:

* no SMARTS candidate covers a ring heteroatom;
* a candidate exists but overlap resolution removes it; and
* a resolved descriptor exists but has no supported conjugate-pair transform.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem import Draw  # type: ignore

from functional_group_pka_analysis import classify_pair_type, normalize_conjugate_family
from pka_evidence_policy import ionization_policy_role
from SMARTS_library import ACIDIC_SMARTS, BASIC_SMARTS
from substructure_match import (
    FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION,
    find_sites_with_metadata,
    resolve_overlapping_sites,
)


DEFAULT_ASSIGNMENTS = "data/processed/functional_group_assignments.csv"
DEFAULT_NETWORKS = "data/processed/pka_molecule_microstate_network_stage3_predictions.csv"
DEFAULT_OUT_DIR = "data/processed/dataset_audit/functional_group_coverage"
HETERO_ATOMIC_NUMBERS = {5, 7, 8, 15, 16}


def _environment(mol: Chem.Mol, atom_index: int, radius: int = 2) -> str:
    bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_index)
    atoms = {atom_index}
    for bond_index in bonds:
        bond = mol.GetBondWithIdx(bond_index)
        atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return Chem.MolFragmentToSmiles(mol, atomsToUse=sorted(atoms), canonical=True)


def _pair_supported(site: Dict) -> bool:
    family = normalize_conjugate_family(site.get("label"))
    mode, member_form = classify_pair_type(site.get("label"), family)
    return mode == "pair_type" and member_form in {"acid_form", "base_form"}


def _scan_records(records: Iterable[Dict], corpus: str, overlap_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    atom_rows: List[Dict] = []
    molecule_rows: List[Dict] = []
    for record in records:
        smiles = str(record.get("smiles", ""))
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            molecule_rows.append({**record, "corpus": corpus, "parse_status": "invalid_smiles"})
            continue
        candidates = find_sites_with_metadata(mol)
        resolved, rejected = resolve_overlapping_sites(
            candidates, overlap_threshold=overlap_threshold
        )
        specific_candidates = [site for site in candidates if not site.get("is_fallback")]
        specific_resolved = [site for site in resolved if not site.get("is_fallback")]
        fallback_resolved = [site for site in resolved if site.get("is_fallback")]
        candidate_atoms = {
            value for site in specific_candidates for value in site.get("atom_set", set())
        }
        resolved_atoms = {
            value for site in specific_resolved for value in site.get("atom_set", set())
        }
        pair_atoms = {
            value
            for site in resolved if _pair_supported(site)
            for value in site.get("atom_set", set())
        }
        ring_atoms = {value for ring in mol.GetRingInfo().AtomRings() for value in ring}
        ring_hetero = {
            value for value in ring_atoms
            if mol.GetAtomWithIdx(value).GetAtomicNum() in HETERO_ATOMIC_NUMBERS
        }
        ring_nitrogen = {
            value for value in ring_atoms if mol.GetAtomWithIdx(value).GetAtomicNum() == 7
        }
        molecule_rows.append({
            **record,
            "corpus": corpus,
            "parse_status": "ok",
            "candidate_site_count": len(candidates),
            "resolved_site_count": len(resolved),
            "specific_candidate_site_count": len(specific_candidates),
            "specific_resolved_site_count": len(specific_resolved),
            "fallback_nitrogen_site_count": len(fallback_resolved),
            "supported_pair_site_count": sum(_pair_supported(site) for site in resolved),
            "ring_heteroatom_count": len(ring_hetero),
            "ring_heteroatoms_with_candidate": len(ring_hetero & candidate_atoms),
            "ring_heteroatoms_resolved": len(ring_hetero & resolved_atoms),
            "ring_nitrogen_count": len(ring_nitrogen),
            "ring_nitrogens_with_candidate": len(ring_nitrogen & candidate_atoms),
            "ring_nitrogens_resolved": len(ring_nitrogen & resolved_atoms),
            "ring_nitrogens_with_supported_pair": len(ring_nitrogen & pair_atoms),
            "resolved_labels": "|".join(sorted({str(site["label"]) for site in resolved})),
        })
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in HETERO_ATOMIC_NUMBERS:
                continue
            atom_index = atom.GetIdx()
            candidate_labels = sorted({
                str(site["label"]) for site in candidates
                if atom_index in site.get("atom_set", set()) and not site.get("is_fallback")
            })
            resolved_labels = sorted({
                str(site["label"]) for site in resolved
                if atom_index in site.get("atom_set", set()) and not site.get("is_fallback")
            })
            fallback_labels = sorted({
                str(site["label"]) for site in resolved
                if atom_index in site.get("atom_set", set()) and site.get("is_fallback")
            })
            pair_labels = sorted({
                str(site["label"]) for site in resolved
                if atom_index in site.get("atom_set", set()) and _pair_supported(site)
            })
            atom_rows.append({
                **record,
                "corpus": corpus,
                "atom_index": atom_index,
                "element": atom.GetSymbol(),
                "is_ring_atom": atom_index in ring_atoms,
                "is_aromatic": atom.GetIsAromatic(),
                "formal_charge": atom.GetFormalCharge(),
                "total_hydrogens": atom.GetTotalNumHs(),
                "degree": atom.GetDegree(),
                "environment_radius2": _environment(mol, atom_index),
                "candidate_covered": bool(candidate_labels),
                "resolved_covered": bool(resolved_labels),
                "supported_pair_covered": bool(pair_labels),
                "fallback_covered": bool(fallback_labels),
                "candidate_labels": "|".join(candidate_labels),
                "resolved_labels_for_atom": "|".join(resolved_labels),
                "supported_pair_labels_for_atom": "|".join(pair_labels),
                "fallback_labels_for_atom": "|".join(fallback_labels),
            })
    return pd.DataFrame(atom_rows), pd.DataFrame(molecule_rows)


def _pattern_inventory() -> pd.DataFrame:
    rows = []
    all_labels = sorted(set(BASIC_SMARTS) | set(ACIDIC_SMARTS))
    for label in all_labels:
        family = normalize_conjugate_family(label)
        mode, form = classify_pair_type(label, family)
        rows.append({
            "label": label,
            "dictionary": "basic" if label in BASIC_SMARTS else "acidic",
            "family": family,
            "assignment_mode": mode,
            "pair_member_form": form,
            "has_supported_conjugate_pair": mode == "pair_type" and form in {
                "acid_form", "base_form",
            },
            "aqueous_ionization_policy_role": ionization_policy_role(label),
        })
    return pd.DataFrame(rows)


def _coverage_summary(atoms: pd.DataFrame, molecules: pd.DataFrame) -> Dict:
    valid = molecules[molecules["parse_status"].eq("ok")]
    nitrogen = atoms[atoms["element"] == "N"]
    ring_heteroatoms = atoms[atoms["is_ring_atom"]]
    ring_n = atoms[(atoms["element"] == "N") & atoms["is_ring_atom"]]
    heterocycles = valid[valid["ring_heteroatom_count"] > 0]
    return {
        "molecules": int(len(valid)),
        "invalid_smiles": int((molecules["parse_status"] != "ok").sum()),
        "molecules_with_no_specific_candidate_sites": int((
            valid["specific_candidate_site_count"] == 0
        ).sum()),
        "molecules_with_no_specific_resolved_sites": int((
            valid["specific_resolved_site_count"] == 0
        ).sum()),
        "molecules_with_unclassified_nitrogen_fallbacks": int((
            valid["fallback_nitrogen_site_count"] > 0
        ).sum()),
        "molecules_with_no_supported_conjugate_pair": int((
            valid["supported_pair_site_count"] == 0
        ).sum()),
        "heterocycle_molecules": int(len(heterocycles)),
        "heterocycle_molecules_with_no_candidate_covering_any_ring_heteroatom": int((
            heterocycles["ring_heteroatoms_with_candidate"] == 0
        ).sum()),
        "nitrogen_atoms": int(len(nitrogen)),
        "nitrogen_atoms_without_any_smarts_candidate": int((
            ~nitrogen["candidate_covered"]
        ).sum()),
        "nitrogen_atoms_assigned_unclassified_fallback": int(
            nitrogen["fallback_covered"].sum()
        ),
        "molecules_with_at_least_one_nitrogen_without_any_smarts_candidate": int(
            nitrogen.loc[~nitrogen["candidate_covered"], "record_id"].nunique()
        ),
        "ring_heteroatoms": int(len(ring_heteroatoms)),
        "ring_heteroatoms_without_any_smarts_candidate": int((
            ~ring_heteroatoms["candidate_covered"]
        ).sum()),
        "ring_nitrogen_atoms": int(len(ring_n)),
        "ring_nitrogen_atoms_without_any_smarts_candidate": int((
            ~ring_n["candidate_covered"]
        ).sum()),
        "molecules_with_at_least_one_ring_nitrogen_without_any_smarts_candidate": int(
            ring_n.loc[~ring_n["candidate_covered"], "record_id"].nunique()
        ),
        "ring_nitrogen_atoms_lost_during_overlap_resolution": int((
            ring_n["candidate_covered"] & ~ring_n["resolved_covered"]
        ).sum()),
        "molecules_with_ring_nitrogen_lost_during_overlap_resolution": int(
            ring_n.loc[
                ring_n["candidate_covered"] & ~ring_n["resolved_covered"],
                "record_id",
            ].nunique()
        ),
        "ring_nitrogen_atoms_resolved_but_without_supported_conjugate_pair": int((
            ring_n["resolved_covered"] & ~ring_n["supported_pair_covered"]
        ).sum()),
        "molecules_with_resolved_ring_nitrogen_but_no_supported_conjugate_pair": int(
            ring_n.loc[
                ring_n["resolved_covered"] & ~ring_n["supported_pair_covered"],
                "record_id",
            ].nunique()
        ),
    }


def _render_unmatched_gallery(rows: pd.DataFrame, output_path: Path, limit: int = 30) -> int:
    examples = rows.drop_duplicates("smiles").head(limit)
    mols, legends, highlights, colors = [], [], [], []
    for _, row in examples.iterrows():
        mol = Chem.MolFromSmiles(str(row["smiles"]))
        if mol is None:
            continue
        index = int(row["atom_index"])
        mols.append(mol)
        legends.append(
            f"{row.get('record_id', '')}\nN atom {index}; env={row['environment_radius2']}\n"
            "no SMARTS candidate"
        )
        highlights.append([index])
        colors.append({index: (0.9, 0.2, 0.2)})
    if mols:
        image = Draw.MolsToGridImage(
            mols, molsPerRow=3, subImgSize=(600, 400), legends=legends,
            highlightAtomLists=highlights, highlightAtomColors=colors, useSVG=False,
        )
        image.save(str(output_path))
    return len(mols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--networks", default=DEFAULT_NETWORKS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--canonical-only", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    networks = pd.read_csv(args.networks, low_memory=False)
    canonical_records = [
        {
            "record_id": str(row["molecule_id"]),
            "molecule_id": str(row["molecule_id"]),
            "source_file": "",
            "record_index": "",
            "smiles": str(row["representative_input_smiles"]),
        }
        for _, row in networks.drop_duplicates("molecule_id").iterrows()
    ]
    atom_frames, molecule_frames = [], []
    atoms, molecules = _scan_records(
        canonical_records, "canonical_stage3", args.overlap_threshold
    )
    atom_frames.append(atoms)
    molecule_frames.append(molecules)

    if not args.canonical_only:
        assignments = pd.read_csv(args.assignments, low_memory=False)
        assignments = assignments[
            assignments["pka_source_method"].astype(str).str.lower().eq("experimental")
        ].drop_duplicates("smiles")
        raw_records = [
            {
                "record_id": f"{row['source_file']}#{int(row['record_index'])}",
                "molecule_id": "",
                "source_file": str(row["source_file"]),
                "record_index": int(row["record_index"]),
                "smiles": str(row["smiles"]),
            }
            for _, row in assignments.iterrows()
        ]
        atoms, molecules = _scan_records(
            raw_records, "raw_experimental_unique_structure", args.overlap_threshold
        )
        atom_frames.append(atoms)
        molecule_frames.append(molecules)

    all_atoms = pd.concat(atom_frames, ignore_index=True)
    all_molecules = pd.concat(molecule_frames, ignore_index=True)
    ring_n = all_atoms[(all_atoms["element"] == "N") & all_atoms["is_ring_atom"]].copy()
    unmatched = ring_n[~ring_n["candidate_covered"]].copy()
    nitrogen_fallbacks = all_atoms[
        (all_atoms["element"] == "N") & all_atoms["fallback_covered"]
    ].copy()
    resolution_lost = ring_n[
        ring_n["candidate_covered"] & ~ring_n["resolved_covered"]
    ].copy()
    no_pair = ring_n[
        ring_n["resolved_covered"] & ~ring_n["supported_pair_covered"]
    ].copy()
    uncovered_heterocycles = all_molecules[
        (all_molecules["parse_status"] == "ok")
        & (all_molecules["ring_heteroatom_count"] > 0)
        & (all_molecules["ring_heteroatoms_with_candidate"] == 0)
    ].copy()
    inventory = _pattern_inventory()
    unmatched_environments = (
        unmatched.groupby(
            [
                "corpus", "environment_radius2", "is_aromatic", "formal_charge",
                "total_hydrogens", "degree",
            ],
            dropna=False,
        )
        .size()
        .rename("atom_count")
        .reset_index()
        .sort_values(["corpus", "atom_count"], ascending=[True, False])
    )

    for frame, name in (
        (all_atoms, "all_heteroatom_atom_audit.csv"),
        (nitrogen_fallbacks, "unclassified_nitrogen_fallbacks.csv"),
        (unmatched, "ring_nitrogens_without_smarts_candidate.csv"),
        (unmatched_environments, "unmatched_ring_nitrogen_environment_summary.csv"),
        (resolution_lost, "ring_nitrogens_lost_by_overlap_resolution.csv"),
        (no_pair, "ring_nitrogens_without_supported_conjugate_pair.csv"),
        (uncovered_heterocycles, "heterocycles_without_ring_heteroatom_description.csv"),
        (inventory, "smarts_pattern_inventory.csv"),
    ):
        frame.to_csv(out_dir / name, index=False)

    summary = {
        "functional_group_detector_schema_version": FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION,
        "overlap_threshold": float(args.overlap_threshold),
        "scope_warning": (
            "Coverage is empirical for these corpora, not proof of chemical exhaustiveness. "
            "Atom membership undercounts recursive center-only SMARTS (for example P is "
            "described through a matched O), so ring-N results are the primary hard audit."
        ),
        "corpora": {
            corpus: _coverage_summary(
                all_atoms[all_atoms["corpus"] == corpus],
                all_molecules[all_molecules["corpus"] == corpus],
            )
            for corpus in sorted(all_molecules["corpus"].unique())
        },
        "smarts_inventory": {
            "labels": int(len(inventory)),
            "labels_with_supported_conjugate_pair": int(
                inventory["has_supported_conjugate_pair"].sum()
            ),
            "descriptor_only_or_unpaired_labels": int(
                (~inventory["has_supported_conjugate_pair"]).sum()
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rendered = _render_unmatched_gallery(
        unmatched[unmatched["corpus"] == "canonical_stage3"],
        out_dir / "canonical_unmatched_ring_nitrogens.png",
    )
    sections = []
    for corpus, values in summary["corpora"].items():
        items = "".join(
            f"<li><b>{html.escape(key.replace('_', ' '))}</b>: {value}</li>"
            for key, value in values.items()
        )
        sections.append(f"<h2>{html.escape(corpus)}</h2><ul>{items}</ul>")
    index = f"""<!doctype html><html><head><meta charset='utf-8'>
    <title>Functional-group coverage audit</title></head><body>
    <h1>Functional-group coverage audit</h1>
    <p>{html.escape(summary['scope_warning'])}</p>{''.join(sections)}
    <h2>Audit files</h2>
    <ul>
      <li><a href='all_heteroatom_atom_audit.csv'>Every audited heteroatom</a></li>
      <li><a href='unclassified_nitrogen_fallbacks.csv'>Nitrogens requiring fallback/quarantine</a></li>
      <li><a href='ring_nitrogens_without_smarts_candidate.csv'>Ring N with no SMARTS candidate</a></li>
      <li><a href='unmatched_ring_nitrogen_environment_summary.csv'>Recurring unmatched ring-N environments</a></li>
      <li><a href='ring_nitrogens_lost_by_overlap_resolution.csv'>Ring N lost during overlap resolution</a></li>
      <li><a href='ring_nitrogens_without_supported_conjugate_pair.csv'>Described ring N without a supported pair</a></li>
      <li><a href='heterocycles_without_ring_heteroatom_description.csv'>Heterocycles with no ring-heteroatom description</a></li>
      <li><a href='smarts_pattern_inventory.csv'>SMARTS/pair inventory</a></li>
    </ul>
    <h2>Unmatched canonical ring nitrogens</h2>
    <p>Rendered examples: {rendered}. Red highlighting marks the unidentified N.</p>
    <img style='max-width:100%' src='canonical_unmatched_ring_nitrogens.png'>
    </body></html>"""
    (out_dir / "index.html").write_text(index)
    print(json.dumps(summary, indent=2))
    print(f"audit_index={out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
