#!/usr/bin/env python3
"""Prepare a de-overlapped PKAD-3 candidate cohort for environment fitting.

PKAD-3's published split files repeat measurements across alternative PDB
structures and contain precomputed/model feature columns.  pKate consumes only
the experimental pKa and structural identifiers, ignores every computed target
or descriptor, removes exact PKAD-R sites, and assigns one half-unit of total
raw weight across all structures representing the same biological measurement.
The downstream trainer recomputes pKate descriptors from freshly acquired RCSB
coordinates and applies a locked PKAD-R grouped-validation gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PKAD3_DIR = (
    PROJECT_ROOT
    / "data/raw/external_datasets/protein_environment/kamls_pkad3/"
    "train_test_split"
)
DEFAULT_PKADR = (
    PROJECT_ROOT
    / "data/raw/external_datasets/protein_environment/pkadr/PKAD-R-250211.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/protein_environment/pkad3"
SUPPORTED_RESIDUES = frozenset({"ASP", "GLU", "HIS", "LYS", "CYS", "TYR"})
PDB_PATTERN = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
SOURCE_DOI = "10.1021/acs.jctc.4c01602"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _site_key(pdb: object, chain: object, residue: object, residue_id: object) -> str:
    return "|".join([
        str(pdb).upper(), str(chain), str(residue).upper(), str(residue_id),
    ])


def _pkadr_site_keys(path: Path) -> set[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        _site_key(
            row.get("PDB", ""), row.get("Chain", ""), row.get("ResName", ""),
            row.get("ResID_in_PDB", ""),
        )
        for row in rows
    }


def prepare(
    pkad3_dir: str | Path = DEFAULT_PKAD3_DIR,
    pkadr_path: str | Path = DEFAULT_PKADR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, object]:
    """Create training candidates and a reason-coded exclusion ledger."""
    source_dir = Path(pkad3_dir)
    train_path, test_path = source_dir / "train0.csv", source_dir / "test0.csv"
    train = pd.read_csv(train_path, low_memory=False).assign(source_split="train0")
    test = pd.read_csv(test_path, low_memory=False).assign(source_split="test0")
    frame = pd.concat([train, test], ignore_index=True, sort=False)
    required = {
        "PDB_ID", "Chain", "Res_Name", "Res_ID", "Expt_pKa", "Uniprot_ID",
        "info", "infor", "group_code",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"PKAD-3 split tables missing columns: {missing}")
    if frame["info"].astype(str).duplicated().any():
        raise ValueError("PKAD-3 split 0 contains duplicated structure-site rows")

    pkadr_keys = _pkadr_site_keys(Path(pkadr_path))
    accepted: list[Dict[str, object]] = []
    excluded: list[Dict[str, object]] = []
    ignored_computed_columns = sorted(
        column for column in frame.columns
        if column not in required | {"source_split"}
    )
    for source_index, row in frame.iterrows():
        pdb = str(row["PDB_ID"]).upper()
        residue = str(row["Res_Name"]).upper()
        base = {
            "source_row": int(source_index),
            "source_split": str(row["source_split"]),
            "pkad3_info": str(row["info"]),
            "pkad3_infor": str(row["infor"]),
            "pkad3_group_code": str(row["group_code"]),
            "pdb": pdb,
            "chain": str(row["Chain"]),
            "residue_name": residue,
            "residue_id": str(row["Res_ID"]),
            "uniprot_id": str(row["Uniprot_ID"]),
        }
        if not PDB_PATTERN.fullmatch(pdb):
            excluded.append({**base, "reason": "no_experimental_four_character_pdb"})
            continue
        if residue not in SUPPORTED_RESIDUES:
            excluded.append({**base, "reason": "unsupported_residue_class"})
            continue
        try:
            experimental = float(row["Expt_pKa"])
        except (TypeError, ValueError):
            excluded.append({**base, "reason": "non_numeric_experimental_pka"})
            continue
        if not np.isfinite(experimental):
            excluded.append({**base, "reason": "non_finite_experimental_pka"})
            continue
        site_key = _site_key(pdb, row["Chain"], residue, row["Res_ID"])
        if site_key in pkadr_keys:
            excluded.append({
                **base,
                "experimental_pka": experimental,
                "reason": "exact_structure_site_overlap_with_pkadr",
            })
            continue
        measurement_key = "|".join([
            str(row["infor"]), residue, f"{experimental:.8g}",
        ])
        accepted.append({
            **base,
            "site_key": site_key,
            "experimental_pka": experimental,
            "measurement_group_key": measurement_key,
            "protein_group_key": str(row["Uniprot_ID"]),
            "label_origin": "experimental",
            "computational_teacher": "",
            "computed_pkad3_features_used": False,
            "computed_model_pka_used": False,
            "eligible_for_training": True,
            "source_dataset": "PKAD-3/KaMLs published split-0 union",
            "source_doi": SOURCE_DOI,
            "source_reference_available": False,
        })

    candidates = pd.DataFrame(accepted)
    if not candidates.empty:
        structures_per_measurement = candidates.groupby("measurement_group_key")[
            "site_key"
        ].transform("count").astype(float)
        candidates["sample_weight_raw"] = 0.5 / structures_per_measurement
        candidates["supplement_id"] = [
            "pkad3_" + hashlib.sha256(
                f"{site}|{measurement}".encode()
            ).hexdigest()[:16]
            for site, measurement in zip(
                candidates["site_key"], candidates["measurement_group_key"]
            )
        ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output / "training_candidates.csv", index=False)
    pd.DataFrame(excluded).to_csv(output / "preparation_exclusions.csv", index=False)
    summary = {
        "schema_version": "pkad3_supplement_1.0.0",
        "source_doi": SOURCE_DOI,
        "source_files": {
            "train0.csv": sha256(train_path),
            "test0.csv": sha256(test_path),
        },
        "published_split_rows": int(len(frame)),
        "published_structure_site_rows": int(frame["info"].nunique()),
        "published_biological_site_groups": int(frame["group_code"].nunique()),
        "candidate_structure_site_rows": int(len(candidates)),
        "candidate_measurement_groups": int(
            candidates["measurement_group_key"].nunique()
            if not candidates.empty else 0
        ),
        "candidate_pdb_entries": int(
            candidates["pdb"].nunique() if not candidates.empty else 0
        ),
        "excluded_rows": int(len(excluded)),
        "exclusion_reasons": {
            str(key): int(value)
            for key, value in pd.Series([
                row["reason"] for row in excluded
            ], dtype=str).value_counts().items()
        },
        "raw_candidate_weight": float(
            candidates["sample_weight_raw"].sum() if not candidates.empty else 0.0
        ),
        "ignored_computed_columns": ignored_computed_columns,
        "policy": (
            "Use experimental Expt_pKa only; ignore published model_pka and all "
            "precomputed descriptors; remove exact PKAD-R structure-sites; share "
            "half a canonical row of raw weight across alternative structures of "
            "one measurement; recompute pKate features from RCSB coordinates"
        ),
        "provenance_limitation": (
            "The published split files do not carry row-level reference URLs; "
            "reference-level de-overlap cannot be asserted from this snapshot"
        ),
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkad3-dir", default=str(DEFAULT_PKAD3_DIR))
    parser.add_argument("--pkadr", default=str(DEFAULT_PKADR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    print(json.dumps(prepare(
        args.pkad3_dir, args.pkadr, args.output_dir
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
