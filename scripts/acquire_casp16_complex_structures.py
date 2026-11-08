#!/usr/bin/env python3
"""Acquire CASP16 pKa-associated protein-ligand complex coordinates.

The structures are retained for reproducible mapping and future evaluation,
not as protein-environment labels: the CASP16 table does not state that its
experimental ligand pKa values were measured in the bound complexes.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import pandas as pd

from acquire_pkad3_structures import RCSB_URL, _acquire_one, sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = (
    PROJECT_ROOT / "data/raw/external_datasets/casp16/chymasecatgSI.txt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/raw/external_datasets/casp16/structures"


def acquire(
    table_path: str | Path = DEFAULT_TABLE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    workers: int = 8,
) -> dict[str, object]:
    """Download complexes associated with numeric experimental pKa rows."""
    source = Path(table_path)
    frame = pd.read_csv(source, sep=";", low_memory=False)
    required = {"pdb", "PKA1", "PKA2"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CASP16 table missing columns: {missing}")
    numeric = (
        pd.to_numeric(frame["PKA1"], errors="coerce").notna()
        | pd.to_numeric(frame["PKA2"], errors="coerce").notna()
    )
    pdb_ids = sorted({
        str(value).strip().upper()
        for value in frame.loc[numeric, "pdb"]
        if pd.notna(value) and str(value).strip()
    })
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {
            executor.submit(_acquire_one, pdb, target): pdb for pdb in pdb_ids
        }
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda row: str(row["pdb"]))
    manifest = {
        "schema_version": "casp16_complex_acquisition_1.0.0",
        "source_table": str(source),
        "source_table_sha256": sha256(source),
        "source_doi": "10.5281/zenodo.16762332",
        "url_template": RCSB_URL,
        "requested_pdb_entries": len(pdb_ids),
        "available_pdb_entries": sum(
            row["status"] != "failed" for row in records
        ),
        "failed_pdb_entries": sum(row["status"] == "failed" for row in records),
        "structures": records,
        "label_semantics": (
            "reference_complex_coordinates_only; experimental ligand pKa is "
            "aqueous_or_assay evidence and is not a bound-state label"
        ),
        "eligible_for_protein_environment_fitting": False,
    }
    (target / "acquisition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=str(DEFAULT_TABLE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    report = acquire(args.table, args.output_dir, workers=args.workers)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failed_pdb_entries"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
