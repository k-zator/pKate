#!/usr/bin/env python3
"""Acquire the experimental structures needed for PKAD-3 augmentation.

Only four-character PDB accessions surviving
``prepare_pkad3_training_data.py`` are downloaded.  Coordinates come directly
from RCSB, are checked as readable gzip-compressed mmCIF files, and are recorded
in a checksum manifest.  Existing files are never silently trusted: their
current digest and basic parseability are checked on every run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    PROJECT_ROOT
    / "data/processed/protein_environment/pkad3/training_candidates.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/raw/external_datasets/protein_environment/kamls_pkad3/structures"
)
RCSB_URL = "https://files.rcsb.org/download/{pdb}.cif.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_mmcif_gzip(path: Path) -> None:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        # RCSB entries can carry hundreds of kilobytes of audit/entity metadata
        # before the coordinate loop.  A 4 KiB prefix therefore rejects valid
        # files; two MiB comfortably reaches _atom_site in the surveyed cohort
        # without loading large structures in full.
        prefix = handle.read(2 * 1024 * 1024)
    if "data_" not in prefix or "_atom_site." not in prefix:
        raise ValueError("download is not a readable coordinate mmCIF gzip")


def _acquire_one(pdb: str, output_dir: Path) -> Dict[str, object]:
    destination = output_dir / f"{pdb}.cif.gz"
    url = RCSB_URL.format(pdb=pdb)
    if destination.is_file():
        try:
            _validate_mmcif_gzip(destination)
            return {
                "pdb": pdb,
                "url": url,
                "path": str(destination),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
                "status": "verified_existing",
            }
        except (OSError, ValueError):
            destination.unlink()
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temporary)
        _validate_mmcif_gzip(temporary)
        temporary.replace(destination)
        return {
            "pdb": pdb,
            "url": url,
            "path": str(destination),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "status": "downloaded_and_verified",
        }
    except Exception as exc:  # network failures belong in the acquisition ledger
        temporary.unlink(missing_ok=True)
        return {
            "pdb": pdb,
            "url": url,
            "path": str(destination),
            "sha256": "",
            "bytes": 0,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def acquire(
    candidates_path: str | Path = DEFAULT_CANDIDATES,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    workers: int = 8,
) -> Dict[str, object]:
    """Download unique candidate PDB entries and write a complete ledger."""
    candidates_file = Path(candidates_path)
    candidates = pd.read_csv(candidates_file, dtype=str, keep_default_na=False)
    if "pdb" not in candidates:
        raise ValueError("PKAD-3 candidate table lacks the 'pdb' column")
    pdb_ids = sorted(set(candidates["pdb"].str.upper()) - {""})
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
        "schema_version": "pkad3_structure_acquisition_1.0.0",
        "candidate_table": str(candidates_file),
        "candidate_table_sha256": sha256(candidates_file),
        "source": "RCSB Protein Data Bank",
        "url_template": RCSB_URL,
        "requested_pdb_entries": len(pdb_ids),
        "available_pdb_entries": sum(
            row["status"] != "failed" for row in records
        ),
        "failed_pdb_entries": sum(row["status"] == "failed" for row in records),
        "structures": records,
    }
    (target / "acquisition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    report = acquire(args.candidates, args.output_dir, workers=args.workers)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["failed_pdb_entries"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
