#!/usr/bin/env python3
"""Download and checksum the public CASP16 pharmaceutical ligand tables.

Only ``chymasecatgSI.txt`` currently contains experimental pKa columns.  The
autotaxin table is retained because the accompanying inventory says it should
contain pKa data, while the released ``atxSI-2.txt`` file does not.  Keeping
both makes that upstream discrepancy auditable instead of silently treating
predicted ADMET values as experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/raw/external_datasets/casp16"
ZENODO_RECORD = "https://zenodo.org/records/16762332/files"
FILES = {
    "CASP16-PL-DataFile-Inventory.txt": (
        "910519cbdd89af6f914a96e82be047a35de6a2d881c6b8af1fa1d4424e29c6b6"
    ),
    "atxSI-2.txt": (
        "3f70a90e60e3909369e911382fc06e73c57c1967f5048e5bd2cfc01971f86297"
    ),
    "chymasecatgSI.txt": (
        "c58db53e4289d61cdef812724fbeff9e57d890f970e8f339ef5192dbf33a2317"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Download the checksum-locked CASP16 release and write an audit record."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, expected in FILES.items():
        destination = output / name
        url = f"{ZENODO_RECORD}/{name}?download=1"
        if not destination.is_file() or sha256(destination) != expected:
            urllib.request.urlretrieve(url, destination)
        actual = sha256(destination)
        if actual != expected:
            raise ValueError(
                f"CASP16 checksum mismatch for {name}: {actual} != {expected}"
            )
        records.append({
            "filename": name,
            "url": url,
            "sha256": actual,
            "bytes": destination.stat().st_size,
        })
    report = {
        "source": "CASP16 pharmaceutical protein-ligand challenge supporting data",
        "doi": "10.5281/zenodo.16762332",
        "files": records,
        "pka_use_policy": (
            "experimental PKA1/PKA2 columns only; all ADMET Predictor fields are "
            "ignored; PDB IDs associate ligands with complexes but do not make the "
            "reported values bound-state pKas"
        ),
    }
    (output / "acquisition_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(acquire(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
