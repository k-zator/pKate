#!/usr/bin/env python3
"""Acquire checksum-pinned structures for direct bound-ligand pKa candidates.

Only exact protein/ligand mappings are downloaded.  Proxy structures remain in
the evidence registry but are deliberately ineligible for calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/raw/external_datasets/protein_environment/direct_bound_ligand"
)
RCSB_URL = "https://files.rcsb.org/download/{pdb}.pdb"
STRUCTURES = {
    "3DFR": {
        "sha256": "e960c716aa1eeda43878516cbfa7cd336460031cf14612419fbb0959e183f709",
        "bytes": 190188,
        "role": "methotrexate_l_casei_dhfr_exact_complex",
    },
    "2HM9": {
        "sha256": "6cadb84b254ff49f222981e951f422b50972f6ab6019bd16e95e83ea6dbb41a7",
        "bytes": 6997671,
        "role": "trimethoprim_l_casei_dhfr_exact_complex_first_nmr_model",
    },
    "1PKW": {
        "sha256": "aeef5f71ea402316b946bfab65506e5ead2f66633c5b35b963d28b7311c6eef4",
        "bytes": 380376,
        "role": "glutathione_human_gsta1_exact_complex",
    },
    "1XW5": {
        "sha256": "e5067ba07ec9ea53c8116c165f5f2988b4e4b95c7ab3a7d862ed523e230a8c37",
        "bytes": 369198,
        "role": "glutathione_human_gstm2_exact_complex",
    },
    "5GSS": {
        "sha256": "2f7b1d8f629df26bd7c67370e60c2134e9ae81b9b9bf23fe63e7a3b7fd11025b",
        "bytes": 331938,
        "role": "glutathione_human_gstp1_exact_complex",
    },
    "1TDI": {
        "sha256": "c933a013a5f1a34fdf27ad268bae26abd5986221a9d7337405913334b43c6e83",
        "bytes": 345060,
        "role": "glutathione_human_gsta3_exact_complex",
    },
    "1XW6": {
        "sha256": "3f1f9b8f6820fe43b783a5ef0833348c4ea85f3fd91d6bf47cd46f205cb79caf",
        "bytes": 698058,
        "role": "glutathione_human_gstm1_exact_complex",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(output_dir: str | Path = DEFAULT_OUTPUT) -> Dict[str, object]:
    """Download exact RCSB files and fail if their pinned bytes changed."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for pdb, expected in STRUCTURES.items():
        destination = output / f"{pdb}.pdb"
        if not destination.is_file():
            urllib.request.urlretrieve(RCSB_URL.format(pdb=pdb), destination)
        actual = _sha256(destination)
        size = destination.stat().st_size
        if actual != expected["sha256"] or size != expected["bytes"]:
            raise ValueError(
                f"RCSB checksum/size mismatch for {pdb}: "
                f"sha256={actual}, bytes={size}"
            )
        records.append({
            "pdb": pdb,
            "path": str(destination),
            "download_url": RCSB_URL.format(pdb=pdb),
            "sha256": actual,
            "bytes": size,
            "role": expected["role"],
        })
    manifest = {
        "schema_version": "direct_bound_ligand_structures_1.0.0",
        "source": "RCSB Protein Data Bank",
        "license": "RCSB PDB data usage policy",
        "files": records,
    }
    (output / "acquisition_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(acquire(args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
