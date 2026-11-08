#!/usr/bin/env python3
"""Download and checksum the RCSB structures for the locked binding benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = (
    PROJECT_ROOT / "data/benchmarks/binding_induced_pka_structure_mapping.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/raw/external_datasets/protein_environment/binding_shift_structures"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(
    mapping_path: str | Path = DEFAULT_MAPPING,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Acquire every exact coordinate snapshot declared by the mapping."""
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    template = str(mapping["download_url_template"])
    expected = dict(mapping["structure_sha256"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for raw_pdb, expected_digest in sorted(expected.items()):
        pdb = str(raw_pdb).upper()
        destination = output / f"{pdb}.cif"
        status = "verified_existing"
        if not destination.is_file() or sha256(destination) != expected_digest:
            url = template.format(pdb=pdb)
            partial = output / f".{pdb}.cif.part"
            try:
                urllib.request.urlretrieve(url, partial)
                actual = sha256(partial)
                if actual != expected_digest:
                    raise ValueError(
                        f"Checksum mismatch for {pdb}: expected "
                        f"{expected_digest}, received {actual}"
                    )
                partial.replace(destination)
            finally:
                partial.unlink(missing_ok=True)
            status = "downloaded_and_verified"
        records.append({
            "pdb": pdb,
            "path": str(destination),
            "sha256": sha256(destination),
            "status": status,
        })
    return {
        "mapping": str(Path(mapping_path)),
        "output": str(output),
        "structures": records,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(acquire(args.mapping, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
