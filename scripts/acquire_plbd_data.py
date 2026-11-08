#!/usr/bin/env python3
"""Acquire the checksum-locked CC BY 4.0 PLBD tabular snapshot.

The PLBD archive contains experimental compound/protein protonation tables and
binding parameters corrected for linked protonation.  The latter are useful for
thermodynamic consistency checks, but are not measurements of a ligand pKa in
the bound complex.  This acquisition boundary records that distinction before
any table is prepared for pKate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data/raw/external_datasets/plbd"
DOWNLOAD_URL = "https://zenodo.org/api/records/7482058/files/main.zip/content"
SOURCE_DOI = "10.5281/zenodo.7482058"
EXPECTED_MD5 = "4850fe9d581fb42558b08794fa4de72a"
EXPECTED_SHA256 = "48e120d05ef55b34b6efc5bb3cf30091ecb4a76fe4786fdabbbde30ba3cd959e"


def _digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def acquire(output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, object]:
    """Download PLBD once, verify both published MD5 and pinned SHA-256."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "main.zip"
    if (
        not destination.is_file()
        or _digest(destination, "sha256") != EXPECTED_SHA256
    ):
        urllib.request.urlretrieve(DOWNLOAD_URL, destination)
    actual_md5 = _digest(destination, "md5")
    actual_sha256 = _digest(destination, "sha256")
    if actual_md5 != EXPECTED_MD5 or actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            "PLBD checksum mismatch: "
            f"md5={actual_md5}, sha256={actual_sha256}"
        )
    report = {
        "source": "Protein Ligand Binding Database (PLBD) snapshot",
        "doi": SOURCE_DOI,
        "zenodo_record": 7482058,
        "download_url": DOWNLOAD_URL,
        "license": "CC-BY-4.0",
        "filename": destination.name,
        "bytes": destination.stat().st_size,
        "md5": actual_md5,
        "sha256": actual_sha256,
        "evidence_boundary": (
            "compound and protein pKa tables are experimental evidence; "
            "intrinsic binding parameters are protonation-corrected derived "
            "quantities and are not direct bound-ligand pKa labels"
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
