#!/usr/bin/env python3
"""Acquire the 2026 metalloprotein benchmark coordinates and raw results.

ACS Figshare marks both Supporting Information records CC BY-NC 4.0.  They are
kept outside Git and may be used only as an evaluation/computational-reference
source; neither the coordinates nor calculated pKas are training labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/raw/external_datasets/metalloprotein_pka_2026"
)
SOURCE_DOI = "10.1021/acs.jctc.6c00884"
FILES = {
    "ct6c00884_si_001.zip": {
        "url": "https://ndownloader.figshare.com/files/66370851",
        "md5": "3d207d220ba8f2cce459b6133f1e8e56",
        "sha256": "c32a0c2047ddd87f25c365975bd87f1fe0bbeaf540017132aab2c3ab7a6efae6",
        "figshare_doi": "10.1021/acs.jctc.6c00884.s001",
        "role": "author_prepared_coordinate_archive",
    },
    "ct6c00884_si_002.pdf": {
        "url": "https://ndownloader.figshare.com/files/66370854",
        "md5": "4984b1cde6b64e17abee5a2b9164fcbe",
        "sha256": "e128ff1fe63e99fc1c2479840adec7cc7a5a79bc1c62c37d7d8a6a2a912a4d7e",
        "figshare_doi": "10.1021/acs.jctc.6c00884.s002",
        "role": "raw_calculated_pka_tables",
    },
}


def _digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def acquire(output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, metadata in FILES.items():
        destination = output / name
        if (
            not destination.is_file()
            or _digest(destination, "sha256") != metadata["sha256"]
        ):
            urllib.request.urlretrieve(str(metadata["url"]), destination)
        actual_md5 = _digest(destination, "md5")
        actual_sha256 = _digest(destination, "sha256")
        if actual_md5 != metadata["md5"] or actual_sha256 != metadata["sha256"]:
            raise ValueError(f"Metalloprotein SI checksum mismatch for {name}")
        records.append({
            "filename": name,
            "bytes": destination.stat().st_size,
            "md5": actual_md5,
            "sha256": actual_sha256,
            **metadata,
        })
    report = {
        "source_doi": SOURCE_DOI,
        "license": "CC-BY-NC-4.0",
        "files": records,
        "training_policy": (
            "experimental Table 1 labels remain locked evaluation data; "
            "published calculated pKas are external computational predictions "
            "and are never consumed as pKate training targets"
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
