#!/usr/bin/env python3
"""Prepare PLBD protonation and linked-binding evidence without label leakage.

PLBD reports aqueous/assay compound pKas and protein metal-site pKas alongside
intrinsic binding constants from which proton linkage has already been removed.
The joined table produced here is useful for testing thermodynamic consistency.
It is deliberately ineligible for direct ligand-environment fitting because it
does not contain matched free and protein-bound ligand pKas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data/raw/external_datasets/plbd/main.zip"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/plbd"
SOURCE_DOI = "10.5281/zenodo.7482058"
SCHEMA_VERSION = "plbd_proton_linkage_1.0.0"


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _records(archive: zipfile.ZipFile, table: str) -> list[dict[str, Any]]:
    path = f"outputs/main/{table}.json"
    payload = json.loads(archive.read(path).decode("utf-8"))
    records = payload.get("data")
    if not isinstance(records, list):
        raise ValueError(f"PLBD table {table} lacks a JSON:API data list")
    return records


def _nested_id(value: object) -> str:
    if not isinstance(value, dict):
        return str(value)
    identifier = value.get("id")
    if identifier is None:
        identifier = dict(value.get("attributes", {})).get("id")
    return str(identifier)


def _attributes(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(record.get("attributes", {})) for record in records]


def _canonical_smiles(value: object) -> tuple[str, str]:
    mol = Chem.MolFromSmiles(str(value or ""))
    if mol is None:
        return "", "invalid_smiles"
    return Chem.MolToSmiles(mol, isomericSmiles=True), "valid"


def prepare(
    archive_path: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Join PLBD pKa and intrinsic-binding tables with explicit semantics."""
    source = Path(archive_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        compounds = _attributes(_records(archive, "compound"))
        proteins = _attributes(_records(archive, "protein"))
        compound_pkas = _attributes(
            _records(archive, "best_compound_protonation_pka")
        )
        protein_pkas = _attributes(
            _records(archive, "best_protein_protonation_pka")
        )
        intrinsic = _attributes(_records(archive, "intrinsic_parameters"))
        structures = _attributes(_records(archive, "crystal_structure"))

    compound_lookup = {str(row["id"]): row for row in compounds}
    protein_lookup = {str(row["id"]): row for row in proteins}
    compound_rows: list[dict[str, object]] = []
    compound_index: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    excluded_compound_rows: list[dict[str, object]] = []
    for source_row in compound_pkas:
        compound_id = _nested_id(source_row.get("compound_id"))
        compound = compound_lookup.get(compound_id, {})
        smiles, structure_status = _canonical_smiles(compound.get("smiles"))
        pka_source = str(source_row.get("pka_sources", "")).strip().lower()
        row = {
            "schema_version": SCHEMA_VERSION,
            "plbd_compound_pka_id": source_row.get("id"),
            "compound_id": compound_id,
            "compound_label": source_row.get("compound_label", compound.get("label", "")),
            "compound_name": source_row.get("name", compound.get("name", "")),
            "smiles": smiles,
            "structure_status": structure_status,
            "experimental_pka": source_row.get("pka"),
            "experimental_pka_uncertainty": source_row.get("sigpka"),
            "temperature_c": source_row.get("temperature"),
            "pka_source": pka_source,
            "reliability": source_row.get("reliability"),
            "source_doi": SOURCE_DOI,
            "label_origin": "experimental" if pka_source == "experiment" else pka_source,
            "computational_teacher": "",
            "source_measurement_scope": "compound_level_aqueous_or_assay_pka",
            "direct_bound_ligand_label": False,
            "eligible_for_direct_environment_fitting": False,
            "eligible_for_stage1": False,
            "stage1_exclusion_reason": (
                "plbd_has_no_site_resolved_transition_assignment"
                if structure_status == "valid" and pka_source == "experiment"
                else "nonexperimental_or_invalid_structure"
            ),
        }
        if pka_source != "experiment" or structure_status != "valid":
            excluded_compound_rows.append(row)
        compound_rows.append(row)
        if pka_source == "experiment" and structure_status == "valid":
            key = (compound_id, float(source_row["temperature"]))
            compound_index[key].append(row)

    protein_rows: list[dict[str, object]] = []
    protein_index: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for source_row in protein_pkas:
        protein_id = _nested_id(source_row.get("protein_id"))
        protein = protein_lookup.get(protein_id, {})
        pka_source = str(source_row.get("pka_source", "")).strip().lower()
        row = {
            "schema_version": SCHEMA_VERSION,
            "plbd_protein_pka_id": source_row.get("id"),
            "protein_id": protein_id,
            "protein_label": source_row.get("protein_label", protein.get("label", "")),
            "protein_name": source_row.get("protein_name", protein.get("name", "")),
            "primary_group_label": source_row.get("primary_group_label", ""),
            "uniprot": protein.get("uniprot", ""),
            "experimental_pka": source_row.get("pka_zn"),
            "experimental_pka_uncertainty": source_row.get("sigpka_zn"),
            "temperature_c": source_row.get("temperature"),
            "pka_source": pka_source,
            "reliability": source_row.get("reliability"),
            "source_doi": SOURCE_DOI,
            "label_origin": "experimental" if pka_source == "experiment" else pka_source,
            "computational_teacher": "",
            "source_measurement_scope": "protein_metal_site_pka",
            "direct_bound_ligand_label": False,
            "eligible_for_direct_environment_fitting": False,
        }
        protein_rows.append(row)
        if pka_source == "experiment":
            key = (str(row["primary_group_label"]), float(source_row["temperature"]))
            protein_index[key].append(row)

    linkage_rows: list[dict[str, object]] = []
    missing_links: list[dict[str, object]] = []
    for source_row in intrinsic:
        compound_id = _nested_id(source_row.get("compound_id"))
        protein_id = _nested_id(source_row.get("protein_id"))
        temperature = float(source_row["temperature"])
        compound_matches = compound_index.get((compound_id, temperature), [])
        protein_matches = protein_index.get(
            (str(source_row.get("primary_group_label", "")), temperature), []
        )
        if not compound_matches or not protein_matches:
            missing_links.append({
                "intrinsic_parameter_id": source_row.get("id"),
                "compound_id": compound_id,
                "protein_id": protein_id,
                "temperature_c": temperature,
                "reason": "missing_experimental_compound_or_protein_pka_join",
            })
            continue
        compound = compound_lookup.get(compound_id, {})
        protein = protein_lookup.get(protein_id, {})
        compound_values = sorted({float(row["experimental_pka"]) for row in compound_matches})
        protein_values = sorted({float(row["experimental_pka"]) for row in protein_matches})
        linkage_rows.append({
            "schema_version": SCHEMA_VERSION,
            "intrinsic_parameter_id": source_row.get("id"),
            "compound_id": compound_id,
            "compound_name": compound.get("name", ""),
            "compound_smiles": _canonical_smiles(compound.get("smiles"))[0],
            "protein_id": protein_id,
            "protein_name": protein.get("name", ""),
            "protein_uniprot": protein.get("uniprot", ""),
            "primary_group_label": source_row.get("primary_group_label", ""),
            "temperature_c": temperature,
            "compound_experimental_pkas_json": json.dumps(compound_values),
            "protein_experimental_pkas_json": json.dumps(protein_values),
            "intrinsic_binding_constant": source_row.get("kb_int"),
            "intrinsic_binding_free_energy_kj_mol": source_row.get("dbg_int"),
            "intrinsic_binding_enthalpy_kj_mol": source_row.get("dbh_int"),
            "intrinsic_parameter_uncertainty": source_row.get("sigdbg_int"),
            "minimum_reliability": source_row.get("min_q"),
            "replicate_count": source_row.get("n"),
            "source_doi": SOURCE_DOI,
            "label_origin": "experimental_binding_with_derived_protonation_correction",
            "computational_teacher": "",
            "evidence_use": "proton_linkage_consistency_or_locked_evaluation",
            "direct_bound_ligand_label": False,
            "eligible_for_direct_environment_fitting": False,
            "eligible_for_stage1": False,
            "eligible_for_stage2": False,
            "exclusion_reason": (
                "intrinsic_binding_parameter_is_not_a_bound_ligand_pka"
            ),
        })

    structure_rows = []
    for row in structures:
        structure_rows.append({
            "schema_version": SCHEMA_VERSION,
            "pdb": str(row.get("pdb_id", "")).upper(),
            "resolution_angstrom": row.get("resolution"),
            "reliability": row.get("reliability"),
            "source_doi": SOURCE_DOI,
            "coordinate_use": "reference_mapping_only",
            "eligible_for_direct_environment_fitting": False,
        })

    pd.DataFrame(compound_rows).to_csv(output / "compound_pka_evidence.csv", index=False)
    pd.DataFrame(protein_rows).to_csv(output / "protein_pka_evidence.csv", index=False)
    pd.DataFrame(linkage_rows).to_csv(output / "proton_linkage_constraints.csv", index=False)
    pd.DataFrame(structure_rows).to_csv(output / "structure_index.csv", index=False)
    pd.DataFrame(excluded_compound_rows).to_csv(
        output / "compound_pka_exclusions.csv", index=False
    )
    (output / "linkage_join_exclusions.json").write_text(
        json.dumps(missing_links, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_doi": SOURCE_DOI,
        "source_sha256": _sha256(source),
        "license": "CC-BY-4.0",
        "compound_rows": len(compound_rows),
        "experimental_compound_rows": int(sum(
            row["label_origin"] == "experimental" for row in compound_rows
        )),
        "experimental_compounds": int(len({
            row["compound_id"] for row in compound_rows
            if row["label_origin"] == "experimental"
        })),
        "protein_pka_rows": len(protein_rows),
        "intrinsic_parameter_rows": len(intrinsic),
        "proton_linkage_rows": len(linkage_rows),
        "unjoined_intrinsic_rows": len(missing_links),
        "crystal_structure_rows": len(structure_rows),
        "direct_bound_ligand_calibration_rows": 0,
        "training_policy": (
            "no PLBD row is treated as a bound-ligand pKa; intrinsic binding "
            "parameters remain consistency/evaluation evidence only"
        ),
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.archive, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
