#!/usr/bin/env python3
"""Audit and benchmark the experimental protein-environment correction.

PKAD-2 is a predecessor of PKAD-R, so an ordinary score on the complete
PKAD-2 table would be leakage, not external validation.  This module builds a
row-level overlap ledger, reports a measurement-deduplicated portability
check, and separately identifies whether any near-duplicate-disjoint rows are
left.  It never fits or changes a model.

The binding-shift panel is a small evaluation-only manifest of experimental
measurements.  Censored values and coupled titration midpoints retain their
native semantics and are not silently converted to ordinary point labels.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence
from urllib.parse import unquote

import numpy as np
import pandas as pd

from protein_environment_correction import (
    DEFAULT_ENVIRONMENT_MODEL,
    PROTEIN_RESIDUE_CHEMISTRY,
    PROTEIN_RESIDUE_REFERENCE_PKA,
    _load_bundle,
    _prediction_for_site,
    environment_descriptors,
    read_structure_atoms,
    residue_site_center,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PKAD2 = (
    PROJECT_ROOT
    / "data/raw/external_datasets/protein_environment/revpkad2/"
    "process_PKAD2_DOWNLOAD_rev.xlsx"
)
DEFAULT_PKAD2_STRUCTURES = DEFAULT_PKAD2.parent / "pdbs"
DEFAULT_PKADR = (
    PROJECT_ROOT
    / "data/raw/external_datasets/protein_environment/pkadr/PKAD-R-250211.json"
)
DEFAULT_PKADR_STRUCTURES = DEFAULT_PKADR.parent / "structures"
DEFAULT_BINDING_PANEL = (
    PROJECT_ROOT / "data/benchmarks/binding_induced_pka_shifts.csv"
)
DEFAULT_BINDING_MAPPING = (
    PROJECT_ROOT / "data/benchmarks/binding_induced_pka_structure_mapping.json"
)
DEFAULT_BINDING_STRUCTURES = (
    PROJECT_ROOT
    / "data/raw/external_datasets/protein_environment/binding_shift_structures"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/processed/ml_models_experimental_only/protein_environment/benchmark"
)
SUPPORTED_RESIDUES = frozenset(PROTEIN_RESIDUE_REFERENCE_PKA)
AMINO_ACID_CODE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def sha256(path: str | Path) -> str:
    """Return a file digest used to make training/evaluation identity explicit."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_float(value: object) -> float | None:
    """Accept finite point values while rejecting censored/approximate strings."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and any(token in value for token in "<>~=±"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clean_text(value: object) -> str:
    """Normalize spreadsheet nulls without turning NaN into an identifier."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def residue_id(value: object) -> str:
    """Normalize Excel's integral residue numbers without losing insertions."""
    text = clean_text(value)
    if re.fullmatch(r"[-+]?\d+[.]0+", text):
        return text.split(".", 1)[0]
    return text


def normalize_reference(value: object) -> str:
    """Prefer DOI or PMID identity, falling back to a normalized URL/string."""
    text = unquote(clean_text(value)).lower()
    doi = re.search(r"10\.\d{4,9}/[^\s?#]+", text)
    if doi:
        return doi.group(0).rstrip(".,;)")
    pmid = re.search(
        r"(?:pubmed(?:\.ncbi\.nlm\.nih\.gov)?[/=:]|pubmed/)(\d+)", text
    )
    if pmid:
        return f"pmid:{pmid.group(1)}"
    return re.sub(r"\s+", " ", text).rstrip("/")


def _strict_pkadr(rows: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    selected = []
    for row in rows:
        if (
            clean_text(row.get("pKa_Classification")) == "Main"
            and not clean_text(row.get("Warning"))
            and clean_text(row.get("ResName")).upper() in SUPPORTED_RESIDUES
            and exact_float(row.get("Expt_pKa")) is not None
        ):
            selected.append(row)
    return selected


def _linked_contexts(rows: Iterable[Mapping[str, object]]) -> set[tuple[str, str]]:
    linked: set[tuple[str, str]] = set()
    for row in rows:
        for token in re.split(
            r"[,;\s]+", clean_text(row.get("Sequence_Identity_>_30%"))
        ):
            match = re.fullmatch(r"([A-Za-z0-9]{4})\.([A-Za-z0-9]+)", token)
            if match:
                linked.add((match.group(1).upper(), match.group(2)))
    return linked


def read_pkad2_wild_type(path: str | Path) -> pd.DataFrame:
    """Read the revised PKAD-2 wild-type table from XLSX or a test CSV."""
    source = Path(path)
    if source.suffix.lower() == ".csv":
        table = pd.read_csv(source)
    else:
        try:
            table = pd.read_excel(source, sheet_name="Wild Type")
        except ImportError as exc:
            raise RuntimeError(
                "Reading PKAD-2 XLSX requires openpyxl (included in pKate's environment)."
            ) from exc
    required = {"PDB ID", "Res Name", "Chain", "Res ID", "pKa", "Reference"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"PKAD-2 table is missing columns: {sorted(missing)}")
    return table


def build_pkad2_overlap_ledger(
    pkad2: pd.DataFrame,
    pkadr_rows: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    """Classify overlap before any prediction or metric is calculated."""
    strict = _strict_pkadr(pkadr_rows)
    training_sites = {
        (
            clean_text(row.get("PDB")).upper(),
            clean_text(row.get("Chain")),
            clean_text(row.get("ResName")).upper(),
            residue_id(row.get("ResID_in_PDB")),
        )
        for row in strict
    }
    training_contexts = {(key[0], key[1]) for key in training_sites}
    training_pdbs = {key[0] for key in training_sites}
    linked_contexts = _linked_contexts(strict)
    training_references = {
        normalize_reference(row.get("Reference"))
        for row in strict
        if normalize_reference(row.get("Reference"))
    }
    records = []
    for source_index, row in pkad2.iterrows():
        pdb = clean_text(row["PDB ID"]).upper()
        chain = clean_text(row["Chain"])
        name = clean_text(row["Res Name"]).upper()
        rid = residue_id(row["Res ID"])
        pka = exact_float(row["pKa"])
        reference = normalize_reference(row["Reference"])
        supported = name in SUPPORTED_RESIDUES
        exact = pka is not None
        site = (pdb, chain, name, rid)
        context = (pdb, chain)
        flags = {
            "overlap_exact_training_site": site in training_sites,
            "overlap_training_context": context in training_contexts,
            "overlap_training_pdb": pdb in training_pdbs,
            "overlap_pkadr_30pct_link": context in linked_contexts,
            "overlap_training_reference": bool(
                reference and reference in training_references
            ),
        }
        reasons = [key.removeprefix("overlap_") for key, value in flags.items() if value]
        portability = supported and exact and not reasons
        measurement_key = (
            f"{reference}|{name}|{rid}|{float(pka):.8g}" if portability else ""
        )
        records.append({
            "pkad2_source_row": int(source_index) + 2,
            "pdb": pdb,
            "chain": chain,
            "residue_name": name,
            "residue_id": rid,
            "experimental_pka": pka,
            "reference": clean_text(row["Reference"]),
            "normalized_reference": reference,
            "supported_residue": supported,
            "exact_numeric_pka": exact,
            **flags,
            "overlap_reasons": ";".join(reasons),
            "portability_eligible": portability,
            "measurement_key": measurement_key,
            "label_origin": "experimental",
            "computational_teacher": "",
            "evaluation_only": True,
            "eligible_for_training": False,
        })
    ledger = pd.DataFrame(records)
    ledger["portability_selected"] = False
    selected = ledger[ledger["portability_eligible"]].drop_duplicates(
        "measurement_key", keep="first"
    )
    ledger.loc[selected.index, "portability_selected"] = True
    ledger["duplicate_measurement_row"] = (
        ledger["portability_eligible"] & ~ledger["portability_selected"]
    )
    return ledger


def validate_evaluation_only_labels(table: pd.DataFrame, *, name: str) -> None:
    """Fail closed if computational labels or training-eligible rows enter a test."""
    required = {
        "label_origin", "computational_teacher", "evaluation_only",
        "eligible_for_training",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{name} lacks provenance columns: {sorted(missing)}")
    origin = table["label_origin"].map(clean_text).str.lower()
    teacher = table["computational_teacher"].map(clean_text)
    evaluation = table["evaluation_only"].map(_truthy)
    training = table["eligible_for_training"].map(_truthy)
    invalid = (origin != "experimental") | teacher.ne("") | ~evaluation | training
    if bool(invalid.any()):
        rows = table.index[invalid].tolist()[:10]
        raise ValueError(
            f"{name} contains non-experimental, teacher-derived, or training-eligible "
            f"labels at rows {rows}"
        )


def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean_text(value).lower() in {"1", "true", "yes", "y"}


def _chain_sequences(atoms) -> Dict[str, str]:
    result: Dict[str, list[str]] = {}
    seen = set()
    for atom in atoms:
        key = (atom.chain, atom.residue_id, atom.insertion_code)
        if (
            atom.group == "ATOM"
            and atom.residue_name in AMINO_ACID_CODE
            and key not in seen
        ):
            seen.add(key)
            result.setdefault(atom.chain, []).append(AMINO_ACID_CODE[atom.residue_name])
    return {chain: "".join(sequence) for chain, sequence in result.items()}


def add_near_duplicate_audit(
    ledger: pd.DataFrame,
    pkadr_rows: Sequence[Mapping[str, object]],
    *,
    pkad2_structures: str | Path,
    pkadr_structures: str | Path,
    threshold: float = 0.80,
    minimum_chain_residues: int = 30,
) -> pd.DataFrame:
    """Exclude near-identical chains using a conservative sequence-similarity proxy.

    The ratio is Python SequenceMatcher similarity, not a formal alignment identity.
    It is intentionally used only at the high 0.80 near-duplicate threshold.  The
    PKAD-R supplied >30% links are enforced separately in the overlap ledger.
    """
    result = ledger.copy()
    for column, default in (
        ("maximum_training_sequence_similarity", np.nan),
        ("nearest_training_sequence_context", ""),
        ("candidate_chain_residues", 0),
        ("sequence_audit_status", "not_selected"),
        ("near_duplicate_disjoint_eligible", False),
    ):
        result[column] = default
    strict = _strict_pkadr(pkadr_rows)
    training_contexts = sorted({
        (clean_text(row.get("PDB")).upper(), clean_text(row.get("Chain")))
        for row in strict
    })
    training_sequences = []
    cache: Dict[str, Dict[str, str]] = {}
    for pdb, chain in training_contexts:
        try:
            if pdb not in cache:
                cache[pdb] = _chain_sequences(read_structure_atoms(
                    Path(pkadr_structures) / f"{pdb}.cif.gz"
                ))
            sequence = cache[pdb].get(chain, "")
        except (FileNotFoundError, ValueError):
            sequence = ""
        if sequence:
            training_sequences.append((pdb, chain, sequence))
    candidate_cache: Dict[str, Dict[str, str]] = {}
    for index, row in result[result["portability_selected"]].iterrows():
        pdb, chain = str(row["pdb"]), str(row["chain"])
        try:
            if pdb not in candidate_cache:
                candidate_cache[pdb] = _chain_sequences(read_structure_atoms(
                    Path(pkad2_structures) / f"{pdb.lower()}.pdb"
                ))
            sequence = candidate_cache[pdb].get(chain, "")
        except (FileNotFoundError, ValueError) as exc:
            result.at[index, "sequence_audit_status"] = f"unavailable:{exc}"
            continue
        if not sequence:
            result.at[index, "sequence_audit_status"] = "target_chain_sequence_unavailable"
            continue
        result.at[index, "candidate_chain_residues"] = int(len(sequence))
        if len(sequence) < int(minimum_chain_residues):
            result.at[index, "sequence_audit_status"] = "target_chain_sequence_too_short"
            continue
        matches = [
            (
                difflib.SequenceMatcher(
                    None, sequence, training_sequence, autojunk=False
                ).ratio(),
                training_pdb,
                training_chain,
            )
            for training_pdb, training_chain, training_sequence in training_sequences
        ]
        similarity, nearest_pdb, nearest_chain = max(matches, default=(0.0, "", ""))
        result.at[index, "maximum_training_sequence_similarity"] = float(similarity)
        result.at[index, "nearest_training_sequence_context"] = (
            f"{nearest_pdb}.{nearest_chain}" if nearest_pdb else ""
        )
        near_duplicate = similarity >= float(threshold)
        result.at[index, "sequence_audit_status"] = (
            "near_duplicate_training_chain" if near_duplicate else "disjoint_at_proxy_threshold"
        )
        result.at[index, "near_duplicate_disjoint_eligible"] = not near_duplicate
    return result


def predict_pkad2_portability(
    ledger: pd.DataFrame,
    *,
    structures_dir: str | Path,
    model_path: str | Path,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Predict the de-overlapped, measurement-deduplicated PKAD-2 rows."""
    validate_evaluation_only_labels(ledger, name="PKAD-2 ledger")
    bundle = _load_bundle(model_path)
    predictions, exclusions = [], []
    structure_cache = {}
    for _, row in ledger[ledger["portability_selected"]].iterrows():
        pdb = str(row["pdb"])
        try:
            if pdb not in structure_cache:
                structure_cache[pdb] = read_structure_atoms(
                    Path(structures_dir) / f"{pdb.lower()}.pdb"
                )
            atoms = structure_cache[pdb]
            center, key = residue_site_center(
                atoms,
                chain=str(row["chain"]),
                residue_id=str(row["residue_id"]),
                residue_name=str(row["residue_name"]),
            )
            chemistry = PROTEIN_RESIDUE_CHEMISTRY[str(row["residue_name"])]
            descriptors = environment_descriptors(
                atoms,
                center,
                chemistry=chemistry,
                excluded_residue_keys=[key],
            )
            prediction = _prediction_for_site(
                bundle,
                descriptors,
                chemistry=chemistry,
                transfer_factor=1.0,
                direct_protein_site=True,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            exclusions.append({
                "pdb": pdb,
                "chain": str(row["chain"]),
                "residue_name": str(row["residue_name"]),
                "residue_id": str(row["residue_id"]),
                "reason": str(exc),
            })
            continue
        reference = float(PROTEIN_RESIDUE_REFERENCE_PKA[str(row["residue_name"])])
        experimental = float(row["experimental_pka"])
        gated = reference + float(prediction["delta_pka"])
        raw = reference + float(prediction["clipped_protein_residue_delta_pka"])
        radius = float(prediction["delta_pka_p90_error"])
        predictions.append({
            **row.to_dict(),
            "chemistry": chemistry,
            "reference_pka_prediction": reference,
            "raw_environment_prediction": raw,
            "gated_environment_prediction": gated,
            "gated_ci90_low": gated - radius,
            "gated_ci90_high": gated + radius,
            "gated_ci90_contains_experiment": gated - radius <= experimental <= gated + radius,
            "reference_absolute_error": abs(reference - experimental),
            "raw_environment_absolute_error": abs(raw - experimental),
            "gated_environment_absolute_error": abs(gated - experimental),
            "representation_correction_scale": prediction[
                "representation_correction_scale"
            ],
            "representation_calibration_tier": prediction[
                "representation_calibration_tier"
            ],
            "prediction_confidence_score": prediction["confidence_score"],
            "prediction_confidence_tier": prediction["confidence_tier"],
        })
    return pd.DataFrame(predictions), exclusions


def _metric_block(table: pd.DataFrame) -> dict[str, object]:
    observed = table["experimental_pka"].to_numpy(dtype=float)
    result: dict[str, object] = {"rows": int(len(table))}
    if not len(table):
        return result
    for label, column in (
        ("reference", "reference_pka_prediction"),
        ("raw_environment", "raw_environment_prediction"),
        ("gated_environment", "gated_environment_prediction"),
    ):
        predicted = table[column].to_numpy(dtype=float)
        error = predicted - observed
        result[f"{label}_mae"] = float(np.mean(np.abs(error)))
        result[f"{label}_rmse"] = float(np.sqrt(np.mean(error ** 2)))
        result[f"{label}_bias"] = float(np.mean(error))
        result[f"{label}_p90_absolute_error"] = float(
            np.quantile(np.abs(error), 0.90)
        )
    result["gated_mae_gain_over_reference"] = (
        result["reference_mae"] - result["gated_environment_mae"]
    )
    result["raw_mae_gain_over_reference"] = (
        result["reference_mae"] - result["raw_environment_mae"]
    )
    result["gated_interval_nominal_coverage"] = 0.90
    result["gated_interval_empirical_coverage"] = float(
        table["gated_ci90_contains_experiment"].mean()
    )
    return result


def pkad2_metrics(predictions: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, object]:
    """Summarize portability and any genuinely near-duplicate-disjoint subset."""
    overall = _metric_block(predictions)
    by_chemistry = {
        str(name): _metric_block(group)
        for name, group in predictions.groupby("chemistry")
    }
    disjoint = predictions[
        predictions["near_duplicate_disjoint_eligible"].map(_truthy)
    ]
    return {
        "interpretation": (
            "PKAD-2 portability check after exact-site/PDB/reference filtering; "
            "not an independent external benchmark; the 0.80 similarity proxy only "
            "identifies obvious near-duplicates"
        ),
        "pkad2_rows": int(len(ledger)),
        "supported_exact_rows": int((
            ledger["supported_residue"] & ledger["exact_numeric_pka"]
        ).sum()),
        "portability_eligible_source_rows": int(ledger["portability_eligible"].sum()),
        "portability_selected_unique_measurements": int(
            ledger["portability_selected"].sum()
        ),
        "near_duplicate_disjoint_selected_rows": int(len(disjoint)),
        "portability_metrics": overall,
        "portability_metrics_by_chemistry": by_chemistry,
        "near_duplicate_disjoint_metrics": _metric_block(disjoint),
    }


def read_binding_panel(path: str | Path) -> pd.DataFrame:
    """Load and validate the locked experimental binding-shift manifest."""
    panel = pd.read_csv(path, keep_default_na=False)
    required = {
        "case_id", "experimental_delta_pka", "qualifier", "label_semantics",
        "label_origin", "computational_teacher", "evaluation_only",
        "eligible_for_training",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Binding-shift panel is missing columns: {sorted(missing)}")
    validate_evaluation_only_labels(panel, name="binding-shift panel")
    allowed = {"exact", "lower_bound", "upper_bound"}
    unexpected = set(panel["qualifier"]) - allowed
    if unexpected:
        raise ValueError(f"Unsupported binding-shift qualifiers: {sorted(unexpected)}")
    if panel["case_id"].duplicated().any():
        raise ValueError("Binding-shift case IDs must be unique")
    return panel


def read_binding_mapping(
    path: str | Path,
    panel: pd.DataFrame,
) -> dict[str, object]:
    """Load the case-specific structural mapping and require complete coverage."""
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(mapping.get("schema_version")) != "1.0.0":
        raise ValueError("Binding-shift structure mapping schema is incompatible")
    cases = list(mapping.get("cases", []))
    case_ids = [clean_text(case.get("case_id")) for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Binding-shift structure mapping has duplicate case IDs")
    expected = set(panel["case_id"].map(clean_text))
    actual = set(case_ids)
    if expected != actual:
        raise ValueError(
            "Binding-shift structure mapping does not exactly cover the panel: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    for case in cases:
        for state_name in ("apo", "holo"):
            state = dict(case.get(state_name, {}))
            if not clean_text(state.get("pdb")) or not state.get("target_groups"):
                raise ValueError(
                    f"{case['case_id']} has incomplete {state_name} structure mapping"
                )
            for group in state["target_groups"]:
                if not group or len(group) > 2:
                    raise ValueError(
                        f"{case['case_id']} target groups must contain one or two sites"
                    )
                for site in group:
                    if not all(
                        clean_text(site.get(key))
                        for key in ("chain", "residue_name", "residue_id")
                    ):
                        raise ValueError(
                            f"{case['case_id']} contains an incomplete target site"
                        )
    return mapping


def _mapped_residue_key(site: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        clean_text(site.get("chain")),
        residue_id(site.get("residue_id")),
        clean_text(site.get("insertion_code")),
        clean_text(site.get("residue_name")).upper(),
    )


def _verify_binding_structures(
    mapping: Mapping[str, object],
    structures_dir: str | Path,
) -> dict[str, Path]:
    """Resolve and checksum every RCSB file before calculating predictions."""
    structures = Path(structures_dir)
    expected = dict(mapping.get("structure_sha256", {}))
    paths: dict[str, Path] = {}
    for pdb, digest in expected.items():
        pdb = str(pdb).upper()
        candidates = [structures / f"{pdb}.cif", structures / f"{pdb.lower()}.cif"]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing binding-shift structure {pdb}.cif")
        actual = sha256(path)
        if actual != str(digest):
            raise ValueError(
                f"Binding-shift structure checksum mismatch for {pdb}: {actual}"
            )
        paths[pdb] = path
    return paths


def _coupled_midpoint(
    site_predictions: Sequence[Mapping[str, object]],
    coupling_parameters: Mapping[str, object],
) -> tuple[float, dict[str, float | str]]:
    """Return the midpoint of the average two-site protonation probability."""
    if len(site_predictions) == 1:
        return float(site_predictions[0]["predicted_pka"]), {
            "method": "individual_site_midpoint",
            "coupling_log10_units": 0.0,
        }
    if len(site_predictions) != 2:
        raise ValueError("Only individual or two-site coupled targets are supported")
    left, right = site_predictions
    left_pka = float(left["predicted_pka"])
    right_pka = float(right["predicted_pka"])
    distance = float(np.linalg.norm(
        np.asarray(left["center"], dtype=float)
        - np.asarray(right["center"], dtype=float)
    ))
    dielectric = float(coupling_parameters["dielectric"])
    debye = float(coupling_parameters["debye_length_angstrom"])
    cap = float(coupling_parameters["maximum_absolute_coupling_log10"])
    reference_ph = float(coupling_parameters["centering_ph"])
    # Keep this expression synchronized with the explicit Stage 3 local network.
    distance_safe = max(distance, 1.5)
    delta_g = (
        138.935456
        * math.exp(-distance_safe / debye)
        / (dielectric * distance_safe)
    )
    coupling = float(np.clip(
        -delta_g / (8.31446261815324e-3 * 298.15 * math.log(10.0)),
        -cap,
        cap,
    ))
    occupancies = [
        1.0 / (1.0 + 10.0 ** (reference_ph - value))
        for value in (left_pka, right_pka)
    ]
    one_body = (
        left_pka - coupling * occupancies[1],
        right_pka - coupling * occupancies[0],
    )

    def average_occupancy(ph: float) -> float:
        log_weights = np.asarray([
            0.0,
            one_body[0] - ph,
            one_body[1] - ph,
            one_body[0] + one_body[1] + coupling - (2.0 * ph),
        ])
        high = float(np.max(log_weights))
        weights = np.power(10.0, log_weights - high)
        return float((weights[1] + weights[2] + 2.0 * weights[3]) / (2.0 * weights.sum()))

    low, high = -15.0, 30.0
    for _ in range(120):
        midpoint = (low + high) / 2.0
        if average_occupancy(midpoint) > 0.5:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0, {
        "method": "two_site_centered_screened_coulomb_partition_midpoint",
        "distance_angstrom": distance,
        "coupling_log10_units": coupling,
        "dielectric": dielectric,
        "debye_length_angstrom": debye,
        "centering_ph": reference_ph,
    }


def predict_binding_shift_panel(
    panel: pd.DataFrame,
    mapping: Mapping[str, object],
    *,
    structures_dir: str | Path,
    model_path: str | Path,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Generate frozen-model apo/holo shift predictions for the mapped panel."""
    validate_evaluation_only_labels(panel, name="binding-shift panel")
    paths = _verify_binding_structures(mapping, structures_dir)
    bundle = _load_bundle(model_path)
    coupling = dict(mapping["coupling_parameters"])
    structure_cache = {
        pdb: read_structure_atoms(path) for pdb, path in paths.items()
    }
    panel_lookup = panel.set_index("case_id", drop=False)
    predictions: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for case in mapping["cases"]:
        case_id = str(case["case_id"])
        observation = panel_lookup.loc[case_id]
        try:
            state_results = {}
            for state_name in ("apo", "holo"):
                state = dict(case[state_name])
                pdb = str(state["pdb"]).upper()
                removals = {
                    _mapped_residue_key(value)
                    for value in state.get("remove_residues", [])
                }
                state_atoms = [
                    atom for atom in structure_cache[pdb]
                    if atom.residue_key not in removals
                ]
                group_midpoints = []
                group_details = []
                uncertainty_radii = []
                for group in state["target_groups"]:
                    site_predictions = []
                    for target in group:
                        residue_name = clean_text(target["residue_name"]).upper()
                        chemistry = PROTEIN_RESIDUE_CHEMISTRY[residue_name]
                        center, key = residue_site_center(
                            state_atoms,
                            chain=clean_text(target["chain"]),
                            residue_id=residue_id(target["residue_id"]),
                            residue_name=residue_name,
                        )
                        descriptors = environment_descriptors(
                            state_atoms,
                            center,
                            chemistry=chemistry,
                            excluded_residue_keys=[key],
                        )
                        residual = _prediction_for_site(
                            bundle,
                            descriptors,
                            chemistry=chemistry,
                            transfer_factor=1.0,
                            direct_protein_site=True,
                        )
                        predicted_pka = (
                            float(PROTEIN_RESIDUE_REFERENCE_PKA[residue_name])
                            + float(residual["delta_pka"])
                        )
                        uncertainty = float(residual["delta_pka_p90_error"])
                        uncertainty_radii.append(uncertainty)
                        site_predictions.append({
                            "chain": clean_text(target["chain"]),
                            "residue_name": residue_name,
                            "residue_id": residue_id(target["residue_id"]),
                            "predicted_pka": predicted_pka,
                            "p90_error_radius": uncertainty,
                            "center": [float(value) for value in center],
                            "representation_correction_scale": float(
                                residual["representation_correction_scale"]
                            ),
                        })
                    midpoint, coupling_details = _coupled_midpoint(
                        site_predictions, coupling
                    )
                    group_midpoints.append(midpoint)
                    group_details.append({
                        "sites": site_predictions,
                        "midpoint_pka": midpoint,
                        "thermodynamics": coupling_details,
                    })
                state_results[state_name] = {
                    "pdb": pdb,
                    "predicted_pka": float(np.mean(group_midpoints)),
                    "maximum_site_p90_error_radius": float(max(uncertainty_radii)),
                    "target_group_count": int(len(group_midpoints)),
                    "removed_residue_keys": [list(value) for value in sorted(removals)],
                    "groups": group_details,
                }
            predicted_delta = (
                float(state_results["holo"]["predicted_pka"])
                - float(state_results["apo"]["predicted_pka"])
            )
            shift_radius = float(math.sqrt(
                float(state_results["apo"]["maximum_site_p90_error_radius"]) ** 2
                + float(state_results["holo"]["maximum_site_p90_error_radius"]) ** 2
            ))
            predictions.append({
                "case_id": case_id,
                "predicted_delta_pka": predicted_delta,
                "prediction_semantics": str(observation["label_semantics"]),
                "predicted_apo_pka": state_results["apo"]["predicted_pka"],
                "predicted_holo_pka": state_results["holo"]["predicted_pka"],
                "predicted_delta_pka_p90_error_radius": shift_radius,
                "predicted_delta_pka_ci90_low": predicted_delta - shift_radius,
                "predicted_delta_pka_ci90_high": predicted_delta + shift_radius,
                "mapping_status": str(case["mapping_status"]),
                "mapping_notes": str(case["mapping_notes"]),
                "prediction_source": "frozen_pkadr_protein_environment_model",
                "prediction_details_json": json.dumps(
                    state_results, sort_keys=True, separators=(",", ":")
                ),
            })
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            exclusions.append({
                "case_id": case_id,
                "reason": str(exc),
                "mapping_status": str(case.get("mapping_status", "")),
            })
    return pd.DataFrame(predictions), exclusions


def score_binding_predictions(
    panel: pd.DataFrame,
    submitted: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score exact labels and inequality satisfaction without mixing semantics."""
    required = {"case_id", "predicted_delta_pka", "prediction_semantics"}
    missing = required - set(submitted.columns)
    if missing:
        raise ValueError(f"Binding predictions are missing columns: {sorted(missing)}")
    if submitted["case_id"].duplicated().any():
        raise ValueError("Binding predictions contain duplicate case IDs")
    merged = panel.merge(submitted, on="case_id", how="left", validate="one_to_one")
    merged["prediction_available"] = merged["predicted_delta_pka"].notna()
    wrong_semantics = (
        merged["prediction_available"]
        & merged["prediction_semantics"].ne(merged["label_semantics"])
    )
    if bool(wrong_semantics.any()):
        raise ValueError(
            "Prediction semantics do not match the experimental target for: "
            + ", ".join(merged.loc[wrong_semantics, "case_id"])
        )
    target = merged["experimental_delta_pka"].astype(float)
    predicted = pd.to_numeric(merged["predicted_delta_pka"], errors="coerce")
    exact = merged["prediction_available"] & merged["qualifier"].eq("exact")
    lower = merged["prediction_available"] & merged["qualifier"].eq("lower_bound")
    upper = merged["prediction_available"] & merged["qualifier"].eq("upper_bound")
    merged["absolute_error"] = np.where(exact, np.abs(predicted - target), np.nan)
    merged["bound_satisfied"] = np.where(
        lower, predicted > target, np.where(upper, predicted < target, np.nan)
    )
    merged["bound_violation"] = np.where(
        lower,
        np.maximum(target - predicted, 0.0),
        np.where(upper, np.maximum(predicted - target, 0.0), np.nan),
    )
    exact_error = merged.loc[exact, "absolute_error"].astype(float)
    censored = merged.loc[lower | upper]
    metrics = {
        "panel_rows": int(len(panel)),
        "exact_label_rows": int(panel["qualifier"].eq("exact").sum()),
        "censored_label_rows": int(panel["qualifier"].ne("exact").sum()),
        "predictions_submitted": int(merged["prediction_available"].sum()),
        "exact_predictions_scored": int(len(exact_error)),
        "exact_mae": float(exact_error.mean()) if len(exact_error) else None,
        "exact_rmse": (
            float(np.sqrt(np.mean(exact_error.to_numpy() ** 2)))
            if len(exact_error) else None
        ),
        "censored_predictions_scored": int(len(censored)),
        "censored_bound_satisfaction": (
            float(censored["bound_satisfied"].astype(float).mean())
            if len(censored) else None
        ),
    }
    return merged, metrics


def run(
    *,
    pkad2_path: str | Path = DEFAULT_PKAD2,
    pkad2_structures: str | Path = DEFAULT_PKAD2_STRUCTURES,
    pkadr_path: str | Path = DEFAULT_PKADR,
    pkadr_structures: str | Path = DEFAULT_PKADR_STRUCTURES,
    model_path: str | Path = PROJECT_ROOT / DEFAULT_ENVIRONMENT_MODEL,
    binding_panel_path: str | Path = DEFAULT_BINDING_PANEL,
    binding_mapping_path: str | Path = DEFAULT_BINDING_MAPPING,
    binding_structures: str | Path = DEFAULT_BINDING_STRUCTURES,
    binding_predictions_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, object]:
    """Run leakage audit, PKAD-2 portability scoring, and panel validation."""
    pkadr_path = Path(pkadr_path)
    model_path = Path(model_path)
    bundle = _load_bundle(model_path)
    training_digest = str(bundle["training_source_sha256"])
    actual_training_digest = sha256(pkadr_path)
    if training_digest != actual_training_digest:
        raise ValueError(
            "Model training-source hash does not match the supplied PKAD-R snapshot"
        )
    if sha256(pkad2_path) == training_digest:
        raise ValueError("Evaluation data are byte-identical to the model training source")
    pkadr_rows = json.loads(pkadr_path.read_text(encoding="utf-8"))
    pkad2 = read_pkad2_wild_type(pkad2_path)
    ledger = build_pkad2_overlap_ledger(pkad2, pkadr_rows)
    ledger = add_near_duplicate_audit(
        ledger,
        pkadr_rows,
        pkad2_structures=pkad2_structures,
        pkadr_structures=pkadr_structures,
    )
    predictions, exclusions = predict_pkad2_portability(
        ledger,
        structures_dir=pkad2_structures,
        model_path=model_path,
    )
    metrics = pkad2_metrics(predictions, ledger)
    panel = read_binding_panel(binding_panel_path)
    binding_mapping = read_binding_mapping(binding_mapping_path, panel)
    binding_metrics: dict[str, object] = {
        "panel_rows": int(len(panel)),
        "exact_label_rows": int(panel["qualifier"].eq("exact").sum()),
        "censored_label_rows": int(panel["qualifier"].ne("exact").sum()),
        "coupled_target_rows": int(
            panel["label_semantics"].eq("combined_titration_midpoint_shift").sum()
        ),
        "status": "locked_evaluation_only_manifest_and_mapping_validated",
    }
    scored_binding = None
    model_binding_predictions = None
    binding_exclusions: list[dict[str, object]] = []
    binding_structure_dir = Path(binding_structures)
    if binding_structure_dir.is_dir():
        model_binding_predictions, binding_exclusions = predict_binding_shift_panel(
            panel,
            binding_mapping,
            structures_dir=binding_structure_dir,
            model_path=model_path,
        )
        scored_binding, binding_metrics = score_binding_predictions(
            panel, model_binding_predictions
        )
        binding_metrics.update({
            "status": "frozen_model_structural_benchmark_completed",
            "mapped_predictions": int(len(model_binding_predictions)),
            "prediction_exclusions": int(len(binding_exclusions)),
            "mapping_caveat_rows": int(
                model_binding_predictions["mapping_status"]
                .str.contains("caveat|convention", regex=True)
                .sum()
            ) if len(model_binding_predictions) else 0,
        })
        exact_mask = scored_binding["qualifier"].eq("exact") & scored_binding[
            "prediction_available"
        ]
        if bool(exact_mask.any()):
            exact_rows = scored_binding.loc[exact_mask]
            binding_metrics["exact_delta_ci90_empirical_coverage"] = float((
                exact_rows["experimental_delta_pka"].astype(float)
                >= exact_rows["predicted_delta_pka_ci90_low"].astype(float)
            ).mul(
                exact_rows["experimental_delta_pka"].astype(float)
                <= exact_rows["predicted_delta_pka_ci90_high"].astype(float)
            ).mean())

    submitted_scored = None
    submitted_metrics = None
    if binding_predictions_path is not None:
        submitted = pd.read_csv(binding_predictions_path)
        submitted_scored, submitted_metrics = score_binding_predictions(panel, submitted)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output / "pkad2_overlap_ledger.csv", index=False)
    predictions.to_csv(output / "pkad2_portability_predictions.csv", index=False)
    (output / "pkad2_prediction_exclusions.json").write_text(
        json.dumps(exclusions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "pkad2_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if model_binding_predictions is not None:
        model_binding_predictions.to_csv(
            output / "binding_shift_model_predictions.csv", index=False
        )
    if scored_binding is not None:
        scored_binding.to_csv(output / "binding_shift_scored_predictions.csv", index=False)
    if submitted_scored is not None:
        submitted_scored.to_csv(
            output / "binding_shift_submitted_scored_predictions.csv", index=False
        )
    (output / "binding_shift_prediction_exclusions.json").write_text(
        json.dumps(binding_exclusions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "1.0.0",
        "policy": (
            "experimental labels only; no computational-teacher labels; no model "
            "selection or refitting on benchmark outputs"
        ),
        "model_sha256": sha256(model_path),
        "model_training_source_sha256": training_digest,
        "pkad2_source_sha256": sha256(pkad2_path),
        "binding_panel_sha256": sha256(binding_panel_path),
        "binding_mapping_sha256": sha256(binding_mapping_path),
        "pkad2": metrics,
        "binding_shift_panel": binding_metrics,
    }
    if submitted_metrics is not None:
        summary["submitted_binding_shift_predictions"] = submitted_metrics
    (output / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkad2", default=str(DEFAULT_PKAD2))
    parser.add_argument("--pkad2-structures", default=str(DEFAULT_PKAD2_STRUCTURES))
    parser.add_argument("--pkadr", default=str(DEFAULT_PKADR))
    parser.add_argument("--pkadr-structures", default=str(DEFAULT_PKADR_STRUCTURES))
    parser.add_argument("--model", default=str(PROJECT_ROOT / DEFAULT_ENVIRONMENT_MODEL))
    parser.add_argument("--binding-panel", default=str(DEFAULT_BINDING_PANEL))
    parser.add_argument("--binding-mapping", default=str(DEFAULT_BINDING_MAPPING))
    parser.add_argument("--binding-structures", default=str(DEFAULT_BINDING_STRUCTURES))
    parser.add_argument("--binding-predictions")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        pkad2_path=args.pkad2,
        pkad2_structures=args.pkad2_structures,
        pkadr_path=args.pkadr,
        pkadr_structures=args.pkadr_structures,
        model_path=args.model,
        binding_panel_path=args.binding_panel,
        binding_mapping_path=args.binding_mapping,
        binding_structures=args.binding_structures,
        binding_predictions_path=args.binding_predictions,
        output_dir=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
