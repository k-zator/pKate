#!/usr/bin/env python3
"""Run canonical Stage 2 and Stage 3 as one provenance-checked inference path.

The input must be a molecule microstate-network CSV produced by
``build_molecule_microstate_network_dataset.py``.  This runner deliberately does
not accept legacy candidate tables because those rows do not describe a complete,
thermodynamically connected molecular state space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

import pandas as pd  # type: ignore

from stage2_apply_network_context import (
    DEFAULT_INPUT as DEFAULT_NETWORK_DATASET,
    DEFAULT_MODEL as DEFAULT_STAGE2_MODEL,
    DEFAULT_OUTPUT as DEFAULT_STAGE2_OUTPUT,
    apply_stage2,
)
from stage3_apply_microstate_inference import (
    DEFAULT_MAX_TAUTOMERS_PER_CONFIGURATION,
    DEFAULT_OUT_DIR as DEFAULT_STAGE3_OUT_DIR,
    DEFAULT_OUTPUT as DEFAULT_STAGE3_OUTPUT,
    DEFAULT_STAGE1_OOF,
    DEFAULT_STAGE2_EVAL,
    DEFAULT_STAGE2_QUARANTINE,
    DEFAULT_TAUTOMER_SCORE_TEMPERATURE,
    apply_stage3,
)


DEFAULT_MANIFEST = (
    "data/processed/ml_models_experimental_only/stage3_microstates/"
    "canonical_pipeline_run_manifest.json"
)


def _sha256(path: str) -> str:
    """Hash each pipeline input and output for the canonical run manifest."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_canonical_pipeline(
    network_dataset_path: str = DEFAULT_NETWORK_DATASET,
    stage2_model_path: str = DEFAULT_STAGE2_MODEL,
    stage2_output_path: str = DEFAULT_STAGE2_OUTPUT,
    stage3_output_path: str = DEFAULT_STAGE3_OUTPUT,
    stage3_out_dir: str = DEFAULT_STAGE3_OUT_DIR,
    manifest_path: str = DEFAULT_MANIFEST,
    ph: float = 7.4,
    allow_compatible_network: bool = False,
    stage1_oof_path: str = DEFAULT_STAGE1_OOF,
    stage2_eval_path: str = DEFAULT_STAGE2_EVAL,
    stage2_quarantine_path: str = DEFAULT_STAGE2_QUARANTINE,
    tautomer_score_temperature: float = DEFAULT_TAUTOMER_SCORE_TEMPERATURE,
    max_tautomers_per_configuration: int = DEFAULT_MAX_TAUTOMERS_PER_CONFIGURATION,
    reuse_stage2_output: bool = False,
) -> Dict:
    """Apply both canonical stages and write a machine-readable run manifest."""
    if reuse_stage2_output:
        if not Path(stage2_output_path).is_file():
            raise FileNotFoundError(
                f"Cannot reuse missing Stage 2 output: {stage2_output_path}"
            )
        stage2_header = pd.read_csv(stage2_output_path, nrows=1, low_memory=False)
        if stage2_header.empty:
            raise ValueError("Cannot reuse an empty Stage 2 output")
        if not Path(manifest_path).is_file():
            raise FileNotFoundError(
                "Stage 2 reuse requires the prior canonical manifest for exact "
                "network/model/output provenance"
            )
        with open(manifest_path, "r", encoding="utf-8") as handle:
            prior_manifest = json.load(handle)
        reuse_checks = {
            "network_dataset_sha256": _sha256(network_dataset_path),
            "stage2_model_sha256": _sha256(stage2_model_path),
            "stage2_output_sha256": _sha256(stage2_output_path),
        }
        mismatches = {
            key: {"prior": prior_manifest.get(key), "current": value}
            for key, value in reuse_checks.items()
            if str(prior_manifest.get(key)) != str(value)
        }
        if mismatches:
            raise ValueError(
                "Refusing Stage 2 reuse because prior canonical provenance differs: "
                f"{mismatches}"
            )
        stage2_report = {
            "model_sha256": _sha256(stage2_model_path),
            "stage2_schema_version": str(
                stage2_header.iloc[0]["stage2_schema_version"]
            ),
            "stage2_free_energy_schema_version": str(
                stage2_header.iloc[0]["stage2_free_energy_schema_version"]
            ),
        }
    else:
        _, stage2_report = apply_stage2(
            network_dataset_path=network_dataset_path,
            model_path=stage2_model_path,
            output_path=stage2_output_path,
            allow_compatible_network=allow_compatible_network,
        )
    _, stage3_report = apply_stage3(
        input_path=stage2_output_path,
        stage2_model_path=stage2_model_path,
        output_path=stage3_output_path,
        out_dir=stage3_out_dir,
        ph=ph,
        stage1_oof_path=stage1_oof_path,
        stage2_eval_path=stage2_eval_path,
        stage2_quarantine_path=stage2_quarantine_path,
        tautomer_score_temperature=tautomer_score_temperature,
        max_tautomers_per_configuration=max_tautomers_per_configuration,
    )

    stage2_output_sha256 = _sha256(stage2_output_path)
    if stage3_report.get("input_sha256") != stage2_output_sha256:
        raise RuntimeError("Stage 3 did not consume the Stage 2 artifact from this run")
    if stage3_report.get("stage2_model_sha256") != stage2_report.get("model_sha256"):
        raise RuntimeError("Stage 2/3 model provenance differs within the same run")
    population_total = int(stage3_report["complete_population_molecules"]) + int(
        stage3_report["unavailable_incomplete_network_molecules"]
    )
    if population_total != int(stage3_report["molecule_rows"]):
        raise RuntimeError("Stage 3 did not account for every input molecule")

    stage3_site_output = str(Path(stage3_out_dir) / "stage3_site_predictions.csv")
    stage3_tautomer_output = str(Path(stage3_out_dir) / "stage3_ranked_tautomers.csv")
    stage3_calibration_output = str(
        Path(stage3_out_dir) / "stage3_empirical_call_calibration.json"
    )
    stage3_metrics_output = str(Path(stage3_out_dir) / "metrics.json")
    stage3_output_schema = str(Path(stage3_out_dir) / "stage3_output_schema.json")
    for required_output in (
        stage3_output_path,
        stage3_site_output,
        stage3_tautomer_output,
        stage3_calibration_output,
        stage3_metrics_output,
        stage3_output_schema,
    ):
        if not Path(required_output).is_file():
            raise RuntimeError(f"Canonical pipeline output was not written: {required_output}")

    manifest = {
        "pipeline": "canonical_molecule_network_stage2_to_stage3",
        "network_dataset": str(network_dataset_path),
        "network_dataset_sha256": _sha256(network_dataset_path),
        "stage2_model": str(stage2_model_path),
        "stage2_model_sha256": _sha256(stage2_model_path),
        "stage2_output": str(stage2_output_path),
        "stage2_output_sha256": stage2_output_sha256,
        "stage2_application_reused": bool(reuse_stage2_output),
        "stage3_output": str(stage3_output_path),
        "stage3_output_sha256": _sha256(stage3_output_path),
        "stage3_site_output": stage3_site_output,
        "stage3_site_output_sha256": _sha256(stage3_site_output),
        "stage3_tautomer_output": stage3_tautomer_output,
        "stage3_tautomer_output_sha256": _sha256(stage3_tautomer_output),
        "stage3_empirical_calibration": stage3_calibration_output,
        "stage3_empirical_calibration_sha256": _sha256(stage3_calibration_output),
        "stage3_metrics": stage3_metrics_output,
        "stage3_metrics_sha256": _sha256(stage3_metrics_output),
        "stage3_output_schema": stage3_output_schema,
        "stage3_output_schema_sha256": _sha256(stage3_output_schema),
        "population_ph": float(ph),
        "marvin_values_used": False,
        "epik_values_used": False,
        "molecule_rows": int(stage3_report["molecule_rows"]),
        "complete_population_molecules": int(
            stage3_report["complete_population_molecules"]
        ),
        "unavailable_incomplete_network_molecules": int(
            stage3_report["unavailable_incomplete_network_molecules"]
        ),
        "ranked_tautomer_rows": int(stage3_report.get("ranked_tautomer_rows", 0)),
        "tautomer_truncated_configurations": int(
            stage3_report.get("tautomer_truncated_configurations", 0)
        ),
        "empirical_pka_side_conflicts_below_half": int(
            stage3_report.get("empirical_pka_side_calibration", {})
            .get("deployment_confidence_summary", {})
            .get("conflicts_below_half", 0)
        ),
        "stage2_schema_version": stage2_report.get("stage2_schema_version"),
        "stage2_free_energy_schema_version": stage2_report.get(
            "stage2_free_energy_schema_version"
        ),
        "stage3_schema_version": stage3_report.get("stage3_schema_version"),
        "tautomer_ranking_schema_version": stage3_report.get(
            "tautomer_ranking_schema_version"
        ),
        "max_tautomers_per_configuration": int(
            stage3_report.get("max_tautomers_per_configuration", 0)
        ),
        "empirical_calibration_schema_version": stage3_report.get(
            "empirical_pka_side_calibration", {}
        ).get("schema_version"),
    }
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse the canonical Stage 2-to-Stage 3 execution and reuse options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-dataset", default=DEFAULT_NETWORK_DATASET)
    parser.add_argument("--stage2-model", default=DEFAULT_STAGE2_MODEL)
    parser.add_argument("--stage2-output", default=DEFAULT_STAGE2_OUTPUT)
    parser.add_argument("--stage3-output", default=DEFAULT_STAGE3_OUTPUT)
    parser.add_argument("--stage3-out-dir", default=DEFAULT_STAGE3_OUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--stage1-oof", default=DEFAULT_STAGE1_OOF)
    parser.add_argument("--stage2-eval", default=DEFAULT_STAGE2_EVAL)
    parser.add_argument("--stage2-quarantine", default=DEFAULT_STAGE2_QUARANTINE)
    parser.add_argument(
        "--tautomer-score-temperature",
        type=float,
        default=DEFAULT_TAUTOMER_SCORE_TEMPERATURE,
    )
    parser.add_argument(
        "--max-tautomers-per-configuration",
        type=int,
        default=DEFAULT_MAX_TAUTOMERS_PER_CONFIGURATION,
    )
    parser.add_argument(
        "--allow-compatible-network",
        action="store_true",
        help=(
            "Allow inference on a network snapshot other than the training snapshot "
            "after schema, detector, and Stage 1 provenance checks pass."
        ),
    )
    parser.add_argument(
        "--reuse-stage2-output",
        action="store_true",
        help=(
            "Skip Stage 2 application and provenance-check the existing Stage 2 CSV "
            "inside Stage 3; useful for Stage-3-only code changes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the canonical downstream pipeline and print its provenance manifest."""
    args = parse_args()
    manifest = run_canonical_pipeline(
        network_dataset_path=args.network_dataset,
        stage2_model_path=args.stage2_model,
        stage2_output_path=args.stage2_output,
        stage3_output_path=args.stage3_output,
        stage3_out_dir=args.stage3_out_dir,
        manifest_path=args.manifest,
        ph=args.ph,
        allow_compatible_network=args.allow_compatible_network,
        stage1_oof_path=args.stage1_oof,
        stage2_eval_path=args.stage2_eval,
        stage2_quarantine_path=args.stage2_quarantine,
        tautomer_score_temperature=args.tautomer_score_temperature,
        max_tautomers_per_configuration=args.max_tautomers_per_configuration,
        reuse_stage2_output=args.reuse_stage2_output,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
