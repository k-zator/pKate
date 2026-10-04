"""LEGACY candidate-table scorer; not part of canonical Stage 1/2/3 inference.

Use ``run_canonical_protonation_pipeline.py`` for deployable molecule-level
microstate inference.  This file is retained only to reproduce older experiments.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from ionization_context_features import build_ionization_context_frame
from stage3_protonation_state_inference import (
    DEFAULT_CARBOXYL_FORM_MODEL_PATH,
    DEFAULT_PAIR_FORM_HEADS_PATH,
    DEFAULT_SITE_MODEL_PATH,
    DEFAULT_STAGE1_MODEL_PATH,
    DEFAULT_STAGE2_MODEL_PATH,
    DEFAULT_THERMO_PAIR_FORM_MODEL_PATH,
    _add_selection_confidence,
    _apply_selection_score,
    _apply_pair_form_layer,
    _apply_uncertainty_ph_fallback,
    _build_stage1_like_features,
    _build_stage2_features,
    _infer_site_probability,
    _label_form,
    _load_bundle,
    _molecule_level_summary,
    _protonated_fraction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LEGACY: score an old candidate-level CSV. This is not canonical Stage 3; "
            "use run_canonical_protonation_pipeline.py for molecule networks."
        )
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--stage1-model", default=DEFAULT_STAGE1_MODEL_PATH)
    parser.add_argument("--stage2-model", default=DEFAULT_STAGE2_MODEL_PATH)
    parser.add_argument("--site-model", default=DEFAULT_SITE_MODEL_PATH)
    parser.add_argument("--pair-form-mode", choices=["thermo", "legacy", "none"], default="thermo")
    parser.add_argument("--site-ranking-mode", choices=["joint", "site_only", "site_then_state"], default="site_only")
    parser.add_argument("--thermo-pair-form-model", default=DEFAULT_THERMO_PAIR_FORM_MODEL_PATH)
    parser.add_argument("--carboxyl-form-model", default=DEFAULT_CARBOXYL_FORM_MODEL_PATH)
    parser.add_argument("--pair-form-heads", default=DEFAULT_PAIR_FORM_HEADS_PATH)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--uncertain-threshold", type=float, default=0.55)
    parser.add_argument("--out-csv", default="data/processed/external_datasets/pipeline_scored_candidates.csv")
    parser.add_argument("--out-json", default="data/processed/external_datasets/pipeline_scored_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_df = pd.read_csv(args.candidates)
    if candidate_df.empty:
        raise RuntimeError("Candidate CSV is empty.")

    required = {
        "molecule_key",
        "smiles",
        "candidate_label",
        "resolved_groups",
        "resolved_group_count",
        "candidate_prior_mean_pka",
        "candidate_prior_median_pka",
        "candidate_prior_iqr_pka",
        "candidate_prior_count",
        "candidate_uniqueness",
        "pka_type_canonical",
    }
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(f"Missing required candidate columns: {sorted(missing)}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    stage1 = _load_bundle(args.stage1_model)
    stage2 = _load_bundle(args.stage2_model)
    site_bundle = _load_bundle(args.site_model) if args.site_model else None
    thermo_bundle = None
    carboxyl_bundle = None
    pair_heads_bundle = None
    if args.pair_form_mode == "thermo":
        thermo_bundle = _load_bundle(args.thermo_pair_form_model) if args.thermo_pair_form_model else None
    elif args.pair_form_mode == "legacy":
        carboxyl_bundle = _load_bundle(args.carboxyl_form_model) if args.carboxyl_form_model else None
        pair_heads_bundle = _load_bundle(args.pair_form_heads) if args.pair_form_heads else None

    candidate_df = candidate_df.copy()
    candidate_df["ph_target"] = float(args.ph)

    X_stage1 = _build_stage1_like_features(
        candidate_df,
        feature_columns=stage1.get("feature_columns"),
        fp_bits=int(stage1.get("fp_bits", 512)),
        fp_radius=int(stage1.get("fp_radius", 2)),
    )
    candidate_df["intrinsic_pred_pka"] = stage1["model"].predict(X_stage1)

    is_multi = pd.to_numeric(candidate_df["resolved_group_count"], errors="coerce").fillna(1) > 1
    candidate_df["pred_delta_pka"] = 0.0
    if is_multi.any():
        X_stage2 = _build_stage2_features(
            candidate_df.loc[is_multi],
            feature_columns=stage2.get("feature_columns"),
            fp_bits=int(stage2.get("fp_bits", 512)),
            fp_radius=int(stage2.get("fp_radius", 2)),
        )
        candidate_df.loc[is_multi, "pred_delta_pka"] = stage2["model"].predict(X_stage2)

    candidate_df["pred_effective_pka"] = candidate_df["intrinsic_pred_pka"] + candidate_df["pred_delta_pka"]

    family_form = candidate_df["candidate_label"].map(_label_form)
    candidate_df["candidate_family"] = family_form.map(lambda value: value[0])
    candidate_df["candidate_member_form"] = family_form.map(lambda value: value[1] if value[1] is not None else "non_pair")

    frac_prot = _protonated_fraction(candidate_df["pred_effective_pka"].to_numpy(dtype=float), pH=float(args.ph))
    candidate_df["pred_protonated_fraction"] = frac_prot
    candidate_df["pred_is_deprotonated"] = candidate_df["pred_effective_pka"] < float(args.ph)
    candidate_df["pred_is_protonated"] = ~candidate_df["pred_is_deprotonated"]

    candidate_df["pred_member_form"] = np.where(
        candidate_df["candidate_member_form"] == "acid_form",
        np.where(candidate_df["pred_is_protonated"], "acid_form", "base_form"),
        np.where(
            candidate_df["candidate_member_form"] == "base_form",
            np.where(candidate_df["pred_is_protonated"], "acid_form", "base_form"),
            candidate_df["candidate_member_form"],
        ),
    )
    candidate_df["member_presence_prob"] = np.where(
        candidate_df["candidate_member_form"] == "acid_form",
        candidate_df["pred_protonated_fraction"],
        np.where(candidate_df["candidate_member_form"] == "base_form", 1.0 - candidate_df["pred_protonated_fraction"], 0.5),
    )

    candidate_df["site_prob"] = _infer_site_probability(candidate_df, site_bundle)
    candidate_df = _apply_selection_score(candidate_df, site_ranking_mode=str(args.site_ranking_mode))
    candidate_df = _add_selection_confidence(candidate_df, pH=float(args.ph))
    candidate_df = pd.concat(
        [
            candidate_df,
            build_ionization_context_frame(
                candidate_df,
                resolved_col="resolved_groups",
                candidate_col="candidate_label",
                smiles_col="smiles",
            ),
        ],
        axis=1,
    )
    candidate_df = _apply_uncertainty_ph_fallback(
        candidate_df,
        pH=float(args.ph),
        uncertain_threshold=float(args.uncertain_threshold),
    )
    candidate_df = _apply_pair_form_layer(
        candidate_df,
        pair_form_mode=str(args.pair_form_mode),
        thermo_bundle=thermo_bundle,
        carboxyl_bundle=carboxyl_bundle,
        pair_heads_bundle=pair_heads_bundle,
    )

    metrics = {
        "rows": int(len(candidate_df)),
        "molecules": int(candidate_df["molecule_key"].nunique()),
        "ph": float(args.ph),
        "uncertain_threshold": float(args.uncertain_threshold),
        "pair_form_mode": str(args.pair_form_mode),
        "site_ranking_mode": str(args.site_ranking_mode),
        "used_site_model": bool(site_bundle is not None and "feature_columns" in site_bundle),
        "used_thermo_pair_form_model": bool(thermo_bundle is not None and "feature_columns" in thermo_bundle),
        "thermo_feature_mode": thermo_bundle.get("feature_mode") if thermo_bundle is not None else None,
        "used_carboxyl_form_model": bool(carboxyl_bundle is not None and "feature_columns" in carboxyl_bundle),
        "used_pair_form_heads": bool(pair_heads_bundle is not None and pair_heads_bundle.get("heads")),
    }
    if {"true_label", "is_true_site"}.issubset(candidate_df.columns):
        inferred_true_form = candidate_df["true_label"].map(
            lambda label: (_label_form(str(label))[1] if pd.notna(label) else None)
        )
        candidate_df["true_pair_member_form"] = inferred_true_form
        metrics.update(_molecule_level_summary(candidate_df))

    candidate_df.to_csv(args.out_csv, index=False)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved pipeline-scored candidates: {args.out_csv}")
    print(f"Saved pipeline metrics: {args.out_json}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
