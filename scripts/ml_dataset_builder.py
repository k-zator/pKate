import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from functional_group_pka_analysis import (
    CONJUGATE_FAMILY_MAP,
    CONJUGATE_MAP,
    PAIR_TYPE_FAMILY_FORMS,
    classify_pair_type,
    training_group_label,
)
from SMARTS_library import ACIDIC_PKA
from training_data_resolver import (
    DEFAULT_CURATED_ASSIGNMENTS_PATH,
    DEFAULT_CURATED_SUMMARY_PATH,
    DEFAULT_RAW_DIR,
    ensure_curated_training_data,
)


DEFAULT_ASSIGNMENTS_PATH = DEFAULT_CURATED_ASSIGNMENTS_PATH
DEFAULT_SUMMARY_PATH = DEFAULT_CURATED_SUMMARY_PATH


@dataclass
class DatasetBuildConfig:
    assignments_path: str = DEFAULT_ASSIGNMENTS_PATH
    summary_path: str = DEFAULT_SUMMARY_PATH
    min_candidates: int = 1


def _split_pipe(values: str) -> List[str]:
    return [part for part in str(values).split("|") if part]


def _molecule_key(row: pd.Series) -> str:
    return f"{row['source_file']}::{int(row['record_index'])}::{row['smiles']}"


def _build_candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        selected_label = getattr(row, "selected_group_label", None)
        if selected_label is None or str(selected_label).strip() == "" or str(selected_label).strip().lower() == "nan":
            selected_label = getattr(row, "final_group_label", None)

        instance_source = getattr(row, "resolved_group_instances", None)
        if instance_source is not None and str(instance_source).strip() and str(instance_source).strip().lower() != "nan":
            candidates = _split_pipe(instance_source)
        else:
            candidates = sorted(set(_split_pipe(row.resolved_groups)))
        if not candidates:
            continue

        # Track whether the true site has already been assigned for this
        # measurement to avoid marking multiple same-label instances as true.
        true_site_assigned = False

        for candidate_idx, candidate in enumerate(candidates):
            candidate_instance_id = f"{candidate}#{candidate_idx}"
            # Compute candidate's OWN conjugate form from its label,
            # independent of which group the measurement was assigned to.
            cand_family = CONJUGATE_FAMILY_MAP.get(candidate, candidate)
            _cand_mode, cand_own_form = classify_pair_type(candidate, cand_family)

            # Only ONE instance of the matching label should be marked
            # as is_true_site=1 per measurement.  Without atom-index
            # information we cannot tell which, so we pick the first.
            if candidate == selected_label and not true_site_assigned:
                is_true = 1
                true_site_assigned = True
            else:
                is_true = int(False)

            rows.append(
                {
                    "molecule_key": f"{row.source_file}::{int(row.record_index)}::{row.smiles}",
                    "source_file": row.source_file,
                    "record_index": int(row.record_index),
                    "smiles": row.smiles,
                    "pka_value": float(row.pka_value),
                    "pka_type_canonical": row.pka_type_canonical,
                    "resolved_group_count": int(row.resolved_group_count),
                    "resolved_groups": row.resolved_groups,
                    "resolved_group_instances": getattr(row, "resolved_group_instances", row.resolved_groups),
                    "assignment_status": row.assignment_status,
                    "candidate_label": candidate,
                    "candidate_training_group_label": training_group_label(candidate),
                    "candidate_instance_id": candidate_instance_id,
                    "true_label": selected_label,
                    "is_true_site": is_true,
                    "final_group_family": row.final_group_family,
                    "pair_member_form": row.pair_member_form,
                    "candidate_own_form": cand_own_form,
                    "candidate_family": cand_family,
                    "group_mode": row.group_mode,
                    "distribution_ok": bool(row.distribution_ok),
                    "molecule_formal_charge": getattr(row, "molecule_formal_charge", 0.0),
                    "neutral_input_risk": getattr(row, "neutral_input_risk", False),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # NOTE: candidate_global_count and candidate_uniqueness are computed from
    # the FULL dataset (all splits).  This is technically data leakage but
    # acceptable because they are label-type aggregates (how common is
    # "carboxylic_acid" overall) not per-sample features.  At deployment
    # time these statistics would come from training data only.
    counts = out["candidate_label"].value_counts().to_dict()
    out["candidate_global_count"] = out["candidate_label"].map(counts).astype(float)
    out["candidate_uniqueness"] = 1.0 / np.log1p(out["candidate_global_count"])
    return out


def _load_intrinsic_group_priors(summary_path: str) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    expected_cols = {"final_group_label", "mean", "median", "iqr", "count"}
    missing = expected_cols - set(summary.columns)
    if missing:
        raise ValueError(f"Missing required summary columns: {sorted(missing)}")

    priors = summary.rename(
        columns={
            "final_group_label": "candidate_training_group_label",
            "mean": "candidate_prior_mean_pka",
            "median": "candidate_prior_median_pka",
            "iqr": "candidate_prior_iqr_pka",
            "count": "candidate_prior_count",
        }
    )
    keep = [
        "candidate_training_group_label",
        "candidate_prior_mean_pka",
        "candidate_prior_median_pka",
        "candidate_prior_iqr_pka",
        "candidate_prior_count",
    ]
    return priors[keep]


def _build_family_priors(summary_path: str) -> pd.DataFrame:
    """Compute family-level pKa priors from per-label summary.

    Conjugate pair members (e.g. carboxylate / carboxylic_acid, primary_amine /
    primary_ammonium) all describe the **same** deprotonation equilibrium, so
    the reference pKa (or pKaH for bases) is the same regardless of which form
    the SMILES was drawn in.  This function aggregates all labels within a
    conjugate family to produce a single family-level prior.
    """
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return pd.DataFrame(columns=["candidate_family", "family_prior_mean_pka",
                                     "family_prior_median_pka", "family_prior_count"])

    summary["family"] = summary["final_group_label"].map(
        lambda lbl: CONJUGATE_FAMILY_MAP.get(lbl, lbl)
    )
    fam_agg = (
        summary.groupby("family")
        .apply(
            lambda g: pd.Series({
                "family_prior_mean_pka": float(
                    (g["mean"] * g["count"]).sum() / g["count"].sum()
                ),
                "family_prior_median_pka": float(
                    g.loc[g["count"].idxmax(), "median"]
                ),
                "family_prior_count": float(g["count"].sum()),
            }),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={"family": "candidate_family"})
    )
    return fam_agg


# Inverse of CONJUGATE_MAP: maps acid form → base form.
_CONJUGATE_MAP_INV = {v: k for k, v in CONJUGATE_MAP.items()}


def _textbook_reference_pka(label: str) -> Optional[float]:
    """Look up a textbook reference pKa / pKaH for *any* label.

    For acid-form labels (e.g. carboxylic_acid, primary_ammonium) this returns
    the direct ACIDIC_PKA entry.  For base-form labels (e.g. primary_amine,
    pyridine) this finds the conjugate acid's ACIDIC_PKA value — since both
    members of a conjugate pair share the same equilibrium.
    """
    # Direct lookup (acid-form labels live in ACIDIC_PKA)
    if label in ACIDIC_PKA:
        return float(ACIDIC_PKA[label])

    # Base form → find conjugate acid → look up its pKaH
    conjugate_acid = CONJUGATE_MAP.get(label)
    if conjugate_acid and conjugate_acid in ACIDIC_PKA:
        return float(ACIDIC_PKA[conjugate_acid])

    # Acid form without entry → try inverse to get base, then its conjugate
    conjugate_base = _CONJUGATE_MAP_INV.get(label)
    if conjugate_base and conjugate_base in ACIDIC_PKA:
        return float(ACIDIC_PKA[conjugate_base])

    return None


def build_candidate_dataset(config: DatasetBuildConfig) -> pd.DataFrame:
    assignments = pd.read_csv(config.assignments_path)
    required_cols = {
        "source_file",
        "record_index",
        "smiles",
        "pka_value",
        "resolved_groups",
        "resolved_group_count",
        "assignment_status",
        "final_group_family",
        "pair_member_form",
        "group_mode",
        "distribution_ok",
        "pka_type_canonical",
    }
    if "selected_group_label" not in assignments.columns and "final_group_label" not in assignments.columns:
        raise ValueError("Assignments must include either 'selected_group_label' or 'final_group_label'.")

    missing = required_cols - set(assignments.columns)
    if missing:
        raise ValueError(f"Missing required assignment columns: {sorted(missing)}")

    label_col = "selected_group_label" if "selected_group_label" in assignments.columns else "final_group_label"
    assignments = assignments.dropna(subset=["smiles", "resolved_groups", label_col]).copy()
    assignments["molecule_key"] = assignments.apply(_molecule_key, axis=1)

    candidate_df = _build_candidate_rows(assignments)
    if candidate_df.empty:
        return candidate_df

    # Per-label priors (original)
    priors = _load_intrinsic_group_priors(config.summary_path)
    candidate_df = candidate_df.merge(priors, on="candidate_training_group_label", how="left")

    # Family-level priors — ensures conjugate pair members share the same
    # reference pKa / pKaH instead of one getting the garbage median fill.
    if "candidate_family" not in candidate_df.columns:
        candidate_df["candidate_family"] = candidate_df["candidate_label"].map(
            lambda lbl: CONJUGATE_FAMILY_MAP.get(lbl, lbl)
        )
    family_priors = _build_family_priors(config.summary_path)
    candidate_df = candidate_df.merge(family_priors, on="candidate_family", how="left")

    # Textbook reference pKa / pKaH from ACIDIC_PKA (via conjugate mapping).
    # This provides a chemically-grounded fallback that is specific to each
    # sub-type (e.g. aniline→5, primary_amine→10) rather than averaging across
    # the whole "amine" family.
    candidate_df["textbook_reference_pka"] = candidate_df["candidate_label"].map(
        _textbook_reference_pka
    )

    fill_zero_cols = [
        "candidate_prior_count",
        "candidate_prior_iqr_pka",
        "family_prior_count",
    ]
    for col in fill_zero_cols:
        if col in candidate_df.columns:
            candidate_df[col] = candidate_df[col].fillna(0.0)

    # Priority cascade for missing per-label priors:
    #   1. Label-specific summary statistics (from training data)
    #   2. Textbook reference pKa / pKaH (ACIDIC_PKA via conjugate mapping)
    #   3. Family-level average from training data
    #   4. Global median (last resort)
    global_median = candidate_df["pka_value"].median()
    for label_col, family_col in [
        ("candidate_prior_mean_pka", "family_prior_mean_pka"),
        ("candidate_prior_median_pka", "family_prior_median_pka"),
    ]:
        candidate_df[label_col] = (
            candidate_df[label_col]
            .fillna(candidate_df["textbook_reference_pka"])
            .fillna(candidate_df.get(family_col, pd.Series(dtype=float)))
            .fillna(global_median)
        )

    group_sizes = candidate_df.groupby("molecule_key")["candidate_label"].transform("size")
    candidate_df = candidate_df[group_sizes >= config.min_candidates].copy()
    candidate_df.reset_index(drop=True, inplace=True)
    return candidate_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate-level site/state ML dataset.")
    parser.add_argument("--assignments", default=DEFAULT_ASSIGNMENTS_PATH)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out", default="data/processed/ml_site_state_candidates.csv")
    parser.add_argument("--min-candidates", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignments_path, summary_path = ensure_curated_training_data(
        assignments_path=args.assignments,
        summary_path=args.summary,
        raw_dir=args.raw_dir,
    )
    config = DatasetBuildConfig(
        assignments_path=assignments_path,
        summary_path=summary_path,
        min_candidates=args.min_candidates,
    )
    dataset = build_candidate_dataset(config)
    if dataset.empty:
        raise RuntimeError("No candidate rows were generated.")

    dataset.to_csv(args.out, index=False)
    print(f"Saved candidate ML dataset: {args.out}")
    print(f"Rows: {len(dataset)} | Molecules: {dataset['molecule_key'].nunique()}")


if __name__ == "__main__":
    main()
