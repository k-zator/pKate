import argparse
import os
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd

from functional_group_pka_analysis import CONJUGATE_FAMILY_MAP, classify_pair_type
from training_data_resolver import (
    DEFAULT_CURATED_ASSIGNMENTS_PATH,
    DEFAULT_CURATED_SUMMARY_PATH,
    DEFAULT_RAW_DIR,
    ensure_curated_training_data,
)


DEFAULT_OUT_DIR = "data/processed/dataset_audit"


def _pair_assignment_slice(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["group_mode"] == "pair_type")
        & (df["pair_member_form"].isin(["acid_form", "base_form"]))
        & (df["pair_assignment_confidence"] == "high")
        & (df["distribution_ok"])
    ].copy()


def _provenance_bucket(row: Mapping[str, object]) -> Tuple[str, str]:
    pka_source_method = str(row.get("pka_source_method", "") or "")
    if pka_source_method == "marvin":
        if bool(row.get("atom_site_match", False)):
            return "marvin_backed", "marvin_atom_match"
        return "marvin_backed", "marvin_no_atom_match"

    if bool(row.get("site_override_applied", False)):
        return "structure_supported_non_marvin", "manual_site_override"

    if bool(row.get("atom_site_match", False)):
        return "structure_supported_non_marvin", "explicit_atom_match_non_marvin"

    if str(row.get("assignment_status", "") or "") == "single_group":
        return "structure_supported_non_marvin", "single_group_drawn_state"

    return "macro_only_no_site_truth", "multi_group_pair_first"


def _split_pipe(values: object) -> List[str]:
    return [part for part in str(values).split("|") if part]


def _expand_pair_head_rows(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for row in assignments.itertuples(index=False):
        instance_source = getattr(row, "resolved_group_instances", None)
        if instance_source is not None and str(instance_source).strip() and str(instance_source).strip().lower() != "nan":
            candidates = _split_pipe(instance_source)
        else:
            candidates = sorted(set(_split_pipe(getattr(row, "resolved_groups", ""))))
        if not candidates:
            continue

        bucket, detail = _provenance_bucket(row._asdict())
        molecule_key = f"{row.source_file}::{int(row.record_index)}::{row.smiles}"
        for candidate_idx, candidate in enumerate(candidates):
            candidate_instance_id = f"{candidate}#{candidate_idx}"
            candidate_family = CONJUGATE_FAMILY_MAP.get(candidate, candidate)
            _, candidate_own_form = classify_pair_type(candidate, candidate_family)
            rows.append(
                {
                    "molecule_key": molecule_key,
                    "source_file": row.source_file,
                    "record_index": int(row.record_index),
                    "smiles": row.smiles,
                    "pka_value": float(row.pka_value),
                    "candidate_label": candidate,
                    "candidate_family": candidate_family,
                    "candidate_instance_id": candidate_instance_id,
                    "candidate_own_form": candidate_own_form,
                    "group_mode": row.group_mode,
                    "distribution_ok": bool(row.distribution_ok),
                    "provenance_bucket": bucket,
                    "provenance_detail": detail,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame = frame[
        (frame["group_mode"] == "pair_type")
        & (frame["candidate_own_form"].isin(["acid_form", "base_form"]))
        & (frame["distribution_ok"])
    ].copy()
    frame = frame.drop_duplicates(subset=["molecule_key", "candidate_instance_id", "pka_value"]).copy()
    frame.reset_index(drop=True, inplace=True)
    return frame


def _fraction_summary(df: pd.DataFrame, group_cols: List[str], dataset_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dataset", *group_cols, "rows", "fraction"])
    summary = df.groupby(group_cols, dropna=False).size().reset_index(name="rows")
    summary["dataset"] = dataset_name
    summary["fraction"] = summary["rows"] / float(len(df))
    cols = ["dataset", *group_cols, "rows", "fraction"]
    return summary[cols].sort_values("rows", ascending=False)


def _source_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dataset", "source_file", "provenance_bucket", "rows", "fraction"])
    summary = (
        df.groupby(["source_file", "provenance_bucket"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", "source_file"], ascending=[False, True])
    )
    summary["dataset"] = dataset_name
    summary["fraction"] = summary["rows"] / float(len(df))
    return summary[["dataset", "source_file", "provenance_bucket", "rows", "fraction"]]


def _family_summary(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dataset", "candidate_family", "provenance_bucket", "rows", "fraction"])
    summary = (
        df.groupby(["candidate_family", "provenance_bucket"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["rows", "candidate_family"], ascending=[False, True])
    )
    summary["dataset"] = dataset_name
    summary["fraction"] = summary["rows"] / float(len(df))
    return summary[["dataset", "candidate_family", "provenance_bucket", "rows", "fraction"]]


def _format_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _top_rows(df: pd.DataFrame, limit: int = 8) -> str:
    if df.empty:
        return "(empty)"
    return df.head(limit).to_string(index=False)


def _build_markdown_report(
    assignments: pd.DataFrame,
    assignment_bucket_summary: pd.DataFrame,
    assignment_detail_summary: pd.DataFrame,
    assignment_source_summary: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    candidate_bucket_summary: pd.DataFrame,
    candidate_detail_summary: pd.DataFrame,
    candidate_source_summary: pd.DataFrame,
    candidate_family_summary: pd.DataFrame,
) -> str:
    assignment_total = len(assignments)
    candidate_total = len(candidate_rows)
    override_count = int(pd.to_numeric(assignments.get("site_override_applied", False), errors="coerce").fillna(0).astype(bool).sum())
    marvin_atom_match = assignment_detail_summary.loc[
        assignment_detail_summary["provenance_detail"] == "marvin_atom_match",
        "rows",
    ].sum()
    marvin_total = assignment_bucket_summary.loc[
        assignment_bucket_summary["provenance_bucket"] == "marvin_backed",
        "rows",
    ].sum()

    lines = [
        "# Pair-Form Training Provenance Audit",
        "",
        "This report measures how much of the current pair-form supervision comes from direct Marvin annotations versus rows that rely only on the drawn protonation state and local site-resolution logic.",
        "",
        "## Assignment-level pair labels",
        f"- total rows: {assignment_total}",
        f"- manual site overrides in this slice: {override_count}",
        f"- Marvin atom-matched rows: {marvin_atom_match} / {marvin_total} direct Marvin rows" if marvin_total else "- Marvin atom-matched rows: 0 / 0 direct Marvin rows",
    ]

    for row in assignment_bucket_summary.itertuples(index=False):
        lines.append(
            f"- {row.provenance_bucket}: {int(row.rows)} rows ({_format_pct(row.fraction)})"
        )

    lines.extend(
        [
            "",
            "## Candidate-level generic pair-head rows",
            f"- total rows: {candidate_total}",
        ]
    )
    for row in candidate_bucket_summary.itertuples(index=False):
        lines.append(
            f"- {row.provenance_bucket}: {int(row.rows)} rows ({_format_pct(row.fraction)})"
        )

    lines.extend(
        [
            "",
            "## Assignment detail breakdown",
            "```text",
            _top_rows(assignment_detail_summary),
            "```",
            "",
            "## Candidate detail breakdown",
            "```text",
            _top_rows(candidate_detail_summary),
            "```",
            "",
            "## Top sources in assignment-level pair labels",
            "```text",
            _top_rows(assignment_source_summary, limit=12),
            "```",
            "",
            "## Top sources in candidate-level pair-head rows",
            "```text",
            _top_rows(candidate_source_summary, limit=12),
            "```",
            "",
            "## Candidate family breakdown",
            "```text",
            _top_rows(candidate_family_summary, limit=15),
            "```",
            "",
            "## Bucket definitions",
            "- marvin_backed: row comes from a `marvin_pKa` / `marvin_atom` / `marvin_pKa_type` measurement record.",
            "- structure_supported_non_marvin: non-Marvin row with either a single resolved site, a manual override, or an explicit atom match.",
            "- macro_only_no_site_truth: non-Marvin multi-group row with no explicit site truth, so site selection is heuristic (`pair_first`).",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Marvin-backed pair-form supervision in the curated training data.")
    parser.add_argument("--assignments", default=DEFAULT_CURATED_ASSIGNMENTS_PATH)
    parser.add_argument("--summary", default=DEFAULT_CURATED_SUMMARY_PATH)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    assignments_path, _ = ensure_curated_training_data(
        assignments_path=args.assignments,
        summary_path=args.summary,
        raw_dir=args.raw_dir,
    )
    assignments_df = pd.read_csv(assignments_path, low_memory=False)

    pair_assignments = _pair_assignment_slice(assignments_df)
    if pair_assignments.empty:
        raise RuntimeError("No pair-type assignment rows were found in the curated assignments file.")

    pair_assignments[["provenance_bucket", "provenance_detail"]] = pair_assignments.apply(
        lambda row: pd.Series(_provenance_bucket(row)),
        axis=1,
    )
    candidate_rows = _expand_pair_head_rows(assignments_df)

    assignment_bucket_summary = _fraction_summary(
        pair_assignments,
        group_cols=["provenance_bucket"],
        dataset_name="assignment_pair_labels",
    )
    assignment_detail_summary = _fraction_summary(
        pair_assignments,
        group_cols=["provenance_bucket", "provenance_detail"],
        dataset_name="assignment_pair_labels",
    )
    assignment_source_summary = _source_summary(pair_assignments, dataset_name="assignment_pair_labels")

    candidate_bucket_summary = _fraction_summary(
        candidate_rows,
        group_cols=["provenance_bucket"],
        dataset_name="candidate_pair_head_rows",
    )
    candidate_detail_summary = _fraction_summary(
        candidate_rows,
        group_cols=["provenance_bucket", "provenance_detail"],
        dataset_name="candidate_pair_head_rows",
    )
    candidate_source_summary = _source_summary(candidate_rows, dataset_name="candidate_pair_head_rows")
    candidate_family_summary = _family_summary(candidate_rows, dataset_name="candidate_pair_head_rows")

    assignment_bucket_summary.to_csv(os.path.join(args.out_dir, "pair_form_training_provenance_assignment_summary.csv"), index=False)
    assignment_detail_summary.to_csv(os.path.join(args.out_dir, "pair_form_training_provenance_assignment_detail.csv"), index=False)
    assignment_source_summary.to_csv(os.path.join(args.out_dir, "pair_form_training_provenance_assignment_by_source.csv"), index=False)
    candidate_bucket_summary.to_csv(os.path.join(args.out_dir, "pair_form_training_provenance_candidate_summary.csv"), index=False)
    candidate_detail_summary.to_csv(os.path.join(args.out_dir, "pair_form_training_provenance_candidate_detail.csv"), index=False)
    candidate_source_summary.to_csv(os.path.join(args.out_dir, "pair_form_training_provenance_candidate_by_source.csv"), index=False)
    candidate_family_summary.to_csv(os.path.join(args.out_dir, "pair_form_training_provenance_candidate_by_family.csv"), index=False)

    report = _build_markdown_report(
        assignments=pair_assignments,
        assignment_bucket_summary=assignment_bucket_summary,
        assignment_detail_summary=assignment_detail_summary,
        assignment_source_summary=assignment_source_summary,
        candidate_rows=candidate_rows,
        candidate_bucket_summary=candidate_bucket_summary,
        candidate_detail_summary=candidate_detail_summary,
        candidate_source_summary=candidate_source_summary,
        candidate_family_summary=candidate_family_summary,
    )
    report_path = os.path.join(args.out_dir, "pair_form_training_provenance.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(f"Saved pair-form provenance audit to {args.out_dir}")
    print(report)


if __name__ == "__main__":
    main()