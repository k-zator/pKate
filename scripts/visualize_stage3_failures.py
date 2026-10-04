import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem import Draw  # type: ignore

from substructure_match import find_sites_with_metadata, resolve_overlapping_sites


def _collect_sites(smiles: str) -> Tuple[Optional[Chem.Mol], List[Dict]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, []
    candidates = find_sites_with_metadata(mol)
    resolved, _ = resolve_overlapping_sites(candidates, overlap_threshold=0.5)
    return mol, resolved


def _best_site_for_label(resolved_sites: List[Dict], label: Optional[str]) -> Optional[Dict]:
    if not label or not resolved_sites:
        return None

    exact = [site for site in resolved_sites if site.get("label") == label]
    if exact:
        return sorted(
            exact,
            key=lambda x: (x.get("priority", 0), x.get("specificity", 0), len(x.get("atom_set", set()))),
            reverse=True,
        )[0]

    return None


def _top_prediction_per_molecule(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["molecule_key", "combined_score"], ascending=[True, False])
        .groupby("molecule_key", as_index=False)
        .first()
    )


def _build_failure_table(top_df: pd.DataFrame) -> pd.DataFrame:
    out = top_df.copy()
    out["site_failed"] = out["candidate_label"] != out["true_label"]

    pair_truth_present = out["true_pair_member_form"].isin(["acid_form", "base_form"])
    out["pair_failed"] = pair_truth_present & (out["pred_member_form"] != out["true_pair_member_form"])
    out["failure_type"] = "ok"
    out.loc[out["site_failed"], "failure_type"] = "site"
    out.loc[out["pair_failed"], "failure_type"] = "pair_form"
    out.loc[out["site_failed"] & out["pair_failed"], "failure_type"] = "site+pair_form"
    return out


def _all_group_predictions(df: pd.DataFrame) -> Dict[str, List[Dict[str, object]]]:
    preds: Dict[str, List[Dict[str, object]]] = {}
    for molecule_key, group in df.groupby("molecule_key", sort=False):
        ranked = group.sort_values("combined_score", ascending=False)
        rows = []
        for rank_idx, (_, row) in enumerate(ranked.iterrows(), start=1):
            group_id = row.get("candidate_instance_id")
            if pd.isna(group_id):
                group_id = row.get("candidate_label")
            rows.append(
                {
                    "rank": rank_idx,
                    "group_id": group_id,
                    "label": row.get("candidate_label"),
                    "pred_member_form": row.get("pred_member_form"),
                    "pka_eff": row.get("pred_effective_pka"),
                    "site_prob": row.get("site_prob"),
                    "member_presence_prob": row.get("member_presence_prob"),
                    "selection_confidence": row.get("selection_confidence"),
                    "pka_eff_confidence": row.get("pka_eff_confidence"),
                    "score": row.get("combined_score"),
                    "is_true_site": row.get("is_true_site"),
                }
            )
        preds[str(molecule_key)] = rows
    return preds


def _resolved_site_summary(resolved: List[Dict]) -> str:
    return "|".join([f"{site.get('label')}#{idx}" for idx, site in enumerate(resolved)])


def _format_group_legend(group_rows: List[Dict[str, object]], max_groups: int, full_details: bool) -> str:
    if max_groups > 0:
        entries = group_rows[:max_groups]
    else:
        entries = group_rows
    if not entries:
        return "NA"

    formatted = []
    for item in entries:
        group_id = str(item.get("group_id", item.get("label", "NA")))
        label = str(item.get("label", "NA"))
        pka_eff = item.get("pka_eff")
        if full_details:
            site_prob = float(item.get("site_prob", 0.0) or 0.0)
            occ_prob = float(item.get("member_presence_prob", 0.0) or 0.0)
            sel_conf = float(item.get("selection_confidence", 0.0) or 0.0)
            formatted.append(
                f"{group_id}:{label} pKa={float(pka_eff):.2f} site={site_prob:.2f} occ={occ_prob:.2f} conf={sel_conf:.2f}"
                if pka_eff is not None and not pd.isna(pka_eff)
                else f"{group_id}:{label} pKa=NA site={site_prob:.2f} occ={occ_prob:.2f} conf={sel_conf:.2f}"
            )
        else:
            formatted.append(
                f"{group_id}:{label}:{float(pka_eff):.2f}"
                if pka_eff is not None and not pd.isna(pka_eff)
                else f"{group_id}:{label}:NA"
            )
    return " | ".join(formatted)


def _format_example_report(sample: pd.DataFrame, all_preds: Dict[str, List[Dict[str, object]]]) -> str:
    lines: List[str] = []
    for example_idx, (_, row) in enumerate(sample.iterrows(), start=1):
        molecule_key = str(row.get("molecule_key"))
        lines.append(
            f"Example {example_idx}: {molecule_key}\n"
            f"  smiles={row.get('smiles')}\n"
            f"  true={row.get('true_label')} | pred={row.get('candidate_label')} | failure_type={row.get('failure_type')}\n"
            f"  pred_form={row.get('pred_member_form')} | true_form={row.get('true_pair_member_form')}\n"
            f"  pKa_eff={float(row.get('pred_effective_pka', float('nan'))):.2f} | "
            f"selection_conf={100.0 * float(row.get('selection_confidence', 0.0)):.1f}% | "
            f"pKa_conf={100.0 * float(row.get('pka_eff_confidence', 0.0)):.1f}%"
        )
        lines.append("  groups:")
        for item in all_preds.get(molecule_key, []):
            group_id = item.get("group_id", item.get("label", "NA"))
            label = item.get("label", "NA")
            pka_eff = item.get("pka_eff")
            pka_text = f"{float(pka_eff):.2f}" if pka_eff is not None and not pd.isna(pka_eff) else "NA"
            lines.append(
                "    "
                f"rank={int(item.get('rank', 0))} "
                f"id={group_id} "
                f"label={label} "
                f"form={item.get('pred_member_form')} "
                f"pKa={pka_text} "
                f"site_prob={float(item.get('site_prob', 0.0) or 0.0):.3f} "
                f"occ_prob={float(item.get('member_presence_prob', 0.0) or 0.0):.3f} "
                f"score={float(item.get('score', 0.0) or 0.0):.3f} "
                f"sel_conf={float(item.get('selection_confidence', 0.0) or 0.0):.3f} "
                f"pKa_conf={float(item.get('pka_eff_confidence', 0.0) or 0.0):.3f}"
            )
        lines.append("")
    return "\n".join(lines).strip()


def _select_example_rows(
    outcome_df: pd.DataFrame,
    example_set: str,
    failure_type: str,
    max_examples: int,
    min_groups: int,
) -> pd.DataFrame:
    work = outcome_df.copy()
    if example_set == "failures":
        if failure_type != "all":
            work = work[work["failure_type"] == failure_type].copy()
        else:
            work = work[work["failure_type"] != "ok"].copy()
        return work.sort_values(["failure_type", "combined_score"], ascending=[True, False]).head(max_examples)

    work = work[work["failure_type"] == "ok"].copy()
    if min_groups > 0:
        filtered = work[pd.to_numeric(work.get("resolved_group_count", 0), errors="coerce").fillna(0) >= int(min_groups)].copy()
        if not filtered.empty:
            work = filtered

    sort_cols = ["resolved_group_count", "selection_confidence", "pka_eff_confidence", "combined_score"]
    ascending = [False, False, False, False]
    work = work.sort_values(sort_cols, ascending=ascending)

    if example_set == "headline":
        work = work.copy()
        work["_resolved_group_count_num"] = pd.to_numeric(work.get("resolved_group_count", 0), errors="coerce").fillna(0)
        work["_smiles_length"] = work.get("smiles", "").astype(str).str.len()
        work["_has_pair_truth"] = work.get("true_pair_member_form", "").isin(["acid_form", "base_form"]).astype(int)
        work["_headline_group_distance"] = (work["_resolved_group_count_num"] - 4.0).abs()

        compact = work[
            (work["_resolved_group_count_num"] >= max(2, int(min_groups)))
            & (work["_resolved_group_count_num"] <= 8)
        ].copy()
        if not compact.empty:
            work = compact

        work = work.sort_values(
            [
                "_has_pair_truth",
                "_headline_group_distance",
                "selection_confidence",
                "pka_eff_confidence",
                "_smiles_length",
                "combined_score",
            ],
            ascending=[False, True, False, False, True, False],
        )
        return work.head(1)
    return work.head(max_examples)


def _render_grid(
    failure_df: pd.DataFrame,
    all_preds: Dict[str, List[Dict[str, object]]],
    out_png: Path,
    out_csv: Path,
    out_groups_csv: Path,
    max_examples: int,
    show_all_groups: bool,
    legend_full_details: bool,
    max_legend_groups: int,
) -> None:
    sample = failure_df.head(max_examples).copy()

    mols = []
    legends = []
    atom_lists = []
    atom_colors = []
    rows = []
    group_detail_rows = []

    for _, row in sample.iterrows():
        smiles = row["smiles"]
        molecule_key = str(row.get("molecule_key"))
        mol, resolved = _collect_sites(smiles)
        if mol is None:
            continue

        pred_site = _best_site_for_label(resolved, row.get("candidate_label"))
        true_site = _best_site_for_label(resolved, row.get("true_label"))

        color_map: Dict[int, Tuple[float, float, float]] = {}

        palette = [
            (0.80, 0.80, 0.25),
            (0.25, 0.75, 0.75),
            (0.80, 0.55, 0.25),
            (0.55, 0.80, 0.35),
            (0.70, 0.60, 0.85),
            (0.85, 0.55, 0.70),
            (0.60, 0.75, 0.55),
        ]

        group_rows = all_preds.get(molecule_key, [])
        if show_all_groups:
            for idx, site in enumerate(resolved):
                color = palette[idx % len(palette)]
                for atom_idx in sorted(list(site.get("atom_set", set()))):
                    color_map[int(atom_idx)] = color

        pred_atoms = sorted(list(pred_site.get("atom_set", set()))) if pred_site else []
        true_atoms = sorted(list(true_site.get("atom_set", set()))) if true_site else []

        for atom_idx in true_atoms:
            color_map[int(atom_idx)] = (0.2, 0.45, 0.95)
        for atom_idx in pred_atoms:
            if int(atom_idx) in color_map:
                color_map[int(atom_idx)] = (0.7, 0.2, 0.7)
            else:
                color_map[int(atom_idx)] = (0.9, 0.2, 0.2)

        highlight_atoms = sorted(color_map.keys())

        all_group_legend = _format_group_legend(
            group_rows,
            max_groups=max_legend_groups,
            full_details=legend_full_details,
        )

        is_wrong_group = row.get("candidate_label") != row.get("true_label")
        pred_tag = "pred✗" if is_wrong_group else "pred✓"
        pka_conf_pct = float(row.get("pka_eff_confidence", 0.0)) * 100.0
        select_conf_pct = float(row.get("selection_confidence", 0.0)) * 100.0
        ci_low = row.get("pred_effective_pka_ci_low", float("nan"))
        ci_high = row.get("pred_effective_pka_ci_high", float("nan"))

        legend = (
            f"fail={row['failure_type']} | true={row.get('true_label', 'NA')} | {pred_tag}={row.get('candidate_label', 'NA')}\n"
            f"true_form={row.get('true_pair_member_form', 'NA')} | pred_form={row.get('pred_member_form', 'NA')} | "
            f"pKa(H)_eff={row.get('pred_effective_pka', float('nan')):.2f} [{float(ci_low):.2f},{float(ci_high):.2f}]\n"
            f"pKa(H)_conf={pka_conf_pct:.1f}% | select_conf={select_conf_pct:.1f}%\n"
            f"all:{all_group_legend}"
        )

        mols.append(mol)
        legends.append(legend)
        atom_lists.append(highlight_atoms)
        atom_colors.append(color_map)

        rows.append(
            {
                "molecule_key": row.get("molecule_key"),
                "smiles": smiles,
                "failure_type": row.get("failure_type"),
                "true_label": row.get("true_label"),
                "pred_label": row.get("candidate_label"),
                "true_pair_member_form": row.get("true_pair_member_form"),
                "pred_member_form": row.get("pred_member_form"),
                "pred_effective_pka": row.get("pred_effective_pka"),
                "combined_score": row.get("combined_score"),
                "site_prob": row.get("site_prob"),
                "member_presence_prob": row.get("member_presence_prob"),
                "selection_confidence": row.get("selection_confidence"),
                "pka_eff_confidence": row.get("pka_eff_confidence"),
                "pred_effective_pka_ci_low": row.get("pred_effective_pka_ci_low"),
                "pred_effective_pka_ci_high": row.get("pred_effective_pka_ci_high"),
                "pred_atoms": "|".join(map(str, pred_atoms)),
                "true_atoms": "|".join(map(str, true_atoms)),
                "resolved_sites_indexed": _resolved_site_summary(resolved),
                "all_group_labels": "|".join([str(item.get("label")) for item in group_rows if item.get("label") is not None]),
                "all_group_ids": "|".join([str(item.get("group_id")) for item in group_rows if item.get("group_id") is not None]),
                "all_group_pka_eff": "|".join(
                    [
                        f"{str(item.get('group_id'))}:{float(item.get('pka_eff')):.3f}"
                        for item in group_rows
                        if item.get("group_id") is not None and item.get("pka_eff") is not None and not pd.isna(item.get("pka_eff"))
                    ]
                ),
            }
        )

        for item in group_rows:
            group_detail_rows.append(
                {
                    "molecule_key": molecule_key,
                    "smiles": smiles,
                    "failure_type": row.get("failure_type"),
                    "true_label": row.get("true_label"),
                    "top_pred_label": row.get("candidate_label"),
                    "group_rank": item.get("rank"),
                    "group_id": item.get("group_id"),
                    "group_label": item.get("label"),
                    "group_pred_member_form": item.get("pred_member_form"),
                    "group_pred_effective_pka": item.get("pka_eff"),
                    "group_site_prob": item.get("site_prob"),
                    "group_member_presence_prob": item.get("member_presence_prob"),
                    "group_combined_score": item.get("score"),
                    "group_selection_confidence": item.get("selection_confidence"),
                    "group_pka_eff_confidence": item.get("pka_eff_confidence"),
                    "group_is_true_site": item.get("is_true_site"),
                }
            )

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    pd.DataFrame(group_detail_rows).to_csv(out_groups_csv, index=False)

    if mols:
        image = Draw.MolsToGridImage(
            mols,
            molsPerRow=4,
            subImgSize=(500, 320),
            legends=legends,
            highlightAtomLists=atom_lists,
            highlightAtomColors=atom_colors,
            useSVG=False,
        )
        image.save(str(out_png))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Stage 3 molecules with predicted vs true highlighted sites.")
    parser.add_argument("--in-csv", required=True, help="Stage 3 candidate scores CSV")
    parser.add_argument("--out-dir", default="data/processed/stage3_failure_visualization")
    parser.add_argument("--max-examples", type=int, default=24)
    parser.add_argument("--example-set", choices=["failures", "correct", "headline"], default="failures")
    parser.add_argument("--failure-type", default="all", help="all|site|pair_form|site+pair_form")
    parser.add_argument("--min-groups", type=int, default=2, help="Minimum resolved_group_count when selecting correct/headline examples")
    parser.add_argument("--hide-all-groups", action="store_true", help="Disable all-group atom overlay and legend list")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv, low_memory=False)
    required = {
        "molecule_key",
        "smiles",
        "candidate_label",
        "true_label",
        "pred_member_form",
        "true_pair_member_form",
        "combined_score",
        "pred_effective_pka",
        "site_prob",
        "member_presence_prob",
        "selection_confidence",
        "pka_eff_confidence",
        "pred_effective_pka_ci_low",
        "pred_effective_pka_ci_high",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")

    top_df = _top_prediction_per_molecule(df)
    outcome_df = _build_failure_table(top_df)
    all_preds = _all_group_predictions(df)

    sample_df = _select_example_rows(
        outcome_df,
        example_set=str(args.example_set),
        failure_type=str(args.failure_type),
        max_examples=int(args.max_examples),
        min_groups=int(args.min_groups),
    )

    base_name = f"stage3_{args.example_set}"
    if args.example_set == "failures":
        safe = args.failure_type.replace("+", "_plus_")
        base_name = f"{base_name}_{safe}"
    out_png = out_dir / f"{base_name}.png"
    out_csv = out_dir / f"{base_name}.csv"
    out_groups_csv = out_dir / f"{base_name}_groups.csv"
    out_txt = out_dir / f"{base_name}.txt"
    summary_csv = out_dir / "stage3_failure_summary.csv"

    _render_grid(
        sample_df,
        all_preds=all_preds,
        out_png=out_png,
        out_csv=out_csv,
        out_groups_csv=out_groups_csv,
        max_examples=args.max_examples,
        show_all_groups=not args.hide_all_groups,
        legend_full_details=bool(args.example_set in {"correct", "headline"}),
        max_legend_groups=0 if args.example_set in {"correct", "headline"} else 6,
    )

    report_text = _format_example_report(sample_df, all_preds)
    out_txt.write_text(report_text + "\n", encoding="utf-8")

    summary = (
        outcome_df
        .groupby("failure_type")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary.to_csv(summary_csv, index=False)

    print(f"Saved example grid:  {out_png}")
    print(f"Saved example table: {out_csv}")
    print(f"Saved group table:   {out_groups_csv}")
    print(f"Saved text report:   {out_txt}")
    print(f"Saved summary:      {summary_csv}")
    if report_text:
        print(report_text)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
