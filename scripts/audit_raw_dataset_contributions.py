import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore

from functional_group_pka_analysis import (
    ACIDIC_FAMILIES,
    BASIC_FAMILIES,
    PAIR_TYPE_FAMILY_FORMS,
    _iter_sdf_mols,
    _molecule_formal_charge,
    _sites_matching_atom_index,
    assign_single_group_label,
    canonical_family_label,
    canonicalize_pka_type,
    classify_pair_type,
    conjugate_check_flag,
    distribution_sanity_check,
    extract_measurements,
    find_sites_with_metadata,
    normalize_conjugate_family,
    refine_group_label,
    resolve_overlapping_sites,
    reinterpret_pka_type_for_two_faced,
    select_pair_first_site,
)


def build_rows_for_sdf(sdf_path: str, overlap_threshold: float) -> pd.DataFrame:
    source_file = os.path.basename(sdf_path)
    all_rows: List[Dict] = []

    for record_idx, mol in _iter_sdf_mols(sdf_path):
        smiles = Chem.MolToSmiles(mol, canonical=True)
        measurements = extract_measurements(mol)
        if not measurements:
            continue

        candidates = find_sites_with_metadata(mol)
        resolved_sites, rejected_sites = resolve_overlapping_sites(
            candidates,
            overlap_threshold=overlap_threshold,
        )

        status = "no_group" if len(resolved_sites) == 0 else ("single_group" if len(resolved_sites) == 1 else "multi_group")
        final_group = assign_single_group_label(resolved_sites)

        for measurement in measurements:
            pka_value = measurement["pka_value"]
            pka_type_canonical = canonicalize_pka_type(measurement["pka_type_raw"])

            atom_matched_sites, atom_meta = _sites_matching_atom_index(
                resolved_sites,
                measurement["atom_index_raw"],
            )
            selection_pool = atom_matched_sites if atom_matched_sites else resolved_sites
            measurement_site, pair_meta = select_pair_first_site(
                selection_pool,
                preferred_type=pka_type_canonical,
                pka_value=pka_value,
            )
            if atom_meta["atom_site_match"]:
                pair_meta = {
                    **pair_meta,
                    "pair_selection_reason": f"atom_index_then_{pair_meta['pair_selection_reason']}",
                }

            measurement_group = measurement_site.get("label") if measurement_site else final_group
            measurement_family = (
                measurement_site.get("pair_family", measurement_site.get("family"))
                if measurement_site
                else normalize_conjugate_family(measurement_group)
            )
            pair_family = measurement_family if measurement_family in PAIR_TYPE_FAMILY_FORMS else None
            measurement_canonical = canonical_family_label(
                measurement_group,
                family=measurement_family,
            )
            measurement_refined = refine_group_label(
                mol=mol,
                site=measurement_site,
                base_label=measurement_group,
            )
            group_mode, pair_member_form = classify_pair_type(
                label=measurement_group,
                family=measurement_family,
            )
            formal_charge = _molecule_formal_charge(mol)
            neutral_input_risk = bool(
                group_mode == "pair_type"
                and measurement_family == "amine"
                and pair_member_form == "base_form"
                and formal_charge == 0
            )

            if group_mode == "pair_type":
                summary_label = measurement_group
            elif measurement_family in (ACIDIC_FAMILIES | BASIC_FAMILIES):
                summary_label = measurement_canonical
            else:
                summary_label = measurement_refined

            dist_ok, dist_reason = distribution_sanity_check(
                group_mode=group_mode,
                pair_member_form=pair_member_form,
                summary_label=summary_label,
                pka_value=pka_value,
            )
            pka_type_effective, two_faced_type_flip = reinterpret_pka_type_for_two_faced(
                pka_type=pka_type_canonical,
                group_family=measurement_family,
            )
            conjugate_flag = conjugate_check_flag(
                group_family=measurement_family,
                pka_type=pka_type_canonical,
                pka_value=pka_value,
            )

            all_rows.append(
                {
                    "source_file": source_file,
                    "record_index": record_idx,
                    "smiles": smiles,
                    "pka_value": pka_value,
                    "pka_source_method": measurement["pka_source_method"],
                    "pka_type_canonical": pka_type_canonical,
                    "pka_type_effective": pka_type_effective,
                    "two_faced_type_flip": two_faced_type_flip,
                    "atom_site_match": atom_meta["atom_site_match"],
                    "resolved_group_count": len(resolved_sites),
                    "rejected_group_count": len(rejected_sites),
                    "assignment_status": status,
                    "group_mode": group_mode,
                    "pair_family": pair_family,
                    "pair_member_form": pair_member_form,
                    "pair_assignment_confidence": pair_meta["pair_assignment_confidence"],
                    "final_group_label": measurement_group,
                    "final_group_summary_label": summary_label,
                    "final_group_family": measurement_family,
                    "molecule_formal_charge": formal_charge,
                    "neutral_input_risk": neutral_input_risk,
                    "distribution_ok": dist_ok,
                    "distribution_reason": dist_reason,
                    "conjugate_check_flag": conjugate_flag,
                }
            )

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["pka_value"] = pd.to_numeric(df["pka_value"], errors="coerce")
    df = df.dropna(subset=["pka_value", "smiles"]) 
    df = df[(df["pka_value"] > -5) & (df["pka_value"] < 20)]
    df = df.drop_duplicates(subset=["source_file", "record_index", "pka_value", "smiles"]).copy()
    df["molecule_key"] = (
        df["source_file"].astype(str)
        + "::"
        + df["record_index"].astype(int).astype(str)
        + "::"
        + df["smiles"].astype(str)
    )
    df["single_group_strict"] = df["assignment_status"] == "single_group"
    return df


def _safe_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.mean())


def summarize_overview(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for source_file, g in df.groupby("source_file", sort=True):
        out_rows.append(
            {
                "source_file": source_file,
                "is_incorrect_prefixed": bool(str(source_file).startswith("INCORRECT_")),
                "rows": int(len(g)),
                "molecules": int(g["molecule_key"].nunique()),
                "single_group_rows": int((g["assignment_status"] == "single_group").sum()),
                "multi_group_rows": int((g["assignment_status"] == "multi_group").sum()),
                "no_group_rows": int((g["assignment_status"] == "no_group").sum()),
                "pair_type_rows": int((g["group_mode"] == "pair_type").sum()),
                "distribution_ok_rate": _safe_rate(g["distribution_ok"].astype(float)),
                "atom_site_match_rate": _safe_rate(g["atom_site_match"].astype(float)),
                "neutral_input_risk_rate": _safe_rate(g["neutral_input_risk"].astype(float)),
                "unique_group_labels": int(g["final_group_label"].nunique(dropna=True)),
                "unique_group_families": int(g["final_group_family"].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(out_rows).sort_values(["rows", "source_file"], ascending=[False, True])


def summarize_group_contrib(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["source_file", "final_group_label"], dropna=False)
        .agg(
            rows=("pka_value", "size"),
            molecules=("molecule_key", "nunique"),
            single_group_rows=("single_group_strict", "sum"),
            distribution_ok_rate=("distribution_ok", "mean"),
            mean_pka=("pka_value", "mean"),
            median_pka=("pka_value", "median"),
        )
        .reset_index()
    )
    return agg.sort_values(["rows", "source_file"], ascending=[False, True])


def summarize_family_contrib(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(["source_file", "final_group_family"], dropna=False)
        .agg(
            rows=("pka_value", "size"),
            molecules=("molecule_key", "nunique"),
            single_group_rows=("single_group_strict", "sum"),
            distribution_ok_rate=("distribution_ok", "mean"),
            mean_pka=("pka_value", "mean"),
            median_pka=("pka_value", "median"),
        )
        .reset_index()
    )
    return agg.sort_values(["rows", "source_file"], ascending=[False, True])


def summarize_uniques(contrib: pd.DataFrame, key_col: str, out_col_name: str) -> pd.DataFrame:
    owners: Dict[str, List[str]] = defaultdict(list)
    for _, row in contrib[["source_file", key_col]].dropna().drop_duplicates().iterrows():
        owners[str(row[key_col])].append(str(row["source_file"]))

    unique_items = {item: srcs[0] for item, srcs in owners.items() if len(srcs) == 1}
    if not unique_items:
        return pd.DataFrame(columns=["source_file", out_col_name, "rows", "molecules", "distribution_ok_rate"])

    rows = []
    for item, source_file in unique_items.items():
        subset = contrib[(contrib["source_file"] == source_file) & (contrib[key_col].astype(str) == item)]
        if subset.empty:
            continue
        entry = subset.iloc[0]
        rows.append(
            {
                "source_file": source_file,
                out_col_name: item,
                "rows": int(entry["rows"]),
                "molecules": int(entry["molecules"]),
                "distribution_ok_rate": float(entry["distribution_ok_rate"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["rows", "source_file"], ascending=[False, True])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit per-dataset contribution of raw SDFs to group/family coverage.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed/dataset_audit")
    parser.add_argument("--overlap-threshold", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sdf_files = sorted(glob.glob(os.path.join(args.raw_dir, "*.sdf")))
    if not sdf_files:
        raise FileNotFoundError(f"No .sdf files found in {args.raw_dir}")

    parts: List[pd.DataFrame] = []
    for sdf in sdf_files:
        print(f"Auditing {os.path.basename(sdf)}")
        part = build_rows_for_sdf(sdf, overlap_threshold=float(args.overlap_threshold))
        if not part.empty:
            parts.append(part)

    if not parts:
        raise RuntimeError("No pKa-bearing rows found across raw SDF files.")

    all_df = pd.concat(parts, ignore_index=True)

    overview = summarize_overview(all_df)
    group_contrib = summarize_group_contrib(all_df)
    family_contrib = summarize_family_contrib(all_df)
    unique_groups = summarize_uniques(group_contrib, key_col="final_group_label", out_col_name="unique_group_label")
    unique_families = summarize_uniques(family_contrib, key_col="final_group_family", out_col_name="unique_group_family")

    all_df.to_csv(os.path.join(args.out_dir, "dataset_audit_rows.csv"), index=False)
    overview.to_csv(os.path.join(args.out_dir, "dataset_overview.csv"), index=False)
    group_contrib.to_csv(os.path.join(args.out_dir, "dataset_group_contributions.csv"), index=False)
    family_contrib.to_csv(os.path.join(args.out_dir, "dataset_family_contributions.csv"), index=False)
    unique_groups.to_csv(os.path.join(args.out_dir, "dataset_unique_group_contributions.csv"), index=False)
    unique_families.to_csv(os.path.join(args.out_dir, "dataset_unique_family_contributions.csv"), index=False)

    print(f"Saved audit rows: {os.path.join(args.out_dir, 'dataset_audit_rows.csv')}")
    print(f"Saved overview: {os.path.join(args.out_dir, 'dataset_overview.csv')}")
    print(f"Saved group contributions: {os.path.join(args.out_dir, 'dataset_group_contributions.csv')}")
    print(f"Saved family contributions: {os.path.join(args.out_dir, 'dataset_family_contributions.csv')}")
    print(f"Saved unique groups: {os.path.join(args.out_dir, 'dataset_unique_group_contributions.csv')}")
    print(f"Saved unique families: {os.path.join(args.out_dir, 'dataset_unique_family_contributions.csv')}")


if __name__ == "__main__":
    main()
