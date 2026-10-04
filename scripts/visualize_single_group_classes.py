import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem import Draw  # type: ignore

from substructure_match import find_sites_with_metadata, resolve_overlapping_sites

REFINED_TO_BASE = {
    "amine_aromatic": ["aniline", "primary_amine", "secondary_amine", "tertiary_amine"],
    "amine_aliphatic": ["primary_amine", "secondary_amine", "tertiary_amine"],
    "ammonium_aromatic": ["aryl_ammonium"],
    "ammonium_aliphatic": [
        "primary_ammonium",
        "secondary_ammonium",
        "tertiary_ammonium",
    ],
    "amide_aromatic": ["amide", "amide_NH"],
    "amide_aliphatic": ["amide", "amide_NH"],
    "ester_aromatic": ["ester"],
    "ester_aliphatic": ["ester"],
}


def _collect_sites(smiles: str) -> Tuple[Optional[Chem.Mol], List[Dict]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, []
    candidates = find_sites_with_metadata(mol)
    resolved, _ = resolve_overlapping_sites(candidates, overlap_threshold=0.5)
    return mol, resolved


def _pick_site(
    resolved_sites: List[Dict],
    final_group_label: str,
    final_group_refined: Optional[str],
) -> Optional[Dict]:
    if not resolved_sites:
        return None

    target_labels = [final_group_label]
    if final_group_refined:
        target_labels.append(final_group_refined)
        target_labels.extend(REFINED_TO_BASE.get(final_group_refined, []))

    for site in resolved_sites:
        if site.get("label") in target_labels:
            return site

    return sorted(
        resolved_sites,
        key=lambda x: (x["priority"], x["specificity"], len(x["atom_set"])),
        reverse=True,
    )[0]


def _render_class_examples(class_df: pd.DataFrame, out_png: Path, out_csv: Path, max_examples: int = 16) -> None:
    sample_df = class_df.drop_duplicates(subset=["smiles"]).head(max_examples).copy()

    mols = []
    legends = []
    highlight_atoms_list = []
    highlight_colors_list = []
    rows = []

    for idx, row in sample_df.iterrows():
        smiles = row["smiles"]
        mol, resolved = _collect_sites(smiles)
        if mol is None:
            continue

        chosen_site = _pick_site(
            resolved_sites=resolved,
            final_group_label=row["final_group_label"],
            final_group_refined=row.get("final_group_refined", None),
        )

        highlight_atoms = []
        highlight_colors = {}
        chosen_label = None

        if chosen_site is not None:
            chosen_label = chosen_site.get("label")
            highlight_atoms = sorted(list(chosen_site.get("atom_set", set())))
            for atom_idx in highlight_atoms:
                highlight_colors[int(atom_idx)] = (0.9, 0.2, 0.2)

        mols.append(mol)
        legends.append(
            f"pKa={row['pka_value']:.2f} | group={row['final_group_label']}\nsite={chosen_label or 'NA'}"
        )
        highlight_atoms_list.append(highlight_atoms)
        highlight_colors_list.append(highlight_colors)

        rows.append(
            {
                "smiles": smiles,
                "pka_value": row["pka_value"],
                "final_group_label": row["final_group_label"],
                "final_group_refined": row.get("final_group_refined", None),
                "resolved_groups": row.get("resolved_groups", None),
                "chosen_site_label": chosen_label,
                "source_file": row.get("source_file", None),
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if mols:
        image = Draw.MolsToGridImage(
            mols,
            molsPerRow=4,
            subImgSize=(420, 280),
            legends=legends,
            highlightAtomLists=highlight_atoms_list,
            highlightAtomColors=highlight_colors_list,
            useSVG=False,
        )
        image.save(str(out_png))


def _sanitize_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render per-class molecule examples from single-group functional assignments."
    )
    parser.add_argument(
        "--in-csv",
        default="data/processed/functional_group_single_group_subset.csv",
        help="Input single-group subset CSV",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/single_group_class_visualization",
        help="Output directory for class visualizations",
    )
    parser.add_argument(
        "--classes",
        default="all",
        help="Comma-separated class labels to render, or 'all'",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=16,
        help="Maximum number of unique molecules rendered per class",
    )
    args = parser.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if "final_group_label" not in df.columns:
        raise RuntimeError("Expected final_group_label in single-group subset file.")

    available_classes = sorted(df["final_group_label"].dropna().astype(str).unique().tolist())
    if args.classes.strip().lower() == "all":
        target_classes = available_classes
    else:
        requested = [item.strip() for item in args.classes.split(",") if item.strip()]
        target_classes = [label for label in requested if label in set(available_classes)]

    index_rows = []
    for class_name in target_classes:
        class_df = df[df["final_group_label"] == class_name].copy()
        n = len(class_df)

        safe_name = _sanitize_label(class_name)
        png_path = out_dir / f"{safe_name}_examples.png"
        csv_path = out_dir / f"{safe_name}_examples.csv"

        if n > 0:
            _render_class_examples(class_df, png_path, csv_path, max_examples=args.max_examples)

        index_rows.append(
            {
                "class_name": class_name,
                "row_count": n,
                "image_file": str(png_path) if n > 0 else None,
                "table_file": str(csv_path) if n > 0 else None,
            }
        )

    index_df = pd.DataFrame(index_rows)
    index_csv = out_dir / "class_visualization_index.csv"
    index_df.to_csv(index_csv, index=False)

    print(f"Saved index: {index_csv}")
    print(index_df.to_string(index=False))


if __name__ == "__main__":
    main()
