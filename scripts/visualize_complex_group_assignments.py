import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd # type: ignore
from rdkit import Chem # type: ignore
from rdkit.Chem import Draw # type: ignore

from substructure_match import find_sites_with_metadata, resolve_overlapping_sites


def _select_complex_smiles(assignments_csv: str, top_n: int = 24) -> pd.DataFrame:
    df = pd.read_csv(assignments_csv)
    complex_df = df[df["assignment_status"] == "multi_group"].copy()
    complex_df = complex_df.sort_values(
        ["resolved_group_count", "rejected_group_count"],
        ascending=False,
    )
    complex_df = complex_df.drop_duplicates(subset=["smiles"]).head(top_n)
    return complex_df[["smiles", "resolved_group_count", "resolved_groups", "source_file"]]


def _collect_site_annotations(smiles: str) -> Tuple[Chem.Mol, List[Dict]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    candidates = find_sites_with_metadata(mol)
    resolved, _ = resolve_overlapping_sites(candidates, overlap_threshold=0.5)
    return mol, resolved


def _build_highlight_data(resolved: List[Dict]) -> Tuple[List[int], Dict[int, Tuple[float, float, float]], str]:
    palette = [
        (0.9, 0.2, 0.2),
        (0.2, 0.5, 0.9),
        (0.2, 0.7, 0.3),
        (0.9, 0.6, 0.2),
        (0.7, 0.3, 0.9),
        (0.2, 0.7, 0.7),
        (0.8, 0.4, 0.5),
    ]

    atom_colors: Dict[int, Tuple[float, float, float]] = {}
    legend_parts = []
    for idx, site in enumerate(resolved):
        color = palette[idx % len(palette)]
        legend_parts.append(f"{idx+1}:{site['label']}")
        for atom_idx in site["atom_set"]:
            atom_colors[int(atom_idx)] = color

    highlight_atoms = sorted(atom_colors.keys())
    legend = " | ".join(legend_parts)
    return highlight_atoms, atom_colors, legend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render complex multi-group assignment examples with resolved SMARTS sites highlighted."
    )
    parser.add_argument(
        "--assignments",
        default="data/processed/functional_group_assignments_pruned_eval_unpooled_families.csv",
        help="Assignments CSV used to select representative multi-group examples",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/complex_group_visualization",
        help="Output directory for complex-group visualization artifacts",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=24,
        help="Maximum number of unique multi-group molecules to render",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = _select_complex_smiles(str(args.assignments), top_n=int(args.top_n))

    summary_rows = []
    grid_mols = []
    grid_legends = []
    grid_highlight_atoms = []
    grid_highlight_colors = []

    for row_idx, row in selected.iterrows():
        smiles = row["smiles"]
        mol, resolved = _collect_site_annotations(smiles)
        highlight_atoms, atom_colors, legend = _build_highlight_data(resolved)

        record_id = len(summary_rows) + 1
        summary_rows.append(
            {
                "record_id": record_id,
                "smiles": smiles,
                "source_file": row["source_file"],
                "resolved_group_count": int(row["resolved_group_count"]),
                "resolved_groups": row["resolved_groups"],
                "site_legend": legend,
            }
        )

        grid_mols.append(mol)
        grid_legends.append(f"ID {record_id} | n={int(row['resolved_group_count'])}\n{legend}")
        grid_highlight_atoms.append(highlight_atoms)
        grid_highlight_colors.append(atom_colors)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "complex_group_examples.csv"
    summary_df.to_csv(summary_csv, index=False)

    if grid_mols:
        image = Draw.MolsToGridImage(
            grid_mols,
            molsPerRow=4,
            subImgSize=(420, 320),
            legends=grid_legends,
            highlightAtomLists=grid_highlight_atoms,
            highlightAtomColors=grid_highlight_colors,
            useSVG=False,
        )
        image_path = out_dir / "complex_group_examples.png"
        image.save(str(image_path))

    print(f"Saved: {summary_csv}")
    print(f"Saved: {out_dir / 'complex_group_examples.png'}")


if __name__ == "__main__":
    main()
