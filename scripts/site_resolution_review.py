import argparse
import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem  # type: ignore
from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore

from functional_group_pka_analysis import (
    SITE_OVERRIDE_KEY_COLUMNS,
    _iter_sdf_mols,
    _site_override_key,
    _sites_matching_atom_index,
    build_functional_group_table,
)
from substructure_match import find_sites_with_metadata, maximal_site_rank_key, resolve_overlapping_sites
from training_data_resolver import (
    DEFAULT_OVERLAP_THRESHOLD,
    DEFAULT_RAW_DIR,
    DEFAULT_SITE_OVERRIDES_PATH,
    TRUSTED_TRAINING_SDF_FILES,
)


PALETTE: List[Tuple[float, float, float]] = [
    (0.90, 0.20, 0.20),
    (0.20, 0.50, 0.90),
    (0.20, 0.70, 0.30),
    (0.90, 0.60, 0.20),
    (0.70, 0.30, 0.90),
    (0.20, 0.70, 0.70),
    (0.80, 0.40, 0.50),
]

DEFAULT_QUEUE_CSV = "data/processed/site_resolution_review_queue.csv"
DEFAULT_IMAGE_DIR = "data/processed/site_resolution_review_images"


def _reason_tags(row: pd.Series) -> List[str]:
    reasons: List[str] = []
    if str(row.get("assignment_status", "")) != "single_group":
        reasons.append(str(row.get("assignment_status", "")) or "non_single_group")
    if int(pd.to_numeric(row.get("atom_candidate_count", 0), errors="coerce") or 0) > 1:
        reasons.append("multiple_atom_matches")
    if bool(row.get("atom_index_raw") == row.get("atom_index_raw")) and not bool(row.get("atom_site_match", False)):
        if int(pd.to_numeric(row.get("resolved_group_count", 0), errors="coerce") or 0) > 1:
            reasons.append("atom_index_unmatched")
    if str(row.get("pair_assignment_confidence", "")) == "ambiguous":
        reasons.append("pair_confidence_ambiguous")
    resolved_labels = [token for token in str(row.get("resolved_groups", "")).split("|") if token]
    if len(set(resolved_labels)) > 1:
        reasons.append("multiple_resolved_labels")
    return reasons


def build_site_resolution_review_queue(
    raw_dir: str = DEFAULT_RAW_DIR,
    include_files: Optional[List[str]] = None,
    overrides_path: str = DEFAULT_SITE_OVERRIDES_PATH,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    include_reviewed: bool = False,
) -> pd.DataFrame:
    df = build_functional_group_table(
        raw_dir=raw_dir,
        overlap_threshold=overlap_threshold,
        include_files=include_files,
        allow_epik=False,
        overrides_path=overrides_path,
    )
    if df.empty:
        return df

    work = df.copy()
    work["review_reason"] = work.apply(lambda row: "|".join(_reason_tags(row)), axis=1)
    queue = work[work["review_reason"] != ""].copy()
    if not include_reviewed and "site_override_applied" in queue.columns:
        queue = queue[~queue["site_override_applied"].astype(bool)].copy()
    queue = queue.drop_duplicates(subset=SITE_OVERRIDE_KEY_COLUMNS).copy()
    queue = queue.sort_values(["source_file", "record_index", "pka_value"]).reset_index(drop=True)
    return queue


class RawMolCache:
    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir
        self._cache: Dict[str, List[Optional[Chem.Mol]]] = {}

    def get(self, source_file: str, record_index: int) -> Chem.Mol:
        path = os.path.join(self.raw_dir, source_file)
        if path not in self._cache:
            # Preserve unreadable records as None so list positions remain the
            # actual zero-based SDF record indices.
            self._cache[path] = list(Chem.SDMolSupplier(path, removeHs=False))
        mols = self._cache[path]
        if record_index < 0 or record_index >= len(mols):
            raise IndexError(f"Record index {record_index} out of range for {source_file}")
        mol = mols[record_index]
        if mol is None:
            raise ValueError(f"Unreadable SDF record {source_file}#{record_index}")
        return Chem.Mol(mol)


def _ordered_candidates(mol: Chem.Mol, row: pd.Series, overlap_threshold: float) -> List[Dict[str, object]]:
    resolved_sites, _ = resolve_overlapping_sites(
        find_sites_with_metadata(mol),
        overlap_threshold=overlap_threshold,
    )
    atom_matches, atom_meta = _sites_matching_atom_index(
        resolved_sites,
        None if pd.isna(row.get("atom_index_raw")) else int(float(row.get("atom_index_raw"))),
    )
    ranked = sorted(resolved_sites, key=maximal_site_rank_key, reverse=True)
    ordered = atom_matches + [site for site in ranked if site not in atom_matches]

    current_label = str(row.get("selected_group_label", "") or "")
    current_auto_label = str(row.get("auto_selected_group_label", current_label) or "")

    candidates: List[Dict[str, object]] = []
    for candidate_index, site in enumerate(ordered):
        candidates.append(
            {
                "candidate_index": candidate_index,
                "label": str(site.get("label", "")),
                "family": str(site.get("family", "")),
                "site_type": str(site.get("type", "")),
                "atom_set": set(site.get("atom_set", set())),
                "atom_text": ",".join(str(atom_idx) for atom_idx in sorted(site.get("atom_set", set()))),
                "priority": int(site.get("priority", 0)),
                "specificity": int(site.get("specificity", 0)),
                "atom_index_hit": bool(site in atom_matches),
                "is_current": str(site.get("label", "")) == current_label,
                "is_auto": str(site.get("label", "")) == current_auto_label,
                "atom_meta": atom_meta,
            }
        )
    return candidates


def _try_open_image(path: str) -> bool:
    system = platform.system()
    if system == "Darwin":
        openers = ["open"]
    elif system == "Linux":
        openers = ["xdg-open", "eog", "feh", "display"]
    else:
        openers = []

    for opener in openers:
        try:
            subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            continue
    return False


def _render_review_png(
    mol: Chem.Mol,
    row: pd.Series,
    candidates: List[Dict[str, object]],
    out_path: str,
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    draw_mol = Chem.RWMol(mol)
    try:
        AllChem.Compute2DCoords(draw_mol)
    except Exception:
        pass

    for atom in draw_mol.GetAtoms():
        atom.SetIntProp("molAtomMapNumber", atom.GetIdx())

    highlight_atoms: List[int] = []
    atom_colors: Dict[int, Tuple[float, float, float]] = {}
    atom_radii: Dict[int, float] = {}
    legend_lines = [
        f"{row['source_file']}#{int(row['record_index'])} pKa={float(row['pka_value']):.2f}",
        f"auto={row.get('auto_selected_group_label', row.get('selected_group_label', ''))} current={row.get('selected_group_label', '')}",
        f"reason={row.get('review_reason', '')}",
    ]

    for idx, candidate in enumerate(candidates):
        color = PALETTE[idx % len(PALETTE)]
        marker_bits = []
        if candidate["atom_index_hit"]:
            marker_bits.append("atom-hit")
        if candidate["is_auto"]:
            marker_bits.append("auto")
        if candidate["is_current"]:
            marker_bits.append("current")
        if candidate.get("pair_supported"):
            marker_bits.append("pair-supported")
        marker_text = f" [{' '.join(marker_bits)}]" if marker_bits else ""
        legend_lines.append(
            f"[{candidate['candidate_index']}] {candidate['label']} ({candidate['site_type']}, {candidate['family']}) atoms={candidate['atom_text']}{marker_text}"
        )
        for atom_idx in sorted(candidate["atom_set"]):
            highlight_atoms.append(atom_idx)
            atom_colors.setdefault(atom_idx, color)
            atom_radii[atom_idx] = 0.35

    unique_highlights = list(dict.fromkeys(highlight_atoms))
    legend = "\n".join(legend_lines)

    drawer = rdMolDraw2D.MolDraw2DCairo(1000, 700)
    options = drawer.drawOptions()
    options.legendFontSize = 18
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        draw_mol,
        legend=legend,
        highlightAtoms=unique_highlights,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    with open(out_path, "wb") as handle:
        handle.write(drawer.GetDrawingText())
    return out_path


def _override_columns() -> List[str]:
    return SITE_OVERRIDE_KEY_COLUMNS + [
        "override_group_label", "note", "reviewed_at", "override_atom_index_raw",
    ]


def _load_overrides(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=_override_columns())
    for column in _override_columns():
        if column not in df.columns:
            df[column] = ""
    return df[_override_columns()].copy()


def _save_overrides(path: str, overrides: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    work = overrides.copy()
    work = work.drop_duplicates(subset=SITE_OVERRIDE_KEY_COLUMNS, keep="last")
    work.to_csv(path, index=False)


def _upsert_override(
    overrides: pd.DataFrame,
    row: pd.Series,
    override_group_label: str,
    note: str = "",
    override_atom_index_raw: Optional[int] = None,
) -> pd.DataFrame:
    work = overrides.copy()
    key = _site_override_key(row)
    if not work.empty:
        work = work[work.apply(_site_override_key, axis=1) != key].copy()
    payload = {column: row.get(column) for column in SITE_OVERRIDE_KEY_COLUMNS}
    payload["override_group_label"] = override_group_label
    payload["note"] = note
    payload["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    payload["override_atom_index_raw"] = (
        "" if override_atom_index_raw is None else int(override_atom_index_raw)
    )
    return pd.concat([work, pd.DataFrame([payload])], ignore_index=True)


def _drop_override(overrides: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    if overrides.empty:
        return overrides
    key = _site_override_key(row)
    return overrides[overrides.apply(_site_override_key, axis=1) != key].copy()


def _print_row_summary(row: pd.Series, candidates: List[Dict[str, object]], png_path: str) -> None:
    print("=" * 100)
    print(f"[{row.name + 1}] {row['source_file']} record={int(row['record_index'])} pKa={float(row['pka_value']):.2f}")
    print(f"smiles: {row['smiles']}")
    print(f"reason: {row.get('review_reason', '')}")
    print(f"resolved groups: {row.get('resolved_groups', '')}")
    print(f"auto label: {row.get('auto_selected_group_label', row.get('selected_group_label', ''))}")
    print(f"current label: {row.get('selected_group_label', '')}")
    print(f"atom index raw: {row.get('atom_index_raw', '')} | atom candidates: {row.get('atom_candidate_labels', '')}")
    print(f"image: {png_path}")
    print("candidates:")
    for candidate in candidates:
        flag_bits = []
        if candidate["atom_index_hit"]:
            flag_bits.append("atom-hit")
        if candidate["is_auto"]:
            flag_bits.append("auto")
        if candidate["is_current"]:
            flag_bits.append("current")
        flag_text = f" [{' '.join(flag_bits)}]" if flag_bits else ""
        print(
            f"  {candidate['candidate_index']}: {candidate['label']} ({candidate['site_type']}, {candidate['family']})"
            f" atoms={candidate['atom_text']}{flag_text}"
        )
    print("commands: <index> choose candidate | k keep auto/no override | m <label> manual label | s skip | q quit")


def review_site_resolution(
    raw_dir: str = DEFAULT_RAW_DIR,
    include_files: Optional[List[str]] = None,
    overrides_path: str = DEFAULT_SITE_OVERRIDES_PATH,
    queue_csv: str = DEFAULT_QUEUE_CSV,
    image_dir: str = DEFAULT_IMAGE_DIR,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    max_records: int = 0,
    include_reviewed: bool = False,
    export_only: bool = False,
    open_images: bool = True,
) -> Tuple[str, str, int]:
    queue = build_site_resolution_review_queue(
        raw_dir=raw_dir,
        include_files=include_files,
        overrides_path=overrides_path,
        overlap_threshold=overlap_threshold,
        include_reviewed=include_reviewed,
    )
    if max_records > 0:
        queue = queue.head(int(max_records)).copy()

    os.makedirs(os.path.dirname(queue_csv), exist_ok=True)
    queue.to_csv(queue_csv, index=False)
    print(f"Saved review queue: {queue_csv} ({len(queue)} rows)")
    print(f"Overrides path: {overrides_path}")

    if export_only or queue.empty:
        return overrides_path, queue_csv, len(queue)

    overrides = _load_overrides(overrides_path)
    mol_cache = RawMolCache(raw_dir=raw_dir)

    for _, row in queue.iterrows():
        mol = mol_cache.get(str(row["source_file"]), int(row["record_index"]))
        candidates = _ordered_candidates(mol, row, overlap_threshold=overlap_threshold)
        image_name = f"{os.path.splitext(str(row['source_file']))[0]}_rec{int(row['record_index'])}_pka{float(row['pka_value']):.2f}.png"
        png_path = _render_review_png(mol, row, candidates, os.path.join(image_dir, image_name))
        if open_images:
            _try_open_image(png_path)
        _print_row_summary(row, candidates, png_path)

        while True:
            command = input("review> ").strip()
            if not command:
                continue
            if command.lower() in {"q", "quit", "exit"}:
                _save_overrides(overrides_path, overrides)
                return overrides_path, queue_csv, len(queue)
            if command.lower() in {"s", "skip"}:
                break
            if command.lower() in {"k", "keep"}:
                overrides = _drop_override(overrides, row)
                _save_overrides(overrides_path, overrides)
                print("  kept current auto assignment; removed any existing override")
                break
            if command.isdigit():
                choice = int(command)
                candidate_map = {int(candidate["candidate_index"]): candidate for candidate in candidates}
                if choice not in candidate_map:
                    print("  invalid candidate index")
                    continue
                selected = candidate_map[choice]
                selected_atoms = sorted(int(value) for value in selected["atom_set"])
                overrides = _upsert_override(
                    overrides,
                    row,
                    str(selected["label"]),
                    override_atom_index_raw=(selected_atoms[0] if selected_atoms else None),
                )
                _save_overrides(overrides_path, overrides)
                print(
                    f"  override saved: {selected['label']} "
                    f"at atoms={selected['atom_text']}"
                )
                break
            if command.lower().startswith("m "):
                manual_label = command[2:].strip()
                if not manual_label:
                    print("  manual label cannot be empty")
                    continue
                overrides = _upsert_override(overrides, row, manual_label, note="manual")
                _save_overrides(overrides_path, overrides)
                print(f"  manual override saved: {manual_label}")
                break
            print("  unrecognized command")

    _save_overrides(overrides_path, overrides)
    return overrides_path, queue_csv, len(queue)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review ambiguous site assignments and write override labels.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--include-files",
        nargs="*",
        default=list(TRUSTED_TRAINING_SDF_FILES),
        help="Specific raw SDF files to review. Default is the trusted training bundle.",
    )
    parser.add_argument("--overrides", default=DEFAULT_SITE_OVERRIDES_PATH)
    parser.add_argument("--queue-csv", default=DEFAULT_QUEUE_CSV)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--overlap-threshold", type=float, default=DEFAULT_OVERLAP_THRESHOLD)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--include-reviewed", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_files = args.include_files or None
    review_site_resolution(
        raw_dir=args.raw_dir,
        include_files=include_files,
        overrides_path=args.overrides,
        queue_csv=args.queue_csv,
        image_dir=args.image_dir,
        overlap_threshold=args.overlap_threshold,
        max_records=args.max_records,
        include_reviewed=bool(args.include_reviewed),
        export_only=bool(args.export_only),
        open_images=not bool(args.no_open),
    )


if __name__ == "__main__":
    main()
