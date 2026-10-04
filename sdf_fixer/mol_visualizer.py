"""
mol_visualizer.py
=================
Render a molecule as a PNG image with atom indices labelled and
ionizable functional groups colour-highlighted.  Designed for the
interactive fixer CLI where the user needs to see which atom to
protonate/deprotonate.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional, Set, Tuple

from rdkit import Chem  # type: ignore
from rdkit.Chem import AllChem, Draw  # type: ignore
from rdkit.Chem.Draw import rdMolDraw2D  # type: ignore

from sdf_fixer.group_identifier import GroupIdentification, IonizableSite


# ---------------------------------------------------------------------------
# Colour palette (same as visualize_complex_group_assignments.py)
# ---------------------------------------------------------------------------

PALETTE: List[Tuple[float, float, float]] = [
    (0.90, 0.20, 0.20),   # red
    (0.20, 0.50, 0.90),   # blue
    (0.20, 0.70, 0.30),   # green
    (0.90, 0.60, 0.20),   # orange
    (0.70, 0.30, 0.90),   # purple
    (0.20, 0.70, 0.70),   # teal
    (0.80, 0.40, 0.50),   # pink
]


# ---------------------------------------------------------------------------
# Atom-index labelling
# ---------------------------------------------------------------------------

def _label_atoms_with_indices(mol: Chem.Mol) -> None:
    """Set each atom's ``molAtomMapNumber`` to its index so that the 2-D
    drawing displays atom numbers."""
    for atom in mol.GetAtoms():
        atom.SetIntProp("molAtomMapNumber", atom.GetIdx())


# ---------------------------------------------------------------------------
# Core renderer
# ---------------------------------------------------------------------------

def render_mol_png(
    mol: Chem.Mol,
    gid: GroupIdentification,
    out_path: str,
    *,
    pka_value: Optional[float] = None,
    record_index: Optional[int] = None,
    width: int = 800,
    height: int = 500,
) -> str:
    """
    Render *mol* to a PNG file at *out_path* with:
      - atom indices printed on every atom
      - each ionizable group highlighted in a distinct colour
      - a legend listing group labels and their atoms

    Parameters
    ----------
    mol : Chem.Mol
    gid : GroupIdentification
    out_path : str
        Where to write the PNG.
    pka_value : float, optional
        Shown in the legend.
    record_index : int, optional
        Shown in the legend.
    width, height : int
        Image dimensions.

    Returns
    -------
    str  — the path written (same as *out_path*).
    """
    # Work on a copy so we don't mutate the original.
    draw_mol = Chem.RWMol(mol)
    try:
        AllChem.Compute2DCoords(draw_mol)
    except Exception:
        pass

    _label_atoms_with_indices(draw_mol)

    # Build highlight maps.
    highlight_atoms: List[int] = []
    highlight_atom_colors: Dict[int, Tuple[float, float, float, float]] = {}
    highlight_radii: Dict[int, float] = {}

    for site_idx, site in enumerate(gid.ionizable_sites):
        r, g, b = PALETTE[site_idx % len(PALETTE)]
        for atom_idx in site.atom_set:
            highlight_atoms.append(atom_idx)
            highlight_atom_colors[atom_idx] = [(r, g, b, 0.45)]
            highlight_radii[atom_idx] = 0.4

    # Also softly highlight non-ionizable resolved groups in grey.
    for site in gid.non_ionizable_sites:
        for atom_idx in site.get("atom_set", set()):
            if atom_idx not in highlight_atom_colors:
                highlight_atoms.append(atom_idx)
                highlight_atom_colors[atom_idx] = [(0.7, 0.7, 0.7, 0.25)]
                highlight_radii[atom_idx] = 0.3

    # De-duplicate while preserving order.
    seen: Set[int] = set()
    unique_highlights: List[int] = []
    for idx in highlight_atoms:
        if idx not in seen:
            seen.add(idx)
            unique_highlights.append(idx)
    highlight_atoms = unique_highlights

    # ---- Draw with rdMolDraw2D ----
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    opts.additionalAtomLabelPadding = 0.15
    opts.annotationFontScale = 0.7
    opts.atomHighlightsAreCircles = True

    drawer.DrawMoleculeWithHighlights(
        draw_mol,
        _build_legend_text(gid, pka_value, record_index),
        dict(highlight_atom_colors),
        {},        # bond highlights
        highlight_radii,
        {},        # bond highlight widths
    )
    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(png_data)

    return out_path


def _build_legend_text(
    gid: GroupIdentification,
    pka_value: Optional[float],
    record_index: Optional[int],
) -> str:
    from sdf_fixer.group_identifier import infer_pka_label
    pka_tag = infer_pka_label(gid)
    parts: List[str] = []
    if record_index is not None:
        parts.append(f"Rec #{record_index}")
    if pka_value is not None:
        parts.append(f"{pka_tag}={pka_value}")
    parts.append(f"charge={gid.formal_charge}")
    for idx, site in enumerate(gid.ionizable_sites, 1):
        eff_ref = site.effective_reference_pka
        ref = f"ref {site.pka_label}≈{eff_ref:.0f}" if eff_ref is not None else ""
        parts.append(f"  {idx}. {site.label} atoms={sorted(site.atom_set)} {ref}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI-friendly helper
# ---------------------------------------------------------------------------

_VIS_DIR: Optional[str] = None


def get_vis_dir() -> str:
    """Return (and lazily create) a temp directory for visualizations."""
    global _VIS_DIR
    if _VIS_DIR is None:
        _VIS_DIR = tempfile.mkdtemp(prefix="sdf_fixer_vis_")
    return _VIS_DIR


def render_for_user(
    mol: Chem.Mol,
    gid: GroupIdentification,
    record_index: int,
    pka_value: Optional[float] = None,
) -> str:
    """Render to a PNG in the session temp directory and return the path."""
    vis_dir = get_vis_dir()
    out_path = os.path.join(vis_dir, f"record_{record_index:05d}.png")
    render_mol_png(
        mol, gid, out_path,
        pka_value=pka_value,
        record_index=record_index,
    )
    return out_path


# ---------------------------------------------------------------------------
# Text-only fallback: ASCII atom table
# ---------------------------------------------------------------------------

def atom_table_str(mol: Chem.Mol, gid: GroupIdentification) -> str:
    """
    Return a human-readable table listing every atom with its index,
    element, charge, H-count, and which ionizable group(s) it belongs to.
    """
    # Build atom→site mapping.
    atom_group_map: Dict[int, List[str]] = {}
    for site_idx, site in enumerate(gid.ionizable_sites, 1):
        tag = f"{site_idx}:{site.label}"
        for aidx in site.atom_set:
            atom_group_map.setdefault(aidx, []).append(tag)

    lines = [
        f"{'Idx':>4s}  {'Elem':4s}  {'Chg':>3s}  {'Hs':>3s}  {'Arom':4s}  Groups",
        f"{'----':>4s}  {'----':4s}  {'---':>3s}  {'---':>3s}  {'----':4s}  ------",
    ]
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        elem = atom.GetSymbol()
        chg = atom.GetFormalCharge()
        hs = atom.GetTotalNumHs()
        arom = "yes" if atom.GetIsAromatic() else ""
        groups = ", ".join(atom_group_map.get(idx, []))
        chg_str = f"{chg:+d}" if chg != 0 else "0"
        lines.append(f"{idx:>4d}  {elem:4s}  {chg_str:>3s}  {hs:>3d}  {arom:4s}  {groups}")
    return "\n".join(lines)
