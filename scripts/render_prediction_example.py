#!/usr/bin/env python3
"""Render a README figure showing the user-facing Stage 3 output contract.

The bundled example is a documented snapshot of the current canonical model's
GABA prediction at pH 7.4.  It intentionally distinguishes thermodynamic
population from calibrated confidence and labels the tautomer weight as a
conditional ranking heuristic rather than a physical probability.
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

# Keep documentation rendering usable in containers and read-only CI jobs.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "pkate-matplotlib"),
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams.update({
    # Matplotlib ships Computer Modern even when a TeX installation is absent.
    # DejaVu Serif supplies the few Unicode sub/superscript glyphs that cmr10
    # does not contain; mathematical text still uses Computer Modern proper.
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    # Embed glyph outlines so browsers cannot reinterpret Computer Modern's
    # symbol-font code points (for example, turning a dot into a currency sign).
    "svg.fonttype": "path",
    "svg.hashsalt": "pkate-prediction-example-v1",
})

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import rdDepictor  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402


BACKGROUND = "#F8FAFC"
WHITE = "#FFFFFF"
INK = "#172033"
MUTED = "#586579"
LINE = "#D8E0EA"
CYAN = "#55C5D2"
CYAN_DARK = "#087C8A"
CYAN_PALE = "#E8F9FB"
PINK = "#EF91A5"
PINK_DARK = "#B53B5A"
PINK_PALE = "#FFF0F4"
GREEN = "#138A5B"
GREEN_PALE = "#EAF8F1"


def _rounded_box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = WHITE,
    edgecolor: str = LINE,
    linewidth: float = 1.2,
    radius: float = 0.016,
    zorder: int = 2,
) -> FancyBboxPatch:
    """Draw one rounded card in normalized figure coordinates."""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _molecule_image(
    smiles: str,
    *,
    site_atoms: Sequence[int],
    size: tuple[int, int] = (900, 520),
) -> Image.Image:
    """Render a molecule with the two documented ionization sites highlighted."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Could not parse example SMILES: {smiles}")
    rdDepictor.Compute2DCoords(molecule)
    for site_number, atom_index in enumerate(site_atoms, start=1):
        molecule.GetAtomWithIdx(int(atom_index)).SetProp("atomNote", f"site {site_number}")

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = 0.14
    options.bondLineWidth = 3.0
    options.annotationFontScale = 0.75
    options.fontFile = str(
        Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "cmr10.ttf"
    )
    highlight_colors = {
        int(site_atoms[0]): (0.33, 0.77, 0.82),
        int(site_atoms[1]): (0.94, 0.57, 0.65),
    }
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        molecule,
        highlightAtoms=[int(value) for value in site_atoms],
        highlightAtomColors=highlight_colors,
        highlightAtomRadii={int(value): 0.38 for value in site_atoms},
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGBA")


def _add_molecule_axes(
    figure: Figure,
    bounds: tuple[float, float, float, float],
    image: Image.Image,
) -> None:
    """Place a transparent RDKit depiction inside a figure card."""
    molecule_ax = figure.add_axes(bounds, zorder=4)
    molecule_ax.imshow(image)
    molecule_ax.set_aspect("equal")
    molecule_ax.axis("off")


def _metric(
    ax: Axes,
    x: float,
    y: float,
    label: str,
    value: str,
    *,
    color: str,
) -> None:
    """Draw a compact label/value pair inside a site card."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.3,
        color=MUTED,
        va="top",
        zorder=5,
    )
    ax.text(
        x,
        y - 0.029,
        value,
        transform=ax.transAxes,
        fontsize=9.3,
        fontweight="bold",
        color=color,
        va="top",
        zorder=5,
    )


def build_figure() -> Figure:
    """Build the complete input-to-microstate example figure."""
    figure, ax = plt.subplots(figsize=(14, 7.6), constrained_layout=False)
    # Molecule insets use figure coordinates while cards use axes coordinates.
    # Filling the figure with the main axes makes those systems identical and
    # keeps both structures inside their cards after tight-bbox export.
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.035,
        0.955,
        r"What pK$_a$te returns",
        transform=ax.transAxes,
        fontsize=23,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0.035,
        0.907,
        r"Example $\cdot$ GABA $\cdot$ requested pH 7.4 $\cdot$ current canonical-model snapshot",
        transform=ax.transAxes,
        fontsize=10.5,
        color=MUTED,
        va="top",
    )

    # Input structure.
    _rounded_box(ax, 0.035, 0.350, 0.245, 0.495, edgecolor="#B8C5D6")
    ax.text(
        0.055,
        0.815,
        "INPUT",
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        color="#64748B",
        va="top",
        zorder=5,
    )
    ax.text(
        0.055,
        0.778,
        "One molecular structure",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=INK,
        va="top",
        zorder=5,
    )
    input_image = _molecule_image("NCCCC(=O)O", site_atoms=(0, 6))
    _add_molecule_axes(figure, (0.052, 0.455, 0.210, 0.235), input_image)
    ax.text(
        0.1575,
        0.410,
        "NCCCC(=O)O",
        transform=ax.transAxes,
        fontsize=9.7,
        family="DejaVu Sans Mono",
        color=INK,
        ha="center",
        va="center",
        zorder=5,
    )
    ax.text(
        0.1575,
        0.377,
        "sites are detected and atom-mapped",
        transform=ax.transAxes,
        fontsize=8.0,
        color=MUTED,
        ha="center",
        va="center",
        zorder=5,
    )

    # Requested-pH arrow.
    arrow = FancyArrowPatch(
        (0.292, 0.596),
        (0.348, 0.596),
        arrowstyle="-|>",
        mutation_scale=19,
        linewidth=2.1,
        color="#64748B",
        transform=ax.transAxes,
        clip_on=False,
        zorder=5,
    )
    ax.add_patch(arrow)
    _rounded_box(
        ax,
        0.297,
        0.625,
        0.046,
        0.057,
        facecolor="#EEF2F7",
        edgecolor="#CBD5E1",
        radius=0.028,
        zorder=5,
    )
    ax.text(
        0.320,
        0.654,
        "pH\n7.4",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=INK,
        ha="center",
        va="center",
        linespacing=0.95,
        zorder=6,
    )

    # Dominant atom-mapped microstate.
    _rounded_box(ax, 0.360, 0.350, 0.245, 0.495, edgecolor=GREEN)
    ax.text(
        0.380,
        0.815,
        "DOMINANT COMPLETE MICROSTATE",
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        color=GREEN,
        va="top",
        zorder=5,
    )
    ax.text(
        0.380,
        0.778,
        "Zwitterionic\nconfiguration",
        transform=ax.transAxes,
        fontsize=13.0,
        fontweight="bold",
        color=INK,
        va="top",
        linespacing=1.02,
        zorder=5,
    )
    output_image = _molecule_image("[NH3+]CCCC(=O)[O-]", site_atoms=(0, 6))
    _add_molecule_axes(figure, (0.377, 0.455, 0.210, 0.235), output_image)
    ax.text(
        0.4825,
        0.424,
        "[NH3+:1][CH2:2][CH2:3][CH2:4]",
        transform=ax.transAxes,
        fontsize=7.8,
        family="DejaVu Sans Mono",
        color=INK,
        ha="center",
        va="center",
        zorder=5,
    )
    ax.text(
        0.4825,
        0.397,
        "[C:5](=[O:6])[O-:7]",
        transform=ax.transAxes,
        fontsize=7.8,
        family="DejaVu Sans Mono",
        color=INK,
        ha="center",
        va="center",
        zorder=5,
    )
    ax.text(
        0.4825,
        0.370,
        "configuration population 99.91%",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=GREEN,
        ha="center",
        va="center",
        zorder=5,
    )

    # Site-resolved predictions.
    _rounded_box(ax, 0.630, 0.350, 0.335, 0.495, edgecolor="#B8C5D6")
    ax.text(
        0.650,
        0.815,
        "SITE-RESOLVED OUTPUT",
        transform=ax.transAxes,
        fontsize=8.4,
        fontweight="bold",
        color="#64748B",
        va="top",
        zorder=5,
    )
    ax.text(
        0.650,
        0.778,
        "Per-site state, pK$_a$,\nmarginal, and confidence",
        transform=ax.transAxes,
        fontsize=11.8,
        fontweight="bold",
        color=INK,
        va="top",
        linespacing=1.03,
        zorder=5,
    )

    _rounded_box(
        ax,
        0.650,
        0.550,
        0.295,
        0.150,
        facecolor=CYAN_PALE,
        edgecolor=CYAN,
        radius=0.013,
        zorder=4,
    )
    ax.text(
        0.668,
        0.673,
        "SITE 1 | PRIMARY AMINE",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=CYAN_DARK,
        va="top",
        zorder=5,
    )
    ax.text(
        0.668,
        0.638,
        r"protonated  $-\mathrm{NH}_3^+$",
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        va="top",
        zorder=5,
    )
    _metric(ax, 0.668, 0.595, "macro pK$_a$", "10.67", color=CYAN_DARK)
    _metric(ax, 0.758, 0.595, "P(protonated)", "99.95%", color=CYAN_DARK)
    _metric(ax, 0.855, 0.595, "confidence", "0.30 | low", color=CYAN_DARK)

    _rounded_box(
        ax,
        0.650,
        0.375,
        0.295,
        0.150,
        facecolor=PINK_PALE,
        edgecolor=PINK,
        radius=0.013,
        zorder=4,
    )
    ax.text(
        0.668,
        0.498,
        "SITE 2 | CARBOXYLIC ACID",
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        color=PINK_DARK,
        va="top",
        zorder=5,
    )
    ax.text(
        0.668,
        0.463,
        r"deprotonated  $-\mathrm{COO}^-$",
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        va="top",
        zorder=5,
    )
    _metric(ax, 0.668, 0.420, "macro pK$_a$", "4.01", color=PINK_DARK)
    _metric(ax, 0.758, 0.420, "P(deprotonated)", "99.96%", color=PINK_DARK)
    _metric(ax, 0.855, 0.420, "confidence", "0.50 | medium", color=PINK_DARK)

    # Output-contract footer.
    _rounded_box(
        ax,
        0.035,
        0.105,
        0.930,
        0.185,
        facecolor=WHITE,
        edgecolor=LINE,
        radius=0.015,
    )
    footer_items = [
        (0.055, "COMPLETE STATE", "99.91%", "dominant configuration", GREEN),
        (0.315, "TAUTOMER RANKING", "#1 | 73.0%*", "conditional heuristic weight", "#7C3AED"),
        (0.590, "AUDITABLE OUTPUTS", "3 linked tables", "molecule | site | tautomer", "#2563EB"),
        (0.810, "PROVENANCE", "atom-mapped", "model + input hashes", "#475569"),
    ]
    for x, eyebrow, value, note, color in footer_items:
        ax.text(
            x,
            0.258,
            eyebrow,
            transform=ax.transAxes,
            fontsize=7.5,
            fontweight="bold",
            color=color,
            va="top",
            zorder=5,
        )
        ax.text(
            x,
            0.220,
            value,
            transform=ax.transAxes,
            fontsize=13.2,
            fontweight="bold",
            color=INK,
            va="top",
            zorder=5,
        )
        ax.text(
            x,
            0.180,
            note,
            transform=ax.transAxes,
            fontsize=8.0,
            color=MUTED,
            va="top",
            zorder=5,
        )

    ax.text(
        0.035,
        0.055,
        "Population and confidence are different: confidence also reflects applicability, calibration, and model identifiability.  "
        "*Tautomer weights rank retained candidates; they are not physical equilibrium probabilities.",
        transform=ax.transAxes,
        fontsize=7.8,
        color=MUTED,
        va="center",
    )
    return figure


def render(output_dir: str, basename: str, formats: Iterable[str], dpi: int) -> list[Path]:
    """Render requested formats and return the written artifact paths."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    written: list[Path] = []
    title = "pKate Stage 3 prediction example"
    description = (
        "GABA example at pH 7.4 showing the input structure, dominant atom-mapped "
        "zwitterionic microstate, site pKa values, marginals, confidence, and provenance."
    )
    for raw_format in formats:
        image_format = str(raw_format).strip().lower()
        if image_format not in {"svg", "png"}:
            raise ValueError(f"Unsupported format: {image_format}")
        path = destination / f"{basename}.{image_format}"
        metadata = (
            {"Title": title, "Description": description, "Date": None}
            if image_format == "svg"
            else {"Title": title, "Description": description, "Software": "pKate"}
        )
        figure.savefig(
            path,
            format=image_format,
            dpi=int(dpi),
            facecolor=BACKGROUND,
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.12,
            metadata=metadata,
        )
        if image_format == "svg":
            # Matplotlib leaves spaces at the end of multiline path data.
            # Normalizing them keeps generated assets friendly to git diff --check.
            svg = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
        written.append(path)
    plt.close(figure)
    return written


def parse_args() -> argparse.Namespace:
    """Parse reproducible output options for the README example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument("--basename", default="pkate_prediction_example")
    parser.add_argument(
        "--formats",
        default="svg,png",
        help="Comma-separated subset of svg,png (default: svg,png).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    """Render the example and print each generated path."""
    args = parse_args()
    formats = [value.strip() for value in args.formats.split(",") if value.strip()]
    if not formats:
        raise ValueError("At least one output format is required")
    for path in render(args.output_dir, args.basename, formats, args.dpi):
        print(path)


if __name__ == "__main__":
    main()
