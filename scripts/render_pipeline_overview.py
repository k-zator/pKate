#!/usr/bin/env python3
"""Render the canonical protonation pipeline as a GitHub-ready figure.

The source of truth remains ``CANONICAL_PIPELINE_GUIDE.md``; this diagram is a
compact orientation aid.  SVG is the preferred GitHub asset because labels
remain sharp when the image is scaled.  PNG is emitted as a convenient fallback.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

# Matplotlib otherwise tries to write below ~/.config, which is undesirable in
# containers and read-only CI jobs.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "pkate-matplotlib"),
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    # Embed glyph outlines so browsers cannot reinterpret Computer Modern's
    # symbol-font code points (for example, turning a dot into a currency sign).
    "svg.fonttype": "path",
    "svg.hashsalt": "pkate-canonical-pipeline-v1",
})

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


BACKGROUND = "#F8FAFC"
INK = "#172033"
MUTED = "#526072"
LINE = "#CBD5E1"


def _wrapped(text: str, width: int) -> str:
    """Wrap a label without splitting chemical or field-name tokens."""
    return textwrap.fill(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _rounded_box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 1.5,
    radius: float = 0.018,
    zorder: int = 2,
) -> FancyBboxPatch:
    """Draw one rounded rectangle in normalized figure coordinates."""
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


def _card(
    ax: Axes,
    x: float,
    *,
    eyebrow: str,
    title: str,
    question: str,
    bullets: Sequence[str],
    output: str,
    facecolor: str,
    accent: str,
) -> None:
    """Draw a stage card with a question, operations, and output contract."""
    y, width, height = 0.215, 0.212, 0.625
    _rounded_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=facecolor,
        edgecolor=accent,
        linewidth=1.7,
    )
    # Accent rail makes stage colors survive grayscale and low-contrast screens.
    _rounded_box(
        ax,
        x + 0.008,
        y + 0.018,
        0.008,
        height - 0.036,
        facecolor=accent,
        edgecolor=accent,
        linewidth=0.0,
        radius=0.004,
        zorder=3,
    )
    left = x + 0.028
    ax.text(
        left,
        y + height - 0.050,
        eyebrow,
        transform=ax.transAxes,
        fontsize=7.8,
        fontweight="bold",
        color=accent,
        va="top",
        zorder=4,
    )
    ax.text(
        left,
        y + height - 0.087,
        _wrapped(title, 25),
        transform=ax.transAxes,
        fontsize=13.0,
        fontweight="bold",
        color=INK,
        va="top",
        linespacing=1.06,
        zorder=4,
    )
    ax.text(
        left,
        y + height - 0.163,
        _wrapped(question, 34),
        transform=ax.transAxes,
        fontsize=8.5,
        fontstyle="italic",
        color=MUTED,
        va="top",
        linespacing=1.18,
        zorder=4,
    )
    ax.plot(
        [left, x + width - 0.020],
        [y + height - 0.235, y + height - 0.235],
        color=LINE,
        linewidth=0.9,
        transform=ax.transAxes,
        zorder=4,
    )
    bullet_y = y + height - 0.267
    for bullet in bullets:
        ax.text(
            left,
            bullet_y,
            r"$\bullet$",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            color=accent,
            va="top",
            zorder=4,
        )
        wrapped = _wrapped(bullet, 34)
        ax.text(
            left + 0.014,
            bullet_y,
            wrapped,
            transform=ax.transAxes,
            fontsize=8.0,
            color=INK,
            va="top",
            linespacing=1.18,
            zorder=4,
        )
        bullet_y -= 0.044 + 0.017 * wrapped.count("\n")

    _rounded_box(
        ax,
        left,
        y + 0.027,
        width - 0.050,
        0.058,
        facecolor="#FFFFFF",
        edgecolor=accent,
        linewidth=1.0,
        radius=0.012,
        zorder=4,
    )
    ax.text(
        x + width / 2 + 0.002,
        y + 0.056,
        _wrapped(output, 34),
        transform=ax.transAxes,
        fontsize=8.1,
        fontweight="bold",
        color=accent,
        ha="center",
        va="center",
        linespacing=1.05,
        zorder=5,
    )


def _arrow(ax: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Connect adjacent cards with a compact directional arrow."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.0,
        color="#64748B",
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    ax.add_patch(arrow)


def _note(
    ax: Axes,
    x: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
) -> None:
    """Draw the supervision or calibration note beneath a stage card."""
    _rounded_box(
        ax,
        x,
        0.100,
        0.212,
        0.092,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0,
        radius=0.012,
        zorder=2,
    )
    ax.text(
        x + 0.106,
        0.146,
        _wrapped(text, 39),
        transform=ax.transAxes,
        fontsize=7.7,
        color=INK,
        ha="center",
        va="center",
        linespacing=1.15,
        zorder=3,
    )


def build_figure() -> Figure:
    """Build the complete four-card pipeline overview figure."""
    figure, ax = plt.subplots(figsize=(16, 9), constrained_layout=False)
    figure.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.025,
        0.952,
        "From experimental pKa evidence to a complete molecular microstate",
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0.025,
        0.906,
        r"Canonical pK$_a$te pipeline $\cdot$ atom-mapped, provenance-checked, and explicit about uncertainty",
        transform=ax.transAxes,
        fontsize=10.5,
        color=MUTED,
        va="top",
    )

    cards = [
        {
            "x": 0.025,
            "eyebrow": "PREREQUISITE | CURATED NETWORK",
            "title": "Evidence & microstate network",
            "question": "Which measurements, sites, and molecular states are defensible?",
            "bullets": [
                "Curate exact-site, ambiguous, and quarantined pKa evidence",
                "Detect atom-mapped ionization transitions",
                "Enumerate protonation configurations, edges, and tautomers",
                "Keep incomplete networks explicit rather than renormalizing them",
            ],
            "output": "Molecule-network dataset",
            "facecolor": "#EEF2F7",
            "accent": "#475569",
        },
        {
            "x": 0.270,
            "eyebrow": "STAGE 1 | LEARNED",
            "title": "Intrinsic transition pKa",
            "question": "What does each mapped transition do in simple systems?",
            "bullets": [
                "Train on exact, single-coordinate experimental transitions",
                "Use molecular, local-site, geometry, and reference-prior features",
                "Validate by scaffold and same-label interpolation",
                "Predict every network site with applicability and intervals",
            ],
            "output": r"Intrinsic site pKas $\rightarrow$ blind macro ladder",
            "facecolor": "#E6F1FF",
            "accent": "#2563EB",
        },
        {
            "x": 0.515,
            "eyebrow": "STAGE 2 | LEARNED + CONSTRAINED",
            "title": "Complex-network correction",
            "question": "How does the multisite ensemble shift the observable macro ladder?",
            "bullets": [
                "Learn residuals from the blind Stage 1 network baseline",
                "Gate sparse families and preserve exact Stage 1 fallbacks",
                "Project successive macroscopic pKas into physical order",
                "Fit regularized one-body and symmetric pair terms",
            ],
            "output": r"Cycle-consistent $b_i / J_{ij}$ energy model",
            "facecolor": "#F2EAFF",
            "accent": "#7C3AED",
        },
        {
            "x": 0.760,
            "eyebrow": "STAGE 3 | PHYSICS + HEURISTIC",
            "title": "Complete state at requested pH",
            "question": "Which full configuration is populated, and how confident is it?",
            "bullets": [
                "Normalize coupled configuration populations at the requested pH",
                "Marginalize each site and propagate pKa uncertainty",
                "Select the thermodynamically dominant protonation configuration",
                "Rank tautomers only inside that fixed configuration",
            ],
            "output": "Mapped microstate + site and tautomer tables",
            "facecolor": "#E4F8EA",
            "accent": "#15803D",
        },
    ]
    for card in cards:
        _card(ax, **card)

    for left, right in zip(cards, cards[1:]):
        _arrow(
            ax,
            (float(left["x"]) + 0.214, 0.525),
            (float(right["x"]) - 0.002, 0.525),
        )

    _note(
        ax,
        0.025,
        "Guardrail: no Marvin or Epik training labels; manual review remains traceable",
        facecolor="#F8FAFC",
        edgecolor="#94A3B8",
    )
    _note(
        ax,
        0.270,
        "Supervision: exact single-coordinate pKas; copied lineages collapsed",
        facecolor="#EFF6FF",
        edgecolor="#93C5FD",
    )
    _note(
        ax,
        0.515,
        "Supervision: exact multisite residuals + safe low-weight weak labels",
        facecolor="#F5F3FF",
        edgecolor="#C4B5FD",
    )
    _note(
        ax,
        0.760,
        "Confidence: held-out residual calibration; tautomer scores stay heuristic",
        facecolor="#F0FDF4",
        edgecolor="#86EFAC",
    )

    ax.text(
        0.025,
        0.045,
        "Key distinction:",
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=INK,
        va="center",
    )
    ax.text(
        0.112,
        0.045,
        "compare scalar experiments with the macroscopic pKa; one-body and background-specific edge pKas are model parameters.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=MUTED,
        va="center",
    )
    ax.text(
        0.975,
        0.045,
        r"pK$_a$te $\cdot$ canonical molecule-network pipeline",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#64748B",
        ha="right",
        va="center",
    )
    return figure


def render(output_dir: str, basename: str, formats: Iterable[str], dpi: int) -> list[Path]:
    """Render requested formats and return the written artifact paths."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    written: list[Path] = []
    title = "Canonical pKate pipeline overview"
    description = (
        "Four-step diagram from curated pKa evidence and atom-mapped microstate "
        "networks through intrinsic pKa, multisite correction, and complete-state inference."
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
    """Parse reproducible output options for the documentation figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument("--basename", default="canonical_pipeline_overview")
    parser.add_argument(
        "--formats",
        default="svg,png",
        help="Comma-separated subset of svg,png (default: svg,png).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    """Render the diagram and print each generated path."""
    args = parse_args()
    formats = [value.strip() for value in args.formats.split(",") if value.strip()]
    if not formats:
        raise ValueError("At least one output format is required")
    for path in render(args.output_dir, args.basename, formats, args.dpi):
        print(path)


if __name__ == "__main__":
    main()
