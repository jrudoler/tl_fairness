"""Centralized matplotlib defaults for publication-ready figures.

Call ``configure_matplotlib()`` at the top of every figure entrypoint so that
figures are styled identically and are ready to drop into the manuscript
without manual editing.
"""

from pathlib import Path

from matplotlib import pyplot as plt


def configure_matplotlib(style_path: str | Path | None = None) -> None:
    """Apply shared, publication-ready matplotlib defaults.

    - Embed text as text in SVG/PDF (editable, TrueType fonts) rather than paths.
    - Keep the clean-figs colors, faint grid, and hidden top/right spines.
    - Override its oversized monospace typography for manuscript figures.
    """
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    if style_path is None:
        personal_style = Path.home() / "clean-figs.mplstyle"
        style_path = (personal_style if personal_style.is_file()
                      else Path(__file__).resolve().parents[1] / "clean-figs.mplstyle")
    plt.style.use(Path(style_path).expanduser())
    # The clean-figs appearance is useful, but its 14/16 pt monospace text
    # crowds these manuscript layouts. Keep typography independent of it.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.1,
    })


# Common manuscript widths (inches). Size figures to their final rendered width
# so text stays readable; do not author wide and rely on later shrinking.
FULL_WIDTH = 6.5
COLUMN_WIDTH = 3.25
