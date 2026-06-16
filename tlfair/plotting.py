"""Centralized matplotlib defaults for publication-ready figures.

Call ``configure_matplotlib()`` at the top of every figure entrypoint so that
figures are styled identically and are ready to drop into the manuscript
without manual editing.
"""

from matplotlib import pyplot as plt


def configure_matplotlib() -> None:
    """Apply shared, publication-ready matplotlib defaults.

    - Embed text as text in SVG/PDF (editable, TrueType fonts) rather than paths.
    - Use a consistent base font size sized for manuscript inclusion.
    """
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 10


# Common manuscript widths (inches). Size figures to their final rendered width
# so text stays readable; do not author wide and rely on later shrinking.
FULL_WIDTH = 6.5
COLUMN_WIDTH = 3.25
