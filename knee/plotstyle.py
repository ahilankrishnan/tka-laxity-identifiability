"""Figure style: fixed inch grid of square panels, Times text with Computer Modern math (as in the manuscript),
boxed axes, inward ticks, and the manuscript-wide color system below."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, ListedColormap, BoundaryNorm

WIDTH = 6.5
LEFT, GAP, RIGHT, BELOW, ABOVE = 0.52, 0.04, 0.10, 0.40, 0.34   # inches; tight grid, labels hug their axes
LW_DATA = 1.6   # data line width

# Manuscript-wide color system:
#   hue = ligament identity (MCL, LCL, PCL, capsule; medial/lateral gaps take the MCL/LCL hues),
#   shade (light to dark) = calibration condition, the same three levels in every figure (COND_SHADE),
#   neutral grays = quantitative indices and comparisons of calibration conditions (COND_GRAY),
#   loading-mode accents = AP, VV and IE tests.
HUE = {"MCL": "#2474A6", "LCL": "#D64B59", "PCL": "#16876C", "CAP": "#8064A8",
       "AP": "#8C510A", "VV": "#D97706", "IE": "#C2185B", "gapM": "#2474A6", "gapL": "#D64B59",
       "gray": "#5A5A5A", "slate": "#3F5068"}
COND_SHADE = (0.3, 0.65, 1.0)                    # least, intermediate and most calibration information
COND_GRAY = ("#B8B8B8", "#6E6E6E", "#222222")    # the same three levels in neutral gray (legends, Fig. 3B)


def mix(a: str, b: str, w: float) -> tuple:
    """Linear RGB blend (1 - w) a + w b."""
    a, b = np.array(to_rgb(a)), np.array(to_rgb(b))
    return tuple((1 - w) * a + w * b)


def shade(base: str, s: float) -> tuple:
    """Shade s in [0, 1] (light to dark) of a base hue."""
    light = mix(base, "white", 0.88)
    dark = mix(base, "black", 0.62)
    return mix(light, dark, s)


def setup() -> None:
    """Global matplotlib style for all manuscript figures."""
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": (r"\usepackage[T1]{fontenc}\usepackage{amsmath}"
                                r"\renewcommand{\rmdefault}{ptm}\renewcommand{\sfdefault}{phv}"),
        "font.family": "serif", "font.serif": ["Computer Modern Roman"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 13.9, "axes.titlepad": 4, "axes.labelpad": 2,
        "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.linewidth": 1.44, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 1.44, "ytick.major.width": 1.44, "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "xtick.major.pad": 3, "ytick.major.pad": 3, "xtick.minor.visible": False, "ytick.minor.visible": False,
        "axes.grid": False, "lines.linewidth": LW_DATA,
        "legend.frameon": False, "legend.handlelength": 1.1, "legend.handletextpad": 0.3,
        "legend.columnspacing": 0.6, "legend.labelspacing": 0.15, "legend.borderpad": 0.1,
        "legend.borderaxespad": 0.3, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def grid(nrows: int, ncols: int, size: float | None = None, extra_below: float = 0.0, extra_left: float = 0.0,
         extra_right: float = 0.0, colsep: float | None = None,
         rowsep: float | None = None) -> tuple[plt.Figure, np.ndarray, float]:
    """Square panels on one grid, packed tightly so that panels of a figure read as one group.

    colsep: horizontal distance (in) between neighboring panels, default LEFT + GAP (room for y labels);
    rowsep: vertical distance (in) between neighboring rows, default BELOW + ABOVE + extra_below.
    Returns fig, axes (nrows x ncols), panel size (in)."""
    colsep = LEFT + GAP if colsep is None else colsep
    rowsep = BELOW + ABOVE + extra_below if rowsep is None else rowsep
    if size is None:
        size = (WIDTH - extra_left - extra_right - LEFT - RIGHT - (ncols - 1) * colsep) / ncols
    width = extra_left + LEFT + ncols * size + (ncols - 1) * colsep + RIGHT + extra_right
    height = ABOVE + nrows * size + (nrows - 1) * rowsep + BELOW + extra_below
    fig = plt.figure(figsize=(width, height))
    axs = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            x0 = extra_left + LEFT + c * (size + colsep)
            y0 = height - ABOVE - size - r * (size + rowsep)
            axs[r, c] = fig.add_axes([x0 / width, y0 / height, size / width, size / height])
    return fig, axs, size


def letter(ax: plt.Axes, s: str) -> None:
    """Bold uppercase panel letter at the top-left corner of the axes."""
    s = r"\textbf{%s}" % s.upper()
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(-30, 4), textcoords="offset points",
                fontsize=15, va="bottom", ha="left")


def sequential_levels(n: int = 10, vmin: float = 0.0, vmax: float = 1.0) -> tuple[ListedColormap, BoundaryNorm, np.ndarray]:
    """White-to-slate color map with n discrete levels for heat maps; slate is a muted blue-gray
    that differs from every ligament and laxity-test hue."""
    colors = [mix("#FFFFFF", HUE["slate"], f) for f in np.linspace(0.0, 1.0, n)]
    cmap = ListedColormap(colors)
    bounds = np.linspace(vmin, vmax, n + 1)
    return cmap, BoundaryNorm(bounds, n), bounds
