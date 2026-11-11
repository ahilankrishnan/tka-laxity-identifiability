"""All manuscript figures (manuscript/figures), drawn at print size on the fixed panel grid.

Hue = quantity (ligament group or laxity test), shade = case (measurement subset or estimator).
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee.plotstyle import (setup, letter, grid, shade, HUE, LW_DATA, LW_PROFILE, WIDTH,  # noqa: E402
                            sequential_levels)
import matplotlib.pyplot as plt                                      # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Patch        # noqa: E402
from matplotlib.lines import Line2D                                  # noqa: E402
from knee.model import FEM, INSERT, BUNDLES_REF as BUNDLES, GROUPS, T_REF                  # noqa: E402
from knee.sampling import PRIOR, PARAM_NAMES                         # noqa: E402
from knee.subsets import SUBSETS                                     # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "manuscript" / "figures"
GLAB = {"MCL": "MCL", "LCL": "LCL", "PCL": "PCL", "CAP": "capsule"}
PLAB = {"k_MCL": r"$k_\mathrm{MCL}$", "eps_MCL": r"$\varepsilon_{\mathrm{r},\mathrm{MCL}}$",
        "k_LCL": r"$k_\mathrm{LCL}$", "eps_LCL": r"$\varepsilon_{\mathrm{r},\mathrm{LCL}}$",
        "k_PCL": r"$k_\mathrm{PCL}$", "eps_PCL": r"$\varepsilon_{\mathrm{r},\mathrm{PCL}}$",
        "k_CAP": r"$k_\mathrm{CAP}$", "eps_CAP": r"$\varepsilon_{\mathrm{r},\mathrm{CAP}}$"}
SUB_SHADE = {"Gaps": 0.3, "Gaps+VV": 0.62, "All": 1.0}
SUBLAB = {"All": "Full exam", "All 0": r"Full exam 0$^\circ$", "All 0/90": r"Full exam 0$^\circ$/90$^\circ$",
          "Gaps 0/90": r"Gaps 0$^\circ$/90$^\circ$"}   # display names of subsets   # case shades of the measurement subsets
DOT = (0, (1, 0.9))


def phys(v, name: str):
    """Normalized parameter -> stiffness in kN or reference strain in percent."""
    lo, hi = PRIOR[name]
    if name.startswith("k_"):
        return 10 ** (np.log10(lo) + v * (np.log10(hi) - np.log10(lo))) / 1000
    return (lo + v * (hi - lo)) * 100


def save(fig, name: str) -> None:
    """Write the figure as vector PDF."""
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


def heat(ax, M, rows, cols, cmap, norm):
    """Annotated heat map of a subset-by-parameter matrix."""
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="equal")
    ax.set_anchor("NW")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] >= 0.6 else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9)
    ax.tick_params(top=False, right=False, length=0)
    ax.set_xticks(np.arange(-0.5, M.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, M.shape[0]), minor=True)
    ax.grid(which="minor", color="k", lw=0.4)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    return im


def colorbar(fig, ax, im, bounds) -> None:
    """Abaqus-style contour legend: stacked level boxes with signed values at the level boundaries."""
    ax.apply_aspect()
    p = ax.get_position()
    W = fig.get_figwidth()
    cax = fig.add_axes([p.x1 + 0.12 / W, p.y0, 0.16 / W, p.height])
    cmap, n = im.get_cmap(), len(bounds) - 1
    for k in range(n):
        cax.add_patch(plt.Rectangle((0, k), 1, 1, fc=cmap(k / (n - 1)), ec="black", lw=0.6))
    cax.set_xlim(0, 1); cax.set_ylim(0, n)
    cax.set_xticks([]); cax.set_yticks(range(n + 1))
    step = 1 if p.height * fig.get_figheight() > 1.6 else 2   # thin out labels on a short legend
    cax.set_yticklabels([f"{b:.1f}" if k % step == 0 else "" for k, b in enumerate(bounds)], fontsize=9)
    cax.yaxis.tick_right(); cax.tick_params(axis="y", length=3, width=0.6, direction="out", pad=2, left=False, right=True)
    for sp in cax.spines.values():
        sp.set_visible(False)


# ---------------------------------------------------------------- figure 1
_BL = {"aMCL": "aMCL", "iMCL": "iMCL", "pMCL": "pMCL", "LCL": "LCL", "aPCL": "alPCL", "pPCL": "pmPCL",
       "mCAP": "pmCAP", "lCAP": "plCAP", "alCAP": "alCAP"}
# (title, horizontal axis, sign, depth axis, depth sign, end labels, labelled bundles)
# right knee, y medial: camera right vector = forward x up
MS_ATT = 3.0
_VIEWS = {"anterior": ("anterior view", 1, +1, 0, +1, ("lateral", "medial"), ("iMCL", "LCL", "alCAP")),
          "medial": ("medial view", 0, -1, 1, +1, ("anterior", "posterior"), ("aMCL", "pMCL", "mCAP")),
          "posterior": ("posterior view", 1, -1, 0, -1, ("medial", "lateral"), ("pPCL", "aPCL", "lCAP")),
          "lateral": ("lateral view", 0, +1, 1, -1, ("posterior", "anterior"), ("LCL", "alCAP", "lCAP"))}


def knee_view(ax, parts, view: str, fs: float = 9.0, origins: bool = False) -> None:
    """Orthographic, Lambert-shaded view of the knee mesh with ligament lines, attachments, and labelled bundles."""
    from matplotlib.collections import PolyCollection
    from knee.geometry3d import COLORS
    title, hax, hs, dax, ds, (left, right), labelled = _VIEWS[view]
    polys, cols, depth = [], [], []
    cam = np.zeros(3); cam[dax] = ds
    light = cam + np.array([0, 0, 0.6]); light /= np.linalg.norm(light)
    for n in ("insert", "femoral_component"):
        V, F = parts[n]
        T = V[F]
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        lam = 0.45 + 0.55 * np.abs(nrm @ light)
        base = np.array(COLORS[n])
        polys += list(np.stack([T[..., hax] * hs, T[..., 2]], -1))
        cols += list(np.clip(base[None] * lam[:, None], 0, 1))
        depth += list(T[..., dax].mean(1) * ds)
    o = np.argsort(depth)
    ax.add_collection(PolyCollection([polys[i] for i in o], facecolors=[cols[i] for i in o], edgecolors="none",
                                     antialiased=False, zorder=1))
    for n, (g, pf, pt) in BUNDLES.items():
        h = np.array([pf[hax], pt[hax]]) * hs
        v = np.array([pf[2], pt[2]])
        ax.plot(h, v, color=HUE[g], lw=1.1, zorder=3, solid_capstyle="round")
        ax.plot(h[0], v[0], "o", ms=MS_ATT, mfc=HUE[g], mec="k", mew=0.4, zorder=4)
        ax.plot(h[1], v[1], "o", ms=MS_ATT, mfc="white", mec=HUE[g], mew=0.8, zorder=4)
    if origins:
        for (z0, lab, tx, ty) in ((0.0, r"$O_\mathrm{F}$", -28, -66), (T_REF[2], r"$O_\mathrm{T}$", -4, -66)):
            ax.plot(0, z0, marker="o", ms=3.8, mfc="white", mec="k", mew=0.7, zorder=6)
            ax.plot(0, z0, marker="+", ms=3.8, mec="k", mew=0.7, zorder=7)
            ax.annotate(lab, xy=(0, z0), xytext=(tx, ty), fontsize=fs, ha="center", va="top", zorder=7,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="#606060", shrinkA=0, shrinkB=2.5))
    ax.set_xlim(-98, 98); ax.set_ylim(-100, 22); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=12, pad=2)
    # compact orientation indicator
    ax.annotate("", xy=(-9, -95), xytext=(9, -95), arrowprops=dict(arrowstyle="<|-|>,head_length=2.5,head_width=1.2",
                                                                  lw=0.5, color="#606060", mutation_scale=1))
    ax.text(-12, -95, left, fontsize=8, ha="right", va="center", color="#606060")
    ax.text(12, -95, right, fontsize=8, ha="left", va="center", color="#606060")
    # axis triad at the lower left: in-plane axes as arrows, the viewing axis as a dot (toward) or cross (away)
    bx, by, L = -88.0, -97.0, 12.0
    tri = dict(arrowstyle="-|>,head_length=2.2,head_width=1.0", lw=0.6, color="#303030", mutation_scale=1,
               shrinkA=0, shrinkB=0)
    names = "xyz"
    # horizontal axis of the view points right for hs = +1 and left for hs = -1, so the triad starts on its tail side
    x0t = bx if hs > 0 else bx + L
    ax.annotate("", xy=(x0t + hs * L, by), xytext=(x0t, by), arrowprops=tri)
    ax.text(x0t + hs * (L + 4.5), by, f"${names[hax]}$", fontsize=8, ha="center", va="center")
    ax.annotate("", xy=(x0t, by + L), xytext=(x0t, by), arrowprops=tri)
    ax.text(x0t, by + L + 4, "$z$", fontsize=8, ha="center", va="center")
    ax.plot(x0t, by, marker="o", ms=4.2, mfc="white", mec="#303030", mew=0.6, zorder=6)
    ax.plot(x0t, by, marker="." if ds > 0 else "x", ms=2.6 if ds > 0 else 2.6, mec="#303030", mfc="#303030",
            mew=0.6, zorder=7)
    ax.text(x0t - hs * 6.0, by, f"${names[dax]}$", fontsize=8, ha="center", va="center")
    items = sorted(((BUNDLES[n][2][hax] * hs, BUNDLES[n][2][2], n) for n in labelled), key=lambda q: q[1])
    used = {"L": [], "R": []}
    for hx, vz, n in items:
        side = "L" if hx < 0 else "R"
        ty = max(vz, -70.0) if side == "L" else vz      # keep left labels clear of the axis triad
        while any(abs(ty - u) < 12 for u in used[side]):
            ty += 12
        used[side].append(ty)
        tx = -97 if side == "L" else 97
        ax.annotate(_BL[n], xy=(hx, vz), xytext=(tx, ty), fontsize=fs, color=HUE[BUNDLES[n][0]],
                    ha="left" if side == "L" else "right", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#808080", shrinkA=1.5, shrinkB=2))


_ISO = {  # camera direction (toward the viewer), title, labelled bundles
    "anterolateral": (np.array([1.0, -1.0, 1.0]), "anterolateral view", ("LCL", "alCAP", "aMCL")),
    "posteromedial": (np.array([-1.0, 1.0, 1.0]), "posteromedial view", ("pMCL", "mCAP", "pPCL")),
}


def knee_iso(ax, parts, view: str, fs: float = 7.5) -> None:
    """Isometric, Lambert-shaded view with depth-sorted implant and ligament meshes, origins, and an axis triad."""
    from matplotlib.collections import PolyCollection
    from knee.geometry3d import COLORS
    c, title, labelled = _ISO[view]
    c = c / np.linalg.norm(c)
    r = np.cross(-c, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, -c)
    P = lambda X: np.stack([np.asarray(X) @ r, np.asarray(X) @ u], -1)
    light = c + np.array([0, 0, 0.5]); light /= np.linalg.norm(light)
    polys, cols, depth = [], [], []
    for n in ("insert", "femoral_component", "MCL", "LCL", "PCL", "CAP"):
        V, F = parts[n]
        T = V[F]
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        lam = 0.45 + 0.55 * np.abs(nrm @ light)
        base = np.array(COLORS[n]) if n in ("insert", "femoral_component") else np.array(mcolors.to_rgb(HUE[n]))
        polys += list(P(T)); cols += list(np.clip(base[None] * lam[:, None], 0, 1)); depth += list(T.mean(1) @ c)
    o = np.argsort(depth)
    ax.add_collection(PolyCollection([polys[i] for i in o], facecolors=[cols[i] for i in o], edgecolors="none",
                                     antialiased=False, zorder=1))
    for n, (g, pf, pt) in BUNDLES.items():
        if n in labelled:
            a, b = P(pf), P(pt)
            ax.plot(*a, "o", ms=3.4, mfc=HUE[g], mec="k", mew=0.4, zorder=4)
            ax.plot(*b, "o", ms=3.4, mfc="white", mec=HUE[g], mew=1.0, zorder=4)
    for z0, lab in ((0.0, r"$O_\mathrm{F}$"), (T_REF[2], r"$O_\mathrm{T}$")):
        p = P([0, 0, z0])
        ax.plot(*p, marker="o", ms=3.8, mfc="white", mec="k", mew=0.9, zorder=6)
        ax.plot(*p, marker="+", ms=3.8, mec="k", mew=0.9, zorder=7)
        ax.text(p[0] + 4, p[1] + 2, lab, fontsize=fs, ha="left", va="bottom", zorder=7,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
    # axis triad at the lower left
    o0 = np.array([0.0, 0.0, 0.0]); base = np.array([-28.0, -80.0]) if view == "anterolateral" else np.array([38.0, -80.0])
    for e, lab in ((np.array([1.0, 0, 0]), "$x$"), (np.array([0, 1.0, 0]), "$y$"), (np.array([0, 0, 1.0]), "$z$")):
        d = P(o0 + 14 * e) - P(o0)
        ax.annotate("", xy=base + d, xytext=base, arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#303030",
                                                                  mutation_scale=6, shrinkA=0, shrinkB=0))
        ax.text(*(base + 1.35 * d), lab, fontsize=fs, ha="center", va="center")
    items = []
    for n in labelled:
        q = P(BUNDLES[n][2]); items.append((q[0], q[1], n))
    used = {"L": [], "R": []}
    for hx, vz, n in sorted(items, key=lambda q: q[1]):
        side = "L" if hx < 0 else "R"
        ty = vz
        while any(abs(ty - w) < 11 for w in used[side]):
            ty += 11
        used[side].append(ty)
        tx = -81 if side == "L" else 81
        ax.annotate(_BL[n], xy=(hx, vz), xytext=(tx, ty), fontsize=fs, color=HUE[BUNDLES[n][0]],
                    ha="left" if side == "L" else "right", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#404040", shrinkA=1, shrinkB=2))
    ax.set_xlim(-82, 82); ax.set_ylim(-90, 38); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11, pad=1)


def attach_legend(fig, rect, ncol: int, fs: float = 8.5) -> None:
    lg = fig.add_axes(rect); lg.axis("off")
    lg.legend([Line2D([], [], ls="none", marker="o", ms=4, mfc="#5A5A5A", mec="k", mew=0.4),
               Line2D([], [], ls="none", marker="o", ms=4, mfc="white", mec="#5A5A5A", mew=0.9)] +
              [Line2D([], [], color=HUE[g], lw=1.6) for g in GROUPS],
              ["femoral attachment", "tibial attachment"] + [GLAB[g] for g in GROUPS],
              loc="center", ncol=ncol, fontsize=fs, handlelength=1.2, columnspacing=1.0)


def fig1() -> None:
    """Overview. (A) Ligament model in four views with ligament legend and loading conditions.
    (B) Workflow of the current study and the intended surgical use."""
    from knee.geometry3d import build
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch as FBP
    parts, _ = build()
    FT, FL, FB = 12, 9, 8.5          # titles, labels, box text (as in Figures 2 and 3)
    GREY, EDGE = "#606060", "#4A4A4A"
    vw = 2.24; vh = vw * 122 / 168; gap = 0.36; hb = 1.75
    y0 = hb + 0.42
    H = y0 + 2 * vh + gap + 0.44
    W = WIDTH
    fig = plt.figure(figsize=(W, H))

    def head(x, y, letter_, title):
        fig.text(x / W, y / H, r"\textbf{%s}" % letter_, fontsize=15, ha="left", va="baseline")
        fig.text((x + 0.24) / W, y / H, r"\underline{%s}" % title, fontsize=FT, ha="left", va="baseline")

    head(0.02, H - 0.22, "A", "ligament model")
    for k, v in enumerate(("anterior", "medial", "posterior", "lateral")):
        r_, c_ = divmod(k, 2)
        ax = fig.add_axes([(0.02 + c_ * (vw + 0.04)) / W, (y0 + (1 - r_) * (vh + gap)) / H, vw / W, vh / H])
        knee_view(ax, parts, v, fs=FL, origins=(v == "anterior"))
    # right column: ligament legend and loading conditions on one icon column and one text column
    xr = 2 * vw + 0.16
    hk = 2 * vh + gap
    axk = fig.add_axes([xr / W, y0 / H, (W - xr - 0.02) / W, hk / H])
    axk.set_xlim(0, 100); axk.set_ylim(0, 100); axk.axis("off")
    XI, XT = 17, 31
    axk.text(0, 100, "ligaments", fontsize=FT, ha="left", va="top")
    axk.plot([0, 100], [94.3, 94.3], color="#B0B0B0", lw=0.5)
    ly = [90.3, 86.0, 81.7, 77.4, 73.1, 68.8]
    axk.plot(XI, ly[0], "o", ms=MS_ATT, mfc="#5A5A5A", mec="k", mew=0.4)
    axk.plot(XI, ly[1], "o", ms=MS_ATT, mfc="white", mec="#5A5A5A", mew=0.8)
    for yy, g in zip(ly[2:], GROUPS):
        axk.plot([XI - 7, XI + 7], [yy, yy], color=HUE[g], lw=1.1)
    for yy, s_ in zip(ly, ["femoral attachment", "tibial attachment"] + [GLAB[g] for g in GROUPS]):
        axk.text(XT, yy, s_, ha="left", va="center", fontsize=FL)
    axk.text(0, 61, "loading conditions", fontsize=FT, ha="left", va="top")
    axk.plot([0, 100], [55.3, 55.3], color="#B0B0B0", lw=0.5)
    c = "#303030"
    hd = "head_length=2.6,head_width=1.3"
    two = dict(arrowstyle=f"<|-|>,{hd}", lw=0.8, color=c, mutation_scale=1, shrinkA=0, shrinkB=0)
    one = dict(arrowstyle=f"-|>,{hd}", lw=0.8, color=c, mutation_scale=1, shrinkA=0, shrinkB=0)
    rows = (47, 34, 21, 8)
    yc = rows[0]
    axk.plot([XI - 8, XI + 8], [yc, yc], color="#A0A0A0", lw=1.6, zorder=0)
    for xc in (XI - 4, XI + 4):
        axk.add_patch(FancyArrowPatch((xc, yc - 4), (xc, yc + 4), **two))
    yc = rows[1]; axk.add_patch(FancyArrowPatch((XI - 8, yc), (XI + 8, yc), **two))
    yc = rows[2]; axk.add_patch(FancyArrowPatch((XI - 8, yc - 1.8), (XI + 8, yc - 1.8), connectionstyle="arc3,rad=-0.55", **two))
    yc = rows[3]
    axk.plot([XI, XI], [yc - 5, yc + 5], color="#A0A0A0", lw=0.7, ls=(0, (2, 1.5)))
    t = np.radians(np.linspace(195, 345, 60)); ex, ey = XI + 8 * np.cos(t), yc + 2.8 * np.sin(t)
    axk.plot(ex[2:-2], ey[2:-2], color=c, lw=0.8)
    axk.add_patch(FancyArrowPatch((ex[3], ey[3]), (ex[0], ey[0]), **one))
    axk.add_patch(FancyArrowPatch((ex[-4], ey[-4]), (ex[-1], ey[-1]), **one))
    labels = ("joint distraction,\n100 N per compartment", "AP translation,\n$\\pm$100 N",
              "varus--valgus rotation,\n$\\pm$10 N m", "internal--external\nrotation, $\\pm$3 N m")
    for k_, (yc, s_) in enumerate(zip(rows, labels)):
        axk.text(3, yc, str(k_ + 1), ha="center", va="center", fontsize=7.5,
                 bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="k", lw=0.5))
        axk.text(XT, yc, s_, ha="left", va="center", fontsize=FL, linespacing=1.1)
    # B: workflow, left to right
    head(0.02, hb + 0.14, "B", "study workflow and intended surgical use")
    axb = fig.add_axes([0.02 / W, 0.03 / H, (W - 0.04) / W, (hb - 0.12) / H])
    axb.set_xlim(0, 100); axb.set_ylim(0, 100); axb.axis("off")
    arr = dict(arrowstyle="-|>,head_length=2.6,head_width=1.3", lw=0.7, color=EDGE, shrinkA=0, shrinkB=0, mutation_scale=1)
    BW, BH = 12.0, 36.0

    def box(x0, yc_, s_, fc="#F2F3F5"):
        axb.add_patch(FBP((x0, yc_ - BH / 2), BW, BH, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=EDGE, lw=0.6))
        axb.text(x0 + BW / 2, yc_, s_, ha="center", va="center", fontsize=FB, linespacing=1.05)

    def arrow(x0, x1, y):
        axb.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=arr)

    YM, YU, YL = 47, 76, 19
    axb.text(0, 99, r"\textsc{current study}", fontsize=FL, ha="left", va="top", color=GREY)
    box(0, YM, "synthetic\nligament\nparameters")
    arrow(BW, 15.0, YM)
    box(15.0, YM, "virtual\nlaxity\nexamination")
    axb.text(15.0 + BW / 2, YM - BH / 2 - 4, "with measurement\nnoise", ha="center", va="top", fontsize=8, linespacing=1.05,
             style="italic", color=GREY)
    xt = 15.0 + BW + 1.6
    axb.plot([15.0 + BW, xt], [YM, YM], color=EDGE, lw=0.7)
    axb.plot([xt, xt], [YL, YU], color=EDGE, lw=0.7)
    arrow(xt - 0.05, 30.2, YU); arrow(xt - 0.05, 30.2, YL)
    box(30.2, YU, "local Fisher\nanalysis")
    box(30.2, YL, "least-squares\ncalibration")
    arrow(30.2 + BW, 45.2, YL)
    box(45.2, YL, "held-out\npassive laxity")
    # intended surgical use: dashed section
    x0c = 59.0
    axb.add_patch(FBP((x0c, 1), 100 - x0c - 0.3, 98, boxstyle="round,pad=0,rounding_size=1.5", fill=False, ec=GREY,
                      lw=0.6, ls=(0, (3, 2))))
    axb.text(x0c + 1.2, 95, r"\textsc{intended surgical use}", fontsize=FL, ha="left", va="top", color=GREY)
    xs = [x0c + 1.2 + k * (BW + 1.4) for k in range(3)]
    for k, s_ in enumerate(("instrumented\ntrial\nexamination", "model\nupdate", "predict effects\nof an\nadjustment")):
        box(xs[k], YM, s_, fc="white")
        if k:
            arrow(xs[k - 1] + BW, xs[k], YM)
    axb.text(xs[2] + BW, YM - BH / 2 - 4, "e.g., insert thickness", ha="right", va="top", fontsize=8,
             style="italic", color=GREY)
    save(fig, "fig1_overview")


# ---------------------------------------------------------------- figure 2
def figS2() -> None:
    """Forward behavior: neutral pose, ligament forces, gaps, and laxity bands."""
    fw = np.load(RES / "forward" / "forward.npz")
    tr = np.load(RES / "data" / "train.npz")
    fig, axs, _ = grid(2, 3, rowsep=0.5)
    flex, Q, F = fw["flex"], fw["Q"], fw["F"]
    ax = axs[0, 0]
    ax.plot(flex, Q[:, 0], color=HUE["AP"], label=r"$u_\mathrm{AP}$")
    ax.plot(flex, np.degrees(Q[:, 4]), color=HUE["IE"], label=r"$\gamma$")
    ax.plot(flex, np.degrees(Q[:, 3]), color=HUE["VV"], label=r"$\beta$")
    ax.set_xlabel(r"flexion $\alpha$ [$^\circ$]"); ax.set_ylabel(r"pose [mm, $^\circ$]")
    ax.set_xlim(0, 120); ax.set_xticks([0, 60, 120]); ax.set_ylim(0, 20); ax.set_yticks([0, 5, 10, 15, 20])
    ax.legend(loc="upper left"); ax.set_title("neutral pose", fontsize=12); letter(ax, "a")
    ax = axs[0, 1]
    names = list(fw["names"])
    for g in GROUPS:
        sel = [i for i, n in enumerate(names) if BUNDLES[n][0] == g]
        ax.plot(flex, F[:, sel].sum(1), color=HUE[g], label=GLAB[g])
    ax.set_xlabel(r"flexion $\alpha$ [$^\circ$]"); ax.set_ylabel(r"force $F$ [N]")
    ax.set_xlim(0, 120); ax.set_xticks([0, 60, 120]); ax.set_ylim(0, 150); ax.set_yticks([0, 50, 100, 150])
    ax.legend(loc="upper right"); ax.set_title("neutral ligament force", fontsize=12); letter(ax, "b")
    feats = list(tr["features"]); Y = tr["y"][tr["ok"]]
    fl = np.array([0, 30, 60, 90])

    def band(ax, key, hue, label=None):
        idx = [feats.index(f"{key}@{f}") for f in fl]
        lo, med, hi = np.percentile(Y[:, idx], [5, 50, 95], axis=0)
        ax.fill_between(fl, lo, hi, color=shade(hue, 0.15), lw=0, alpha=0.8)
        ax.plot(fl, med, color=hue, label=label)

    ax = axs[0, 2]
    band(ax, "gapM", HUE["gapM"], "medial"); band(ax, "gapL", HUE["gapL"], "lateral")
    ax.set_xlabel(r"flexion $\alpha$ [$^\circ$]"); ax.set_ylabel(r"gap $g$ [mm]")
    ax.set_xlim(0, 90); ax.set_xticks([0, 30, 60, 90]); ax.set_ylim(-0.5, 8); ax.set_yticks([0, 2, 4, 6, 8])   # small negative gaps are foundation overlap
    ax.legend(loc="upper left"); ax.set_title("gap, 100 N distraction", fontsize=12); letter(ax, "c")
    specs = [(("ant", "post"), "AP", r"translation $\Delta u_\mathrm{AP}$ [mm]", (-16, 16), [-16, -8, 0, 8, 16], "AP laxity, 100 N", "d"),
             (("var", "val"), "VV", r"rotation $\Delta\beta$ [$^\circ$]", (-16, 16), [-16, -8, 0, 8, 16], "VV laxity, 10 N m", "e"),
             (("int", "ext"), "IE", r"rotation $\Delta\gamma$ [$^\circ$]", (-40, 40), [-40, -20, 0, 20, 40], "IE laxity, 3 N m", "f")]
    for ax, (pair, key, lab, lim, ticks, title, s) in zip(axs[1], specs):
        for t in pair:
            band(ax, t, HUE[key])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(r"flexion $\alpha$ [$^\circ$]"); ax.set_ylabel(lab)
        ax.set_xlim(0, 90); ax.set_xticks([0, 30, 60, 90]); ax.set_ylim(*lim); ax.set_yticks(ticks)
        ax.set_title(title, fontsize=12); letter(ax, s)
    for ax in axs.ravel():
        ax.set_xlim(0, 90); ax.set_xticks([0, 30, 60, 90])
    for ax in axs[0]:
        ax.set_xticklabels([]); ax.set_xlabel("")
    save(fig, "figS2_forward")


# ---------------------------------------------------------------- figure 3
def fig2() -> None:
    """Fisher identifiability index by measurement subset and stiffness-strain correlation (equal heights)."""
    idf = json.loads((RES / "identifiability" / "summary.json").read_text())
    rows = list(SUBSETS)
    cols = [PLAB[p] for p in PARAM_NAMES]
    A = np.array([idf["subsets"][r]["ident_median"] for r in rows])
    h, xa, xb, yb = 1.98, 1.25, 3.80, 0.62
    W, H = WIDTH, yb + h + 0.34
    fig = plt.figure(figsize=(W, H))
    ax0 = fig.add_axes([xa / W, yb / H, h * A.shape[1] / A.shape[0] / W, h / H])
    ax = fig.add_axes([xb / W, yb / H, h / W, h / H])
    cmap, norm, bounds = sequential_levels()
    im = heat(ax0, A, [SUBLAB.get(r_, r_) for r_ in rows], cols, cmap, norm)
    ax0.set_title("Fisher identifiability index", fontsize=12)
    colorbar(fig, ax0, im, bounds)
    letter(ax0, "a"); ax0.texts[-1].set_position((-86, 4))
    x = np.arange(4)
    for o, (name, s) in zip((-0.27, 0, 0.27), SUB_SHADE.items()):
        C = np.array(idf["subsets"][name]["corr_median"])
        ax.bar(x + o, [C[2 * i, 2 * i + 1] for i in range(4)], width=0.27,
               color=[shade(HUE[g], s) for g in GROUPS], edgecolor="k", lw=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([GLAB[g] for g in GROUPS], rotation=45, ha="right", rotation_mode="anchor"); ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel(r"correlation $\rho(\log k, \varepsilon_\mathrm{r})$ [-]")
    ax.set_ylim(-1, 0.5); ax.set_yticks([-1, -0.5, 0, 0.5]); ax.set_yticklabels(["$-1$", "$-0.5$", "$0$", "$0.5$"])
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", labelleft=False, labelright=True)
    ax.legend([Patch(fc=shade(HUE["gray"], s), ec="k", lw=0.4) for s in SUB_SHADE.values()], [SUBLAB.get(n, n) for n in SUB_SHADE],
              loc="upper right")
    ax.set_title("stiffness-strain correlation", fontsize=12); letter(ax, "b"); ax.texts[-1].set_position((-12, 4))
    save(fig, "fig2_identifiability")




# ---------------------------------------------------------------- figure 4
def fig3() -> None:
    """Least-squares calibration: parameter error (top) and held-out passive laxity error (bottom), same 40 knees."""
    ho = np.load(RES / "lsq" / "heldout.npz")
    T = ho["theta_true"]
    fits = [("gaps", "gaps only, nominal attachments", 0.35), ("nominal", "full exam, nominal attachments", 0.7),
            ("known", "full exam, true attachments", 1.0)]
    W, H = WIDTH, 4.25
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.62 / W, 2.62 / H, 3.55 / W, 1.36 / H])
    x = np.arange(8)
    for o, (key, _, sh) in zip((-0.27, 0, 0.27), fits):
        e = ho[f"theta_{key}"] - T
        ax.bar(x + o, np.sqrt((e ** 2).mean(0)), width=0.27,
               color=[shade(HUE[p.split("_")[1]], sh) for p in PARAM_NAMES], edgecolor="k", lw=0.4)
    e0 = np.sqrt(((T - 0.5) ** 2).mean(0))
    for j in range(8):
        ax.plot([j - 0.42, j + 0.42], [e0[j]] * 2, color="k", lw=1.4, ls="-", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([PLAB[p] for p in PARAM_NAMES]); ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 7.6); ax.set_ylim(0, 0.6); ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_ylabel(r"RMSE $\hat\vartheta$ [-]")
    ax.set_title("parameter error (normalized)", fontsize=12); letter(ax, "a")
    lg = fig.add_axes([4.25 / W, 2.62 / H, 2.2 / W, 1.36 / H]); lg.axis("off")
    lg.legend([Patch(fc=shade(HUE["gray"], sh), ec="k", lw=0.4) for *_, sh in fits] + [Line2D([], [], color="k", lw=1.4, ls="-")],
              [lab for _, lab, _ in fits] + ["prior center"], loc="center left", fontsize=9, handlelength=1.4)
    names = list(ho["h_names"])
    kinds = {"AP": [i for i, n in enumerate(names) if n.split("@")[0] in ("ant", "post") or n.endswith(":ap@45")],
             "VV": [i for i, n in enumerate(names) if n.split("@")[0] in ("var", "val") or n.endswith(":vv@45")],
             "IE": [i for i, n in enumerate(names) if n.split("@")[0] in ("int", "ext") or n.endswith(":ie@45")]}
    est = [("prior_center", "prior center"), ("gaps", "gaps only,\nnominal attachments"),
           ("nominal", "full exam,\nnominal attachments"), ("known", "full exam,\ntrue attachments"),
           ("truth_nominal_geometry", "true parameters,\nnominal attachments")]
    true = ho["h_true"]
    rm = lambda e, idx: np.sqrt(((ho[f"h_{e}"][:, idx] - true[:, idx]) ** 2).mean())
    y = np.arange(len(est))[::-1]
    axb = fig.add_axes([1.95 / W, 0.42 / H, 1.85 / W, 1.62 / H])
    axb.barh(y, [rm(e, kinds["AP"]) for e, _ in est], height=0.55, color=HUE["AP"], edgecolor="k", lw=0.4)
    apmax = max(rm(e, kinds["AP"]) for e, _ in est)
    axb.set_xlabel("AP RMSE [mm]"); axb.set_xlim(0, 1.15 * apmax); axb.xaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
    axb.set_yticks(y); axb.set_yticklabels([lab for _, lab in est], fontsize=10, linespacing=0.95)
    axb.set_title("held-out translation", fontsize=12); letter(axb, "b"); axb.texts[-1].set_position((-140, 4))
    axc = fig.add_axes([4.35 / W, 0.42 / H, 1.95 / W, 1.62 / H])
    for o, kind in zip((0.17, -0.17), ("VV", "IE")):
        axc.barh(y + o, [rm(e, kinds[kind]) for e, _ in est], height=0.34, color=HUE[kind], edgecolor="k", lw=0.4, label=kind)
    rmax = max(rm(e, kinds[k_]) for e, _ in est for k_ in ("VV", "IE"))
    axc.set_xlabel(r"VV, IE RMSE [$^\circ$]"); axc.set_xlim(0, 1.15 * rmax); axc.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    axc.set_yticks(y); axc.set_yticklabels([])
    axc.legend(loc="lower right"); axc.set_title("held-out rotation", fontsize=12); letter(axc, "c"); axc.texts[-1].set_position((-14, 4))
    for a_ in (axb, axc):
        a_.tick_params(axis="y", length=0); a_.set_ylim(-0.6, len(est) - 0.4)
    save(fig, "fig3_calibration")


if __name__ == "__main__":
    setup()
    FIG.mkdir(parents=True, exist_ok=True)
    for w in sys.argv[1:] or ["1", "2", "3", "S2"]:
        globals()[f"fig{w}"]()
        print("figure", w)
