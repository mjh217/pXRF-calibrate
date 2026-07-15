#!/usr/bin/env python3
"""
eiv_crossplots.py
==================

Build an X x Y grid of pXRF calibration cross-plots.

Each panel is one (element, session date) pair and shows:

  * The 1:1 reference as a dashed black line.
  * Retained standards, coloured/shaped per standard, with x error bars
    (sigma_x) and y error bars (sigma_y_eff).                  [screen_passed.csv]
  * Excluded-but-measured standards greyed out, read straight from the
    screening output (already carrying x, y and effective errors, plus the
    rule/reason that dropped them).                        [screen_exclusions.csv]
  * The session EIV slope forced through the origin (y = m x) as an orange
    line, with 1-sigma (darker) and 2-sigma (lighter) fan bands built from
    sigma_m.                                             [per_session_slopes.csv]
  * A bottom-right info box: session m +/- sigma_m, the all-sessions CV and
    reduced chi2 (2 d.p.), and the per-standard exclusion reasons
    ("manual excl." / "below LOD" / ...).                 [metrics_after_rule4.csv]

Configure the PANELS list and GRID below, then run:  python3 eiv_crossplots.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# ----------------------------------------------------------------------------
# 1. CONFIGURATION  -- edit this block
# ----------------------------------------------------------------------------
FILES = {
    "passed":     f"screening_outputs/screen_passed.csv",        # retained readings
    "exclusions": f"screening_outputs/screen_exclusions.csv",    # excluded readings + reasons
    "metrics":    f"screening_outputs/metrics_after_rule4.csv",  # element-level (all-sessions) CV
    "slopes":     f"calibration_outputs/per_session_slopes.csv", # session m, sigma_m, red_chi2
}

# One (element, date) per panel, filling the grid row-major (left->right, top->bottom).
PANELS = [
    ("Fe", "2025-05-20"),
    ("V",  "2025-05-20"),
    ("Ni", "2025-05-20"),
    ("Th", "2025-05-20"),
#    ("P", "2025-02-27"),
#    ("P", "2025-04-18"),
#    ("P", "2025-05-13"),
#    ("P", "2025-05-27"),
]
GRID = (2, 2)                    # (nrows, ncols)  -- must hold len(PANELS)
PANEL_SIZE = 4.5                 # inches per panel (square)
OUTFILE = "figures/eiv_fits.png"
#OUTFILE = "figures/P_eiv_fits.png"
LEGEND_PANEL = 0                 # index of panel that carries the standard legend

# ----------------------------------------------------------------------------
# 2. STYLE  -- marker/colour per standard, sampled from the reference figure
# ----------------------------------------------------------------------------
# order here == legend order
STANDARD_STYLE = {
    "Carbonate-Rich Shale": dict(color="#3170B5", marker="o"),
    "Marine Shale":         dict(color="#2D9A3E", marker="s"),
    "Black Shale":          dict(color="#C23B2F", marker="^"),
    "Carbonatite":          dict(color="#7EC8E3", marker="D"),
    "Phosphate":            dict(color="#9B7EC8", marker="p"),
}
FIT_COLOR   = "#D35400"          # orange fit line
GREY_FACE   = "#BFBFBF"
GREY_EDGE   = "#8A8A8A"
MARKER_SIZE = 10
EDGE_W      = 0.8
# Per-shape size multipliers so all markers read at a similar visual weight
# (a square/diamond at a given ms covers more area than a circle/triangle).
SIZE_SCALE = {"o": 1.00, "s": 0.90, "^": 1.00, "D": 0.78, "p": 0.98}


def msize(marker):
    return MARKER_SIZE * SIZE_SCALE.get(marker, 1.0)

# Screening rule code -> short label used in the info box.
# 1 certified-zero | 2 all readings <LOD | 3 out of concentration range
# 4 manual | 5 too few standards left after rules 1-4
RULE_LABEL = {1: "cert. zero", 2: "below LOD", 3: "out of range",
              4: "manual excl.", 5: "too few std."}
# When a standard trips several rules, show the highest-priority label.
RULE_PRIORITY = [4, 3, 2, 1, 5]


# ----------------------------------------------------------------------------
# 3. DATA HELPERS
# ----------------------------------------------------------------------------
def load_data():
    return {
        "passed":     pd.read_csv(FILES["passed"]),
        "exclusions": pd.read_csv(FILES["exclusions"]),
        "slopes":     pd.read_csv(FILES["slopes"]),
        "metrics":    pd.read_csv(FILES["metrics"]),
    }


def fit_params(slopes, element, date):
    """(m, sigma_m, red_chi2) for the session, or None if absent."""
    r = slopes[(slopes["element"] == element) & (slopes["date"] == date)]
    if r.empty:
        return None
    r = r.iloc[0]
    return float(r["slope"]), float(r["sigma_slope"]), float(r["red_chi2"])


def allsessions_cv(metrics, element):
    """
    Element-level (all-sessions) coefficient of variation of the per-standard
    response ratios, taken straight from metrics_after_rule4.csv. Constant
    across standards within an element; NaN for elements with too few
    standards to define it.
    """
    r = metrics[metrics["element"] == element]
    if r.empty:
        return np.nan
    return float(r["elem_CV"].iloc[0])


def excluded_points(exclusions, element, date):
    """
    Excluded readings for one (element, date), ready to grey-plot.

    Returns {standard: (x, y, sigma_x, sigma_y_eff)}. <LOD readings (y == '<LOD')
    are placed at y = 0 with no y error bar so below-LOD / certified-zero
    standards still appear on the axis; genuinely missing readings are dropped.
    """
    sub = exclusions[(exclusions["element"] == element) &
                     (exclusions["date"] == date)]
    out = {}
    for std, g in sub.groupby("sample_id"):
        y_raw = g["y"].astype(str)
        is_lod = y_raw.str.contains("<LOD", na=False).to_numpy()
        y_num = pd.to_numeric(g["y"], errors="coerce").to_numpy(dtype=float)
        keep = is_lod | np.isfinite(y_num)      # drop only truly missing rows
        if not keep.any():
            continue
        is_lod = is_lod[keep]
        y = np.where(is_lod, 0.0, y_num[keep])
        sy = g["sigma_y_eff"].to_numpy(dtype=float)[keep]
        sy = np.where(is_lod, 0.0, np.nan_to_num(sy))   # no y-bar on censored pts
        x = g["x"].to_numpy(dtype=float)[keep]
        sx = g["sigma_x"].to_numpy(dtype=float)[keep]
        out[std] = (x, y, sx, sy)
    return out


def exclusion_records(exclusions, element, date):
    """
    (standard, primary_rule) for each standard excluded in this (element, date),
    ordered manual-first. Primary rule = highest-priority rule tripped across
    that standard's readings.
    """
    sub = exclusions[(exclusions["element"] == element) &
                     (exclusions["date"] == date)]
    recs = []
    for std, g in sub.groupby("sample_id"):
        rules = set()
        for val in g["rules"].astype(str):
            rules.update(int(x) for x in val.split(","))
        primary = next((r for r in RULE_PRIORITY if r in rules), min(rules))
        recs.append((std, primary))
    recs.sort(key=lambda sr: RULE_PRIORITY.index(sr[1]))   # manual first
    return recs


# ----------------------------------------------------------------------------
# 4. PANEL DRAWING
# ----------------------------------------------------------------------------
def draw_panel(ax, data, element, date, with_legend):
    scr = data["passed"]

    # --- retained (coloured) points ---------------------------------------
    sub = scr[(scr["element"] == element) & (scr["date"] == date)]
    xy_max = 0.0

    for std, style in STANDARD_STYLE.items():
        g = sub[sub["sample_id"] == std]
        if g.empty:
            continue
        ax.errorbar(
            g["x"], g["y"], xerr=g["sigma_x"], yerr=g["sigma_y_eff"],
            fmt=style["marker"], ms=msize(style["marker"]), mfc=style["color"],
            mec="black", mew=EDGE_W, ecolor="black", elinewidth=1.0,
            capsize=3, capthick=1.0, linestyle="none", zorder=5,
        )
        xy_max = max(xy_max, (g["x"] + g["sigma_x"]).max(),
                     (g["y"] + g["sigma_y_eff"]).max())

    # --- axis limits (from RETAINED points only) --------------------------
    lim = xy_max * 1.06 if xy_max > 0 else 1.0
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    # --- excluded (grey) points -------------------------------------------
    excl_xy = excluded_points(data["exclusions"], element, date)
    for std, (gx, gy, gsx, gsy) in excl_xy.items():
        if not len(gx):
            continue
        gmarker = STANDARD_STYLE.get(std, {}).get("marker", "o")
        overflows = (gx + gsx).max() > lim or (gy + gsy).max() > lim
        ax.errorbar(
            gx, gy, xerr=gsx, yerr=gsy, fmt=gmarker, ms=msize(gmarker),
            mfc=GREY_FACE, mec=GREY_EDGE, mew=EDGE_W, ecolor=GREY_EDGE,
            elinewidth=1.0, capsize=3, capthick=1.0, alpha=0.9,
            linestyle="none", zorder=3, clip_on=overflows,
        )

    # --- info-box exclusion lines -----------------------------------------
    excl_lines = [f"{std}: {RULE_LABEL.get(rule, 'excluded')}"
                  for std, rule in exclusion_records(data["exclusions"],
                                                     element, date)]

    # --- 1:1 reference -----------------------------------------------------
    ax.plot([0, lim], [0, lim], ls="--", color="black", lw=1.0, zorder=1)

    # --- session fit line + 1/2 sigma fan bands ----------------------------
    fp = fit_params(data["slopes"], element, date)
    xline = np.array([0, lim])
    if fp is not None:
        m, sm, chi2 = fp
        ax.fill_between(xline, (m - 2 * sm) * xline, (m + 2 * sm) * xline,
                        color=FIT_COLOR, alpha=0.15, lw=0, zorder=1)
        ax.fill_between(xline, (m - 1 * sm) * xline, (m + 1 * sm) * xline,
                        color=FIT_COLOR, alpha=0.30, lw=0, zorder=1)
        ax.plot(xline, m * xline, color=FIT_COLOR, lw=2.0, zorder=2)
    else:
        m = sm = chi2 = None

    # --- element label (top-left) + session date (bold, top-centre) --------
    ax.text(0.035, 0.965, element, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")
    ax.text(0.5, 0.965, date, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="center")

    # --- info box (bottom-right) -------------------------------------------
    cv = allsessions_cv(data["metrics"], element)
    lines = []
    if m is not None:
        lines.append(rf"Session $m = {m:.3f} \pm {sm:.3f}$")
    if chi2 is not None:
        lines.append(rf"Session $\chi^2_\nu = {chi2:.2f}$")
    if np.isfinite(cv):
        lines.append(rf"All sessions $CV = {cv:.3f}$")
    if excl_lines:
        lines.append("Excluded from fit:")
        lines.extend(excl_lines)
    else:
        lines.append("No exclusions")
    ax.text(0.97, 0.03, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va="bottom", ha="right", family="monospace",
            linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#9A9A9A", lw=0.9))

    # --- legend (top-left panel only) --------------------------------------
    if with_legend:
        handles = [
            Line2D([0], [0], marker=s["marker"], color="none", mfc=s["color"],
                   mec="black", mew=EDGE_W, ms=msize(s["marker"]), label=name)
            for name, s in STANDARD_STYLE.items()
        ]
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(0.01, 0.90), frameon=True, fontsize=8,
                  handletextpad=0.4, borderpad=0.6, labelspacing=1.0)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(labelsize=10)


# ----------------------------------------------------------------------------
# 5. FIGURE ASSEMBLY
# ----------------------------------------------------------------------------
def main():
    nrows, ncols = GRID
    assert len(PANELS) <= nrows * ncols, "PANELS does not fit the GRID"
    data = load_data()

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * PANEL_SIZE, nrows * PANEL_SIZE),
                             squeeze=False)

    for i, (element, date) in enumerate(PANELS):
        ax = axes[i // ncols][i % ncols]
        draw_panel(ax, data, element, date, with_legend=(i == LEGEND_PANEL))

    # blank any unused cells
    for j in range(len(PANELS), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.supxlabel("True concentration (ppm, USGS)", fontsize=12, y=0.08)
    fig.supylabel("pXRF concentration (ppm)", fontsize=12, x=0.02)
    fig.tight_layout(rect=(0.015, 0.055, 1, 1))
    fig.savefig(OUTFILE, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUTFILE}")


if __name__ == "__main__":
    main()
