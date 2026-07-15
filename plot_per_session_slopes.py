"""
Plot per-session EIV slope vs session date for every calibratable element

  - green dashed line + green box ("Consistent factor OK") when
    recommend_single_factor is True, i.e. between-session homogeneity passes
  - red solid line (pooled slope) + red box ("Individual session factors
    required") when recommend_single_factor is False


Reads per_session_slopes.csv (written by calibrate.py).
Writes PNGs to OUT_DIR
"""
import csv
import os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "calibration_outputs/per_session_slopes.csv"
OUT_DIR = "./figures"
os.makedirs(OUT_DIR, exist_ok=True)

rows = list(csv.DictReader(open(CSV_PATH)))

by_el = defaultdict(list)
for r in rows:
    by_el[r["element"]].append(r)

all_dates = sorted(set(r["date"] for r in rows))

# Order: flagged elements (needing individual session factors) first, in the
# order given in the handover table, then everything else alphabetically.
flagged_order = ["Si", "Ca", "P", "Al", "Mg", "Sr"]
other_els = sorted(e for e in by_el if e not in flagged_order)
element_order = flagged_order + other_els

#flagged_order = ["Fe", "Mo", "Th"]
#flagged_order = ["Mn", "U", "V", "Cu", "Zn", "Pb"]
#flagged_order = ["Al", "Mg", "K", "S", "Ba"]
#flagged_order = ["Si", "Ca", "P", "Sr"]
#element_order = flagged_order 

PANELS_PER_FIG = 6
#PANELS_PER_FIG = 4


def make_figure(elements, out_path):
    n = len(elements)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.6 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, el in zip(axes, elements):
        el_rows = sorted(by_el[el], key=lambda r: r["date"])
        dates = [r["date"] for r in el_rows]
        m = np.array([float(r["slope"]) for r in el_rows])
        sm = np.array([float(r["sigma_slope"]) for r in el_rows])
        m_pooled = float(el_rows[0]["pooled_slope"])
        sigma_pooled = float(el_rows[0]["pooled_sigma"])
        chi2_homog = float(el_rows[0]["homogeneity_red_chi2"])
        consistent = el_rows[0]["recommend_single_factor"] == "True"

        x = np.arange(len(dates))
        ax.errorbar(x, m, yerr=sm, fmt="o", color="#3170b5",
                    ecolor="#3170b5", elinewidth=1.2, capsize=3,
                    markersize=6, zorder=10)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, zorder=1)

        color = "#2d9a3e" if consistent else "#c23b2f"
        linestyle = "--" if consistent else "-"
        ax.axhline(m_pooled, color=color, linestyle=linestyle,
                    linewidth=1.6, zorder=2)

        # y-range: symmetric-ish around data, always including 1.0 and pooled
        lo = min(np.min(m - sm), 1.0, m_pooled) 
        hi = max(np.max(m + sm), 1.0, m_pooled)
        pad = 0.35 * (hi - lo) if hi > lo else 0.05
        ax.set_ylim(lo - pad, hi + pad)

        ax.text(0.01, 0.93, el, transform=ax.transAxes,
                fontsize=15, fontweight="bold", va="top")

        verdict = ("Consistent factor OK" if consistent
                   else "Individual session factors required")
        info = (r"$\bar{m}$" + f" = {m_pooled:.3f} ± {sigma_pooled:.3f}\n"
                r"$\chi^2_{\rm hom}/\nu$" + f" = {chi2_homog:.2f}\n{verdict}")
        ax.text(0.985, 0.93, info, transform=ax.transAxes,
                fontsize=11, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                          edgecolor=color, linewidth=1.6))

        ax.set_ylabel("Slope $m$")
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(dates, rotation=45, ha="right")
    axes[-1].set_xlabel("Session date")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("wrote", out_path)


for i in range(0, len(element_order), PANELS_PER_FIG):
    chunk = element_order[i:i + PANELS_PER_FIG]
    out_path = os.path.join(OUT_DIR, f"session_slopes_{i // PANELS_PER_FIG + 1:02d}.png")
    make_figure(chunk, out_path)
