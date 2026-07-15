"""
Calibration (fitting) stage for the pXRF calibration pipeline.
==============================================================

Reads the sceened per-reading table written by screen_standards.py
(screen_passed.csv) and fits the zero-intercept errors-in-variables (EIV,
Deming) calibration y = m*x. All screening and error-model work is
done upstream; this stage only fits and reports slopes and homogeneity.
It asseses both between session drift and within session drift.

Full pipeline
--------
    python3 screen_standards.py     # raw data  -> screened_all.csv
    python3 calibrate.py            # screened_all.csv -> the two outputs below

Outputs (into OUT_DIR)
----------------------
  per_session_slopes.csv   one row per (element, session): per-session slope m
                           (with sigma and reduced chi2), the pooled slope
                           across sessions, and the between-session
                           homogeneity chi2_nu (drift BETWEEN sessions).

  per_block_slopes.csv     one row per (element, session, block): per-block
                           slope m, plus the cross-block homogeneity chi2_nu
                           for that session (drift WITHIN a session).
"""
import csv
import math
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize_scalar

# ============================ CONFIG ======================================= #
SCREENED_PATH = "./screening_outputs/screen_passed.csv" # written by screen_standards.py
OUT_DIR = "calibration_outputs"                         # where slope tables are written

HOMOG_THRESHOLD = 2.0   # chi2_nu above this -> flag (between- or within-session)
MIN_SESSION_POINTS = 4  # session fit needs >= this many readings (>=3 dof) and >=2 stds
MIN_BLOCK_POINTS = 3    # single block fit needs >= this many points and >=2 stds 
                        #    (lower threshold as just used to assess within-session drift) 
SIGMA_X_FLOOR = 1e-6    # floor on sigma_x to avoid divide-by-zero in the fit
# =========================================================================== #


# --------------------------------------------------- EIV (Deming) fit ------
def _eiv_chi2(m, xs, sx2, ys, sy2):
    w = 1.0 / sx2
    v = 1.0 / sy2
    denom = w + m * m * v
    X = (w * xs + m * v * ys) / denom
    return np.sum(w * (xs - X) ** 2 + v * (ys - m * X) ** 2)


def fit_eiv_through_origin(xs, sx, ys, sy):
    """Errors-in-variables (Deming) regression of y = m*x through the origin.
    Returns dict(m, sigma_m, chi2, dof, red_chi2, n) or None if n < 2.

    sigma_m is the curvature-based (Delta chi2 = 1) uncertainty, rescaled by
    sqrt(reduced chi2) when the fit is over-dispersed relative to the stated
    errors (red_chi2 > 1) -- standard practice for possibly-misestimated bars.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    sx2 = np.asarray(sx, float) ** 2
    sy2 = np.asarray(sy, float) ** 2
    n = len(xs)
    if n < 2:
        return None

    num = np.sum(xs * ys / sy2)
    den = np.sum(xs * xs / sy2)
    m0 = num / den if den != 0 else 1.0
    if not np.isfinite(m0) or m0 <= 0:
        m0 = 1.0
    res = minimize_scalar(_eiv_chi2, args=(xs, sx2, ys, sy2),
                          bounds=(m0 * 1e-3, m0 * 1e3), method="bounded",
                          options=dict(xatol=1e-10))
    m = res.x
    chi2_min = res.fun
    dof = n - 1

    h = max(abs(m) * 1e-4, 1e-8)
    c0 = _eiv_chi2(m, xs, sx2, ys, sy2)
    cp = _eiv_chi2(m + h, xs, sx2, ys, sy2)
    cm = _eiv_chi2(m - h, xs, sx2, ys, sy2)
    d2 = (cp - 2 * c0 + cm) / h ** 2
    sigma_m_raw = math.sqrt(2.0 / d2) if d2 > 0 else float("nan")
    red_chi2 = chi2_min / dof if dof > 0 else float("nan")
    scale = math.sqrt(red_chi2) if red_chi2 == red_chi2 and red_chi2 > 1 else 1.0
    sigma_m = sigma_m_raw * scale

    return dict(m=m, sigma_m=sigma_m, chi2=chi2_min, dof=dof,
                red_chi2=red_chi2, n=n)


# ------------------------------------------------------------ helpers -------
def load_screened(path):
    """Read screened_all.csv -> list of reading dicts. Every row already
    passed screening and carries the full EIV input + error model."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(dict(
                element=row["element"], date=row["date"], time=row["time"],
                block=row["block"], reading=row["reading"],
                sample_id=row["sample_id"],
                x=float(row["x"]), sigma_x=float(row["sigma_x"]),
                y=float(row["y"]), sigma_y_c=float(row["sigma_y_c"]),
                epsilon=float(row["epsilon"]),
                sigma_y_eff=float(row["sigma_y_eff"]),
            ))
    return out


def fit_group(rows):
    """Fit one (element, session) or (element, session, block) group using the
    precomputed columns. Returns (fit_dict_or_None, n_standards)."""
    xs, sx, ys, sy = [], [], [], []
    for r in rows:
        if r["x"] <= 0:
            continue
        xs.append(r["x"])
        sx.append(max(r["sigma_x"], SIGMA_X_FLOOR))
        ys.append(r["y"])
        sy.append(r["sigma_y_eff"])
    n_stds = len(set(r["sample_id"] for r in rows if r["x"] > 0))
    if len(xs) < 2 or n_stds < 2:
        return None, n_stds
    return fit_eiv_through_origin(xs, sx, ys, sy), n_stds


def homogeneity(ms, sms):
    """Weighted-mean slope and reduced chi2 of a set of slopes about it."""
    ms = np.asarray(ms, float)
    sms = np.asarray(sms, float)
    valid = sms > 0
    if valid.sum() >= 2:
        w = 1.0 / sms[valid] ** 2
        m_bar = float(np.sum(w * ms[valid]) / np.sum(w))
        sigma_bar = float((1.0 / np.sum(w)) ** 0.5)
        chi2 = float(np.sum(w * (ms[valid] - m_bar) ** 2))
        dof = int(valid.sum() - 1)
        red = chi2 / dof if dof > 0 else float("nan")
    elif valid.sum() == 1:
        m_bar = float(ms[valid][0]); sigma_bar = float(sms[valid][0])
        red = float("nan")
    else:
        m_bar = float(np.mean(ms)); sigma_bar = float("nan"); red = float("nan")
    return m_bar, sigma_bar, red


# =========================================================== main ===========
def main():
    long = load_screened(SCREENED_PATH)
    elements = sorted(set(r["element"] for r in long))

    session_rows_out = []
    block_rows_out = []

    for el in elements:
        el_rows = [r for r in long if r["element"] == el]

        by_session = defaultdict(list)
        for r in el_rows:
            by_session[r["date"]].append(r)

        # ── per-session fits ─────────────────────────────────────────────
        session_fits = {}
        for date in sorted(by_session):
            fit, n_stds = fit_group(by_session[date])
            # Enforce the session-level minimum: >= MIN_SESSION_POINTS readings
            # (>= 3 dof for one fitted parameter m). The >= 2 distinct standards
            # requirement is already enforced inside fit_group. This mirrors the
            # explicit MIN_BLOCK_POINTS gate used in the per-block loop below.
            if fit and fit["n"] >= MIN_SESSION_POINTS:
                fit["n_standards"] = n_stds
                session_fits[date] = fit

        if session_fits:
            ms = [f["m"] for f in session_fits.values()]
            sms = [f["sigma_m"] for f in session_fits.values()]
            m_pooled, sigma_pooled, red_homog = homogeneity(ms, sms)
            consistent = (red_homog <= HOMOG_THRESHOLD) \
                if red_homog == red_homog else True
            for date, f in session_fits.items():
                session_rows_out.append(dict(
                    element=el, date=date, n_points=f["n"],
                    n_standards=f["n_standards"], slope=f["m"],
                    sigma_slope=f["sigma_m"], red_chi2=f["red_chi2"],
                    pooled_slope=m_pooled, pooled_sigma=sigma_pooled,
                    homogeneity_red_chi2=red_homog,
                    recommend_single_factor=consistent))

        # ── per-block fits (within-session drift) ────────────────────────
        for date in sorted(by_session):
            by_block = defaultdict(list)
            for r in by_session[date]:
                by_block[r["block"]].append(r)

            block_fits = []
            for b in sorted(by_block, key=lambda x: (x == "", x)):
                rows = by_block[b]
                if len(rows) < MIN_BLOCK_POINTS:
                    continue
                fit, n_stds = fit_group(rows)
                if fit and fit["n"] >= MIN_BLOCK_POINTS:
                    block_fits.append((b, fit, n_stds))

            if not block_fits:
                continue
            bms = [f["m"] for _, f, _ in block_fits]
            bsms = [f["sigma_m"] for _, f, _ in block_fits]
            if len(block_fits) >= 2:
                _, _, red_within = homogeneity(bms, bsms)
            else:
                red_within = float("nan")
            within_ok = (red_within <= HOMOG_THRESHOLD) \
                if red_within == red_within else True

            for b, f, n_stds in block_fits:
                block_rows_out.append(dict(
                    element=el, date=date, block=b,
                    n_blocks_fitted=len(block_fits), n_points=f["n"],
                    n_standards=n_stds, slope=f["m"], sigma_slope=f["sigma_m"],
                    red_chi2=f["red_chi2"],
                    within_session_red_chi2=red_within,
                    within_session_consistent=within_ok))

    # ── write per-session slopes ─────────────────────────────────────────
    with open(f"{OUT_DIR}/per_session_slopes.csv", "w", newline="") as f:
        cols = ["element", "date", "n_points", "n_standards", "slope",
                "sigma_slope", "red_chi2", "pooled_slope", "pooled_sigma",
                "homogeneity_red_chi2", "recommend_single_factor"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(session_rows_out, key=lambda r: (r["element"], r["date"])):
            w.writerow(r)

    # ── write per-block slopes ───────────────────────────────────────────
    with open(f"{OUT_DIR}/per_block_slopes.csv", "w", newline="") as f:
        cols = ["element", "date", "block", "n_blocks_fitted", "n_points",
                "n_standards", "slope", "sigma_slope", "red_chi2",
                "within_session_red_chi2", "within_session_consistent"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(block_rows_out,
                        key=lambda r: (r["element"], r["date"], str(r["block"]))):
            w.writerow(r)

    calibrated = sorted(set(r["element"] for r in session_rows_out))
    excluded = [el for el in elements if el not in set(calibrated)]
    print(f"Read screened data: {len(long)} readings, {len(elements)} elements.")
    print(f"  calibrated: {len(calibrated)} elements produced session slopes"
          + (f"; excluded {len(excluded)} due to insufficent DOF: {', '.join(excluded)}" if excluded else ""))
    print(f"  per_session_slopes.csv: {len(session_rows_out)} (element, session) rows")
    print(f"  per_block_slopes.csv  : {len(block_rows_out)} (element, session, block) rows")


if __name__ == "__main__":
    main()
