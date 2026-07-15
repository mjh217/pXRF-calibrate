"""
Standard screening + data preparation stage for pXRF calibration pipeline.
============================================================================

Reads table of certified reference concentrations and the pXRF analyses,
decides per element which USGS standards are usable, computes the empirical
intrinsic-scatter term epsilon, and writes a single enriched per-reading
table (screen_passed.csv) that carries everything the calibration step and
plotting needs -- x, sigma_x, y, sigma_y_c, epsilon and sigma_y_eff.

Full pipeline
--------
    python3 screen_standards.py     # raw data -> screen_passed.csv + metrics
    python3 calibrate.py            # screen_passed.csv -> calibration slopes

Five exclusion rules
--------------------
Rule 1  Certified concentration indistinguishable from zero
        (value <= RULE1_K * uncertainty). A through-origin slope learns
        nothing from an x ~ 0 point.  [AUTOMATIC, from reference table]
        e.g. Se / Marine Shale.

Rule 2  Element truly present but pXRF never reads above its LOD across the
        whole campaign: every reading of that element for that standard is
        <LOD (matrix suppression / interference).  [AUTOMATIC, from readings]
        e.g. Co, K, V / Phosphate ; Th / Carbonatite.

Rule 3  Standard extremely enriched relative to real samples: the phosphate
        or carbonatite standard exceeds RULE3_FACTOR x the most-enriched of
        the three shale standards. A high-leverage point would dominate the
        fit and risks non-linearity.  [AUTOMATIC, from reference table]
        e.g. Ca, Sr, Ce, La / Carbonatite ; P, Ca, Mn, Ce / Phosphate.

Rule 4  Knock-on spectral interference / matrix absorption: the standard
        DOES read above LOD, but the value sits off the linear array. Cannot
        be detected from LOD or reference value alone; curated by hand in
        MANUAL_RULE4_EXCL below (guided by elevated CV / cross-plots).
        [MANUAL] -- applied as a union on top of Rules 1-3.

Rule 5  Element-level incoherence: after Rules 1-4, the standards retained
        for an element must form a coherent through-origin array. The element
        is excluded if it keeps fewer than MIN_STANDARDS standards or its CV
        (below) is >= CV_MAX. Unlike Rules 1-4 this acts on the whole element,
        so it drops every remaining reading of that element.  [AUTOMATIC, from
        the retained-standard CV] -- evaluated over the Rule 1-4 survivors.

Error model (epsilon)
---------------------
epsilon[element][standard] is the empirical intrinsic scatter (ppm): the
pooled within-session standard deviation of a standard's readings, pooled
across all sessions that have at least 2 timed measurement blocks. It captures
reading-to-reading (re-positioning) reproducibility beyond the instrument's
counting error and is used to inflate the y-error using:
        sigma_y_eff = sqrt( sigma_y_c^2 + epsilon^2 )
epsilon is a per-(element,standard) campaign constant, applied to every
reading of that standard.

Coherence metric (CV)
---------------------
For the standards retained for an element, with session-pooled mean pXRF
concentration ybar_j and certified value x_j:
        r_j = ybar_j / x_j ,   rbar = mean_j r_j ,
        CV  = (1/rbar) * sqrt( (1/(n-1)) * sum_j (r_j - rbar)^2 )
An element needs >= MIN_STANDARDS retained standards and CV < CV_MAX to be
calibratable; failing this is Rule 5 (see above).

Outputs (into OUT_DIR)
----------------------
  screen_passed.csv                per reading that survives ALL five rules:
                                   element, date, time, block, reading,
                                   sample_id, x, sigma_x, y, sigma_y_c,
                                   epsilon, sigma_y_eff
                                   (standalone calibration input)

  screen_pass_rules1234.csv        per reading that survives Rules 1-4, i.e.
                                   BEFORE the element-level Rule 5 verdict.
                                   Same columns as screen_passed.csv; it equals
                                   screen_passed.csv plus the readings of
                                   elements that go on to fail Rule 5. Use it to
                                   eyeball per-session behaviour of the Rule 1-4
                                   survivors and spot candidates for manual
                                   Rule 4 exclusion.

  screen_exclusions.csv            per excluded reading, identical columns to
                                   screen_passed.csv plus rules + reason at the
                                   end. Together with screen_passed.csv it
                                   partitions the numeric readings. A reading is
                                   listed with its standard-level rule(s) (1-4),
                                   or with Rule 5 if its standard survived 1-4
                                   but its element is incoherent. Rule-2 pairs
                                   (every reading <LOD) carry no numeric row, so
                                   they appear as one placeholder row each with
                                   y=<LOD and NaN in the fields that need data.

  metrics_after_rules123.csv       per retained (element, standard) after
                                   Rules 1-3: r_j and element rbar / CV / n

  metrics_after_rule4.csv          same, after Rule 4 as well (this CV is the
                                   quantity Rule 5 thresholds on)
"""
import csv
import math
from collections import defaultdict
from datetime import datetime
import numpy as np

# ============================ CONFIG ======================================= #
# All paths and thresholds for the screening stage live here. Nothing is
# imported from other project modules.
REF_PATH = "USGS_pXRF_standards.csv"       # certified reference values
ALLTESTS_PATH = "all_reference_tests.csv"  # instrument export (tab-delim)
OUT_DIR = "screening_outputs"              # where outputs are written

RULE1_K = 1.0        # value <= RULE1_K * uncertainty  -> indistinguishable from 0
RULE3_FACTOR = 4.0   # PC standard > RULE3_FACTOR x max shale -> over-enriched
CV_MAX = 0.5         # coherence guide: CV below this considered a "coherent array"
MIN_STANDARDS = 2    # need at least this many retained standards to calibrate
GAP_MIN = 20         # minutes; a time gap larger than this starts a new block

STD_NAME_FIX = {"Carbonate_Rich Shale": "Carbonate-Rich Shale"}
SHALE_STDS = ["Carbonate-Rich Shale", "Marine Shale", "Black Shale"]
PC_STDS = ["Carbonatite", "Phosphate"]
ALL_STDS = set(SHALE_STDS) | set(PC_STDS)

# ── Rule 4: MANUAL knock-on interference / matrix-absorption exclusions ─────
# Hand-curated. A standard listed here reads above LOD but does not sit on the
# element's linear array. Editing this dict IS the manual screening step.
MANUAL_RULE4_EXCL = {
    "V":  {"Carbonatite"},
    "K":  {"Carbonatite"},
    "Mg": {"Carbonatite"},
    "Zr": {"Carbonatite"},
    "Ti": {"Carbonatite", "Phosphate"},
    "Cu": {"Carbonatite", "Phosphate"},
    "Ba": {"Carbonatite"},
    "Zn": {"Carbonatite"},
    "U":  {"Phosphate"},
    "Ni": {"Phosphate"},
    "Th": {"Phosphate"},
    "Cr": {"Carbonatite", "Phosphate"},
}
# =========================================================================== #


# ---------------------------------------------------------------- parsing ---
def load_reference():
    """element -> {std_name: (value, uncertainty)}  (only the 5 pressed stds)."""
    ref = defaultdict(dict)
    with open(REF_PATH, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        std_names = header[1::2]
        for row in r:
            el = row[0]
            vals = row[1:]
            for i, std in enumerate(std_names):
                if std not in ALL_STDS:
                    continue
                ref[el][std] = (float(vals[2 * i]), float(vals[2 * i + 1]))
    return ref


def load_readings_with_lod():
    """Parse the instrument export, keeping <LOD structure.

    Returns:
      readings       list of dicts (date, time, reading, sample_id, element,
                     conc, err) for NUMERIC readings with err > 0
      physical_rows  list of dicts (date, time, reading, sample_id), one per
                     physical measurement (regardless of numeric content) --
                     used for timed-block splitting
      total_counts / lod_counts  (element, standard) -> int, for Rule 2
      elements       list of element symbols present in the export
    """
    with open(ALLTESTS_PATH, newline="", encoding="utf-8-sig") as f:
        first = f.readline()
        f.seek(0)
        delim = "\t" if first.count("\t") >= first.count(",") else ","
        r = csv.reader(f, delimiter=delim)
        header = next(r)
        rows = list(r)

    sid_idx = header.index("Sample ID")
    date_idx = header.index("Date")
    time_idx = header.index("Time")
    reading_idx = header.index("Reading #")
    elements = [h[:-len(" Concentration")]
                for h in header if h.endswith(" Concentration")]
    conc_idx = {el: header.index(f"{el} Concentration") for el in elements}
    err_idx = {el: header.index(f"{el} Error1s") for el in elements}

    readings = []
    physical_rows = []
    total_counts = defaultdict(int)
    lod_counts = defaultdict(int)
    for row in rows:
        sid = row[sid_idx].strip()
        sid = STD_NAME_FIX.get(sid, sid)
        if sid not in ALL_STDS:
            continue
        date = row[date_idx]
        time = row[time_idx]
        reading = row[reading_idx]
        physical_rows.append(dict(date=date, time=time, reading=reading,
                                  sample_id=sid))
        for el in elements:
            cval = row[conc_idx[el]]
            total_counts[(el, sid)] += 1
            if cval == "<LOD":
                lod_counts[(el, sid)] += 1
                continue
            if cval == "":
                continue
            try:
                conc = float(cval)
                err = float(row[err_idx[el]])
            except ValueError:
                continue
            if err <= 0:
                continue
            readings.append(dict(date=date, time=time, reading=reading,
                                 sample_id=sid, element=el, conc=conc, err=err))
    return readings, physical_rows, total_counts, lod_counts, elements


# ------------------------------------------------- timed blocks + epsilon ---
def split_blocks(session_rows):
    """Split one session's physical readings into timed blocks (a gap larger
    than GAP_MIN minutes starts a new block). Returns a list of blocks, each a
    list of the input row dicts."""
    session_rows = sorted(session_rows, key=lambda r: r["time"])
    blocks = [[session_rows[0]]]
    for prev, curr in zip(session_rows, session_rows[1:]):
        t0 = datetime.strptime(prev["time"], "%H:%M:%S")
        t1 = datetime.strptime(curr["time"], "%H:%M:%S")
        gap = (t1 - t0).seconds / 60
        if gap > GAP_MIN:
            blocks.append([])
        blocks[-1].append(curr)
    return blocks


def assign_blocks(physical_rows):
    """Return block_of: (date, reading) -> block index (1-based), and
    n_blocks_by_date: date -> number of timed blocks in that session."""
    by_date = defaultdict(list)
    for pr in physical_rows:
        by_date[pr["date"]].append(pr)

    block_of = {}
    n_blocks_by_date = {}
    for date, rows in by_date.items():
        blocks = split_blocks(rows)
        n_blocks_by_date[date] = len(blocks)
        for b_idx, block in enumerate(blocks, start=1):
            for pr in block:
                block_of[(date, pr["reading"])] = b_idx
    return block_of, n_blocks_by_date


def compute_epsilon(readings, n_blocks_by_date, ref):
    """Empirical intrinsic scatter epsilon[element][standard] (ppm).

    For every session with >= 2 timed blocks, pool all readings of a given
    (element, standard) in that session and take their sample variance
    (ddof=1, needs >= 2 readings). Pool those session variances across all
    sessions weighted by degrees of freedom (n-1):
        epsilon = sqrt( sum((n_s - 1) * var_s) / sum(n_s - 1) ).
    Sessions with a single block are skipped (no within-session repeat
    structure to assess)."""
    per_session = defaultdict(list)  # (el, std, date) -> [conc, ...]
    for r in readings:
        per_session[(r["element"], r["sample_id"], r["date"])].append(r["conc"])

    within_vars = defaultdict(lambda: defaultdict(list))  # el -> std -> [(n,var)]
    for (el, s, date), concs in per_session.items():
        if n_blocks_by_date.get(date, 1) < 2:
            continue
        if s not in ALL_STDS or el not in ref:
            continue
        if len(concs) < 2:
            continue
        arr = np.asarray(concs, float)
        within_vars[el][s].append((len(arr), float(np.var(arr, ddof=1))))

    epsilon = defaultdict(dict)
    for el in within_vars:
        for s in within_vars[el]:
            obs = within_vars[el][s]
            df = sum(n - 1 for n, _ in obs)
            ss = sum((n - 1) * v for n, v in obs)
            if df > 0:
                epsilon[el][s] = math.sqrt(ss / df)
    return epsilon


# ------------------------------------------------------ rule evaluation -----
def evaluate_rules(ref, total_counts, lod_counts):
    """excl[el][std] = set of rule numbers; reasons[el][std] = reason string."""
    excl = defaultdict(lambda: defaultdict(set))
    reasons = defaultdict(dict)

    for el in ref:
        shale_vals = [ref[el][s][0] for s in SHALE_STDS if s in ref[el]]
        shale_max = max(shale_vals) if shale_vals else 0.0

        for std in ref[el]:
            v, u = ref[el][std]
            rules_here = set()
            why = []

            # Rule 1: certified value indistinguishable from zero
            if v <= RULE1_K * u:
                rules_here.add(1)
                why.append(f"certified value {v:g} <= {RULE1_K:g}x uncertainty "
                           f"{u:g} (~0)")

            # Rule 2: element never above LOD for this standard
            tot = total_counts.get((el, std), 0)
            lod = lod_counts.get((el, std), 0)
            if tot > 0 and lod == tot:
                rules_here.add(2)
                why.append(f"all {tot} readings <LOD")

            # Rule 3: over-enriched phosphate/carbonatite standard
            if std in PC_STDS:
                if shale_max > 0 and v > RULE3_FACTOR * shale_max:
                    rules_here.add(3)
                    why.append(f"{v:g} ppm > {RULE3_FACTOR:g}x max shale "
                               f"({shale_max:g} ppm)")
                elif shale_max == 0 and v > 0:
                    rules_here.add(3)
                    why.append(f"{v:g} ppm vs shales ~0 (over-enriched)")

            # Rule 4: manual knock-on interference list
            if std in MANUAL_RULE4_EXCL.get(el, set()):
                rules_here.add(4)
                why.append("manual: knock-on interference / matrix absorption")

            if rules_here:
                excl[el][std] = rules_here
                reasons[el][std] = "; ".join(why)

    return excl, reasons


# --------------------------------------------------------- metrics ----------
def session_pooled_means(readings):
    """(element, standard) -> (grand mean conc over all sessions, n_readings)."""
    acc = defaultdict(list)
    for r in readings:
        acc[(r["element"], r["sample_id"])].append(r["conc"])
    return {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}


def per_session_means(readings):
    """(element, standard, date) -> (mean conc, n_readings)."""
    acc = defaultdict(list)
    for r in readings:
        acc[(r["element"], r["sample_id"], r["date"])].append(r["conc"])
    return {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}


def coherence(ref_el, pooled_el, retained):
    """r_j per standard (pooled), rbar, CV over the retained standards."""
    rj = {}
    for s in retained:
        if s not in ref_el:
            continue
        x = ref_el[s][0]
        key_pool = pooled_el.get(s)
        if x <= 0 or key_pool is None:
            continue
        rj[s] = key_pool[0] / x
    n = len(rj)
    if n == 0:
        return rj, float("nan"), float("nan"), 0
    rbar = float(np.mean(list(rj.values())))
    if n >= 2 and rbar != 0:
        var = sum((r - rbar) ** 2 for r in rj.values()) / (n - 1)
        cv = math.sqrt(var) / rbar
    else:
        cv = float("nan")
    return rj, rbar, cv, n


# =========================================================== main ===========
def main():
    ref = load_reference()
    readings, physical_rows, total_counts, lod_counts, _ = load_readings_with_lod()

    # timed blocks (per physical reading) and epsilon (per element, standard)
    block_of, n_blocks_by_date = assign_blocks(physical_rows)
    epsilon = compute_epsilon(readings, n_blocks_by_date, ref)

    excl, reasons = evaluate_rules(ref, total_counts, lod_counts)

    # pooled means from ALL numeric readings (used for r_j / CV)
    pooled_all = session_pooled_means(readings)
    pooled_by_el = defaultdict(dict)
    for (el, s), val in pooled_all.items():
        pooled_by_el[el][s] = val

    # retained standard sets per element
    retained_123, retained_1234 = {}, {}
    for el in ref:
        excluded_123 = {s for s in ref[el]
                        if excl[el].get(s) and (excl[el][s] & {1, 2, 3})}
        excluded_1234 = {s for s in ref[el] if excl[el].get(s)}
        retained_123[el] = [s for s in ref[el] if s not in excluded_123]
        retained_1234[el] = [s for s in ref[el] if s not in excluded_1234]

    # Rule 5: element-level coherence verdict, evaluated on the Rule 1-4
    # survivors (>= MIN_STANDARDS retained AND CV < CV_MAX). usable_flag[el] is
    # the pass; rule5_fail[el]/rule5_reason[el] record and explain a failure.
    usable_flag, rule5_fail, rule5_reason = {}, {}, {}
    for el in ref:
        _, _, cv, n = coherence(ref[el], pooled_by_el[el], retained_1234[el])
        passes = (n >= MIN_STANDARDS) and (cv == cv) and (cv < CV_MAX)
        usable_flag[el] = passes
        rule5_fail[el] = not passes
        if not passes:
            if n < MIN_STANDARDS:
                rule5_reason[el] = (f"only {n} standard(s) retained after Rules "
                                    f"1-4 (< MIN_STANDARDS={MIN_STANDARDS})")
            elif not (cv == cv):
                rule5_reason[el] = "element CV undefined after Rules 1-4"
            else:
                rule5_reason[el] = (f"element CV {cv:.5f} >= {CV_MAX:g} after "
                                    f"Rules 1-4 (incoherent array)")

    # ---- write exclusions log (per reading, same schema as screen_passed) ----
    # One row per EXCLUDED reading, laid out exactly like screen_passed.csv with
    # the triggering rule(s) and reason appended. Attribution is minimal: a
    # reading carries its standard-level rules (1-4) if any fired, otherwise
    # Rule 5 when its standard survived 1-4 but its element failed coherence.
    # This makes screen_passed.csv and screen_exclusions.csv an exact partition
    # of the numeric readings. Rule-2 pairs (every reading <LOD) have no numeric
    # readings, so each is emitted once as a placeholder: x/sigma_x from the
    # reference table, y=<LOD, and NaN in the fields that need measured data.
    def excl_reason(el, s, applicable):
        if applicable == {5}:
            return rule5_reason[el]
        return reasons[el][s]

    pairs_with_readings = {(r["element"], r["sample_id"]) for r in readings}
    with open(f"{OUT_DIR}/screen_exclusions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["element", "date", "time", "block", "reading", "sample_id",
                    "x", "sigma_x", "y", "sigma_y_c", "epsilon", "sigma_y_eff",
                    "rules", "reason"])

        # numeric readings that fail screening
        for r in readings:
            el, s = r["element"], r["sample_id"]
            pair_rules = set(excl[el].get(s) or ())
            if pair_rules:
                applicable = pair_rules                 # standard-level (1-4)
            elif rule5_fail.get(el):
                applicable = {5}                        # element-level (Rule 5)
            else:
                continue                                # survives -> not here
            x, sigma_x = ref[el][s]
            eps = epsilon.get(el, {}).get(s, 0.0)
            sy_eff = math.sqrt(r["err"] ** 2 + eps ** 2)
            block = block_of.get((r["date"], r["reading"]), "")
            rules_str = ",".join(str(rn) for rn in sorted(applicable))
            w.writerow([el, r["date"], r["time"], block, r["reading"], s,
                        f"{x:.6f}", f"{sigma_x:.6f}", f"{r['conc']:.6f}",
                        f"{r['err']:.6f}", f"{eps:.6f}", f"{sy_eff:.6f}",
                        rules_str, excl_reason(el, s, applicable)])

        # placeholder rows for excluded pairs with NO numeric readings (Rule 2)
        for el in sorted(excl):
            for s in sorted(excl[el]):
                if (el, s) in pairs_with_readings:
                    continue
                applicable = set(excl[el][s])
                x, sigma_x = ref[el][s]
                y_str = "<LOD" if 2 in applicable else "NaN"
                rules_str = ",".join(str(rn) for rn in sorted(applicable))
                w.writerow([el, "NaN", "NaN", "NaN", "NaN", s,
                            f"{x:.6f}", f"{sigma_x:.6f}", y_str, "NaN",
                            "NaN", "NaN", rules_str,
                            excl_reason(el, s, applicable)])

    # ---- write the Rule 1-4 survivors (screen_pass_rules1234.csv) -----------
    # Same schema as screen_passed.csv, but WITHOUT the Rule 5 element filter:
    # one row per reading whose standard survived Rules 1-4 (x > 0). It is a
    # superset of screen_passed.csv -- the extra rows are exactly the readings
    # of elements that then fail Rule 5 -- and lets per-session behaviour of the
    # survivors be inspected (via date/time) to guide manual Rule 4 curation.
    with open(f"{OUT_DIR}/screen_pass_rules1234.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["element", "date", "time", "block", "reading", "sample_id",
                    "x", "sigma_x", "y", "sigma_y_c", "epsilon", "sigma_y_eff"])
        for r in readings:
            el, s = r["element"], r["sample_id"]
            if s not in retained_1234.get(el, []):
                continue
            x, sigma_x = ref[el][s]
            if x <= 0:
                continue
            eps = epsilon.get(el, {}).get(s, 0.0)
            sy_eff = math.sqrt(r["err"] ** 2 + eps ** 2)
            block = block_of.get((r["date"], r["reading"]), "")
            w.writerow([el, r["date"], r["time"], block, r["reading"], s,
                        f"{x:.6f}", f"{sigma_x:.6f}", f"{r['conc']:.6f}",
                        f"{r['err']:.6f}", f"{eps:.6f}", f"{sy_eff:.6f}"])

    # ---- write coherence metrics after Rules 1-3 and after Rule 4 ------------
    def write_metrics(path, retained_map):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["element", "standard", "ref_value", "pooled_mean_y",
                        "n_readings", "r_j", "elem_r_bar", "elem_CV",
                        "elem_n_standards", "elem_usable"])
            for el in sorted(ref):
                rj, rbar, cv, n = coherence(ref[el], pooled_by_el[el],
                                            retained_map[el])
                usable = (n >= MIN_STANDARDS) and (cv == cv) and (cv < CV_MAX)
                for s in sorted(rj):
                    x = ref[el][s][0]
                    pm = pooled_by_el[el].get(s)
                    w.writerow([el, s, f"{x:g}",
                                f"{pm[0]:.4f}" if pm else "",
                                pm[1] if pm else 0, f"{rj[s]:.5f}",
                                f"{rbar:.5f}" if rbar == rbar else "",
                                f"{cv:.5f}" if cv == cv else "", n, usable])

    write_metrics(f"{OUT_DIR}/metrics_after_rules123.csv", retained_123)
    write_metrics(f"{OUT_DIR}/metrics_after_rule4.csv", retained_1234)

    # ---- write the enriched per-reading table (screen_passed.csv) -----------
    # One row per reading that survives all five rules (usable element = passes
    # Rule 5, standard retained through Rules 1-4, x > 0), carrying the full EIV
    # input (x, sigma_x, y, sigma_y_c) plus epsilon and sigma_y_eff and the
    # timed-block index. This is the single source for both calibration and
    # any plotting, so figures need not re-run the fits.
    with open(f"{OUT_DIR}/screen_passed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["element", "date", "time", "block", "reading", "sample_id",
                    "x", "sigma_x", "y", "sigma_y_c", "epsilon", "sigma_y_eff"])
        kept = 0
        for r in readings:
            el, s = r["element"], r["sample_id"]
            if not usable_flag.get(el):
                continue
            if s not in retained_1234.get(el, []):
                continue
            x, sigma_x = ref[el][s]
            if x <= 0:
                continue
            eps = epsilon.get(el, {}).get(s, 0.0)
            sy_eff = math.sqrt(r["err"] ** 2 + eps ** 2)
            block = block_of.get((r["date"], r["reading"]), "")
            w.writerow([el, r["date"], r["time"], block, r["reading"], s,
                        f"{x:.6f}", f"{sigma_x:.6f}", f"{r['conc']:.6f}",
                        f"{r['err']:.6f}", f"{eps:.6f}", f"{sy_eff:.6f}"])
            kept += 1

    # ---- console summary -----------------------------------------------------
    usable = sorted(el for el in ref if usable_flag[el])
    unusable = sorted(el for el in ref if not usable_flag[el])
    print("Screening complete.")
    print(f"  numeric readings in:   {len(readings)}")
    print(f"  screened readings out: {kept}")
    print(f"  sessions with >=2 blocks (epsilon contributors): "
          f"{sum(1 for n in n_blocks_by_date.values() if n >= 2)}"
          f"/{len(n_blocks_by_date)}")
    print(f"  Elements passing all rules ({len(usable)}): {usable}")
    print(f"  Elements UNUSABLE ({len(unusable)}): {unusable}")
    print("\n  Effect of manual exclusion (Rule 4) on elements it changes (CV before -> after):")
    for el in sorted(ref):
        if not MANUAL_RULE4_EXCL.get(el):
            continue
        _, _, cv3, n3 = coherence(ref[el], pooled_by_el[el], retained_123[el])
        _, _, cv4, n4 = coherence(ref[el], pooled_by_el[el], retained_1234[el])
        if n3 != n4 or (cv3 == cv3 and cv4 == cv4 and abs(cv3 - cv4) > 1e-9):
            c3 = f"{cv3:.3f}" if cv3 == cv3 else "nan"
            c4 = f"{cv4:.3f}" if cv4 == cv4 else "nan"
            print(f"    {el:3s}: n {n3}->{n4},  CV {c3} -> {c4}")


if __name__ == "__main__":
    main()
