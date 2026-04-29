#!/usr/bin/env python3
"""
Align CanSat probe timelines to the ground-truth microphone timeline.

  ScamSat (Probe 6) — spike-train cross-correlation.
    Convention: t_gt = t_scamsat_mission + delta_sc
    Scan range: -700 … -500 s  (GT events ~200-420 s, ScamSat ~830-994 s)

  OBAMA (Probe 5) — interval count-matching (detonation rate correlation).
    Convention: t_gt = t_obama_mission + delta_ob
    Scan range: +100 … +400 s  (GT events ~200-420 s, OBAMA ~49-110 s)

Ground-truth weights: DET = 2, LAUNCH = 1. Inter-site arrivals excluded.
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs"

GT_FILE      = f"{OUTPUT_DIR}/site_event_labels_verified.txt"
SCAMSAT_FILE = f"{OUTPUT_DIR}/ScamSat_events_labels_OOSync.txt"
OBAMA_FILE   = f"{OUTPUT_DIR}/OBAMA_events_labels_OOSync.txt"

WEIGHT_DET    = 2.0
WEIGHT_LAUNCH = 1.0
MATCH_TOL     = 2.0   # [s] coincidence window for ScamSat spike matching

SC_DELTA  = np.arange(-700, -500, 0.2)   # ScamSat offset scan [s]
OB_DELTA  = np.arange( 100,  400, 0.2)   # OBAMA   offset scan [s]


# ── loaders ───────────────────────────────────────────────────────────────────

def load_ground_truth(path):
    """[(time_s, weight), ...] — local events only, no inter-site, no uncertain."""
    cross_re = re.compile(r"S([123])([123])_+\d+")
    local_re  = re.compile(r"\bS([123])_\d+\b")
    events = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 2)
            if len(parts) < 3:
                continue
            t, label = float(parts[0]), parts[2]
            if "?" in label or cross_re.search(label):
                continue
            if not local_re.search(label):
                continue
            weight = WEIGHT_LAUNCH if "LAUNCH" in label else WEIGHT_DET
            events.append((t, weight))
    return sorted(events)


def load_scamsat(path):
    """[time_s, ...] — start time of each ScamSat detection."""
    events = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 2)
            if len(parts) >= 2:
                events.append(float(parts[0]))
    return sorted(events)


def load_obama(path):
    """[(t_start, t_end, count), ...] — one entry per telemetry packet."""
    packets: dict = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 2)
            if len(parts) < 3:
                continue
            t_start, t_end = float(parts[0]), float(parts[1])
            key = (t_start, t_end)
            packets[key] = packets.get(key, 0) + 1
    return sorted((a, b, c) for (a, b), c in packets.items())


# ── ScamSat: weighted spike-train cross-correlation ──────────────────────────

def scamsat_scores(gt_events, sc_events, deltas):
    gt_t = np.array([t for t, _ in gt_events])
    gt_w = np.array([w for _, w in gt_events])
    sc   = np.array(sc_events)
    scores = np.zeros(len(deltas))
    for j, delta in enumerate(deltas):
        sc_shifted = sc + delta          # bring ScamSat into GT timeline
        for i, t_gt in enumerate(gt_t):
            if np.any(np.abs(sc_shifted - t_gt) <= MATCH_TOL):
                scores[j] += gt_w[i]
    return scores


def report_scamsat(best_delta, gt_events, sc_events):
    sc = np.array(sc_events)
    sc_shifted = sc + best_delta
    print(f"\nScamSat best delta = {best_delta:.1f} s  "
          f"(t_wall ≈ t_scamsat + {best_delta:.1f})")
    print(f"{'GT time':>10}  {'SC time':>10}  {'Δt':>6}  weight  label")
    print("-" * 55)
    matched_sc = set()
    for t_gt, w in gt_events:
        diffs = np.abs(sc_shifted - t_gt)
        idx   = np.argmin(diffs)
        if diffs[idx] <= MATCH_TOL:
            matched_sc.add(idx)
            label = "DET" if w == WEIGHT_DET else "LAUNCH"
            print(f"{t_gt:10.2f}  {sc[idx]:10.2f}  {t_gt - sc_shifted[idx]:+6.2f}  "
                  f"  {w:.0f}    {label}")
    unmatched = [sc[i] for i in range(len(sc)) if i not in matched_sc]
    if unmatched:
        print(f"\n  {len(unmatched)} unmatched ScamSat detections "
              f"(likely false positives): {[f'{t:.1f}' for t in unmatched]}")


# ── OBAMA: interval count-matching (dot-product of weighted counts) ───────────

def obama_scores(gt_events, obama_packets, deltas):
    gt_t = np.array([t for t, _ in gt_events])
    gt_w = np.array([w for _, w in gt_events])
    scores = np.zeros(len(deltas))
    for j, delta in enumerate(deltas):
        # transform GT to OBAMA mission time: t_mission = t_gt - delta
        gt_mission = gt_t - delta
        for a, b, count in obama_packets:
            mask = (gt_mission >= a) & (gt_mission < b)
            scores[j] += count * gt_w[mask].sum()
    return scores


def report_obama(best_delta, gt_events, obama_packets):
    gt_t = np.array([t for t, _ in gt_events])
    gt_w = np.array([w for _, w in gt_events])
    gt_mission = gt_t - best_delta
    print(f"\nOBAMA best delta = {best_delta:.1f} s  "
          f"(t_wall ≈ t_obama + {best_delta:.1f})")
    print(f"{'Interval':>22}  {'OBAMA cnt':>9}  {'GT weight':>9}  match")
    print("-" * 55)
    for a, b, count in obama_packets:
        mask = (gt_mission >= a) & (gt_mission < b)
        gt_w_sum = gt_w[mask].sum()
        match = "✓" if gt_w_sum > 0 else ""
        print(f"[{a:6.1f} – {b:6.1f}]  {count:9d}  {gt_w_sum:9.1f}  {match}")


# ── metrics ───────────────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """Wilson 95 % confidence interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def alignment_uncertainty(scores, deltas, threshold=0.90):
    """Half-width of the δ range where score ≥ threshold × peak."""
    above = deltas[scores >= threshold * scores.max()]
    return (above.max() - above.min()) / 2 if len(above) > 1 else 0.0


def greedy_match(gt_t, sc_shifted, tol):
    """One-to-one greedy matching (closest pair first) within ±tol."""
    pairs = sorted(
        (abs(t_gt - t_sc), i, j)
        for i, t_gt in enumerate(gt_t)
        for j, t_sc in enumerate(sc_shifted)
        if abs(t_gt - t_sc) <= tol
    )
    used_gt, used_sc, matched = set(), set(), []
    for d, i, j in pairs:
        if i not in used_gt and j not in used_sc:
            matched.append((i, j, d))
            used_gt.add(i)
            used_sc.add(j)
    return matched, used_gt, used_sc


def scamsat_metrics(best_sc, sc_scores, gt_events, sc_events, tol=MATCH_TOL):
    gt_t = np.array([t for t, _ in gt_events])
    gt_w = np.array([w for _, w in gt_events])
    sc   = np.array(sc_events)
    sc_shifted = sc + best_sc

    matched, used_gt, used_sc = greedy_match(gt_t, sc_shifted, tol)

    n_gt      = len(gt_t)
    n_sc      = len(sc)
    tp        = len(matched)
    fp        = n_sc - len(used_sc)
    fn        = n_gt - len(used_gt)

    gt_i_matched = list(used_gt)
    tp_det    = int(sum(gt_w[i] == WEIGHT_DET    for i in gt_i_matched))
    tp_launch = int(sum(gt_w[i] == WEIGHT_LAUNCH for i in gt_i_matched))
    fn_det    = int(sum(gt_w[i] == WEIGHT_DET    for i in range(n_gt) if i not in used_gt))
    fn_launch = int(sum(gt_w[i] == WEIGHT_LAUNCH for i in range(n_gt) if i not in used_gt))

    precision = tp / n_sc if n_sc else 0.0
    recall    = tp / n_gt if n_gt else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Sensitivity: recall at different tolerances
    sensitivity = {}
    for t in [0.5, 1.0, 2.0, 3.0, 5.0]:
        _, ug, _ = greedy_match(gt_t, sc_shifted, t)
        sensitivity[t] = len(ug) / n_gt if n_gt else 0.0

    delta_unc = alignment_uncertainty(sc_scores, SC_DELTA)

    return dict(
        n_gt=n_gt, n_sc=n_sc, tp=tp, fp=fp, fn=fn,
        tp_det=tp_det, tp_launch=tp_launch,
        fn_det=fn_det, fn_launch=fn_launch,
        precision=precision, recall=recall, f1=f1,
        precision_ci=wilson_ci(tp, n_sc),
        recall_ci=wilson_ci(tp, n_gt),
        recall_det_ci=wilson_ci(tp_det, int((gt_w == WEIGHT_DET).sum())),
        recall_launch_ci=wilson_ci(tp_launch, int((gt_w == WEIGHT_LAUNCH).sum())),
        sensitivity=sensitivity,
        delta=best_sc, delta_unc=delta_unc,
    )


def obama_metrics(best_ob, ob_scores, gt_events, obama_packets):
    gt_t = np.array([t for t, _ in gt_events])
    gt_w = np.array([w for _, w in gt_events])
    gt_mission = gt_t - best_ob

    ob_counts = np.array([c for _, _, c in obama_packets], dtype=float)

    gt_sums = np.array([
        gt_w[(gt_mission >= a) & (gt_mission < b)].sum()
        for a, b, _ in obama_packets
    ])
    # GT-side: which events are covered by at least one OBAMA interval
    covered = np.zeros(len(gt_t), dtype=bool)
    for a, b, _ in obama_packets:
        covered |= (gt_mission >= a) & (gt_mission < b)
    n_gt_covered = int(covered.sum())
    n_gt_missed  = int((~covered).sum())

    # Interval-level classification (all OBAMA packets have count > 0 by construction)
    n_packets      = len(obama_packets)
    n_fp_packets   = int((gt_sums == 0).sum())   # OBAMA count > 0, no GT events
    n_tp_packets   = int((gt_sums  > 0).sum())   # OBAMA count > 0, GT events present

    # Pearson correlation between OBAMA counts and GT weighted sums per interval
    if ob_counts.std() > 0 and gt_sums.std() > 0:
        pearson = float(np.corrcoef(ob_counts, gt_sums)[0, 1])
    else:
        pearson = float("nan")

    delta_unc = alignment_uncertainty(ob_scores, OB_DELTA)

    n_gt = len(gt_t)
    n_gt_det    = int((gt_w == WEIGHT_DET).sum())
    n_gt_launch = int((gt_w == WEIGHT_LAUNCH).sum())
    covered_det    = int((covered & (gt_w == WEIGHT_DET)).sum())
    covered_launch = int((covered & (gt_w == WEIGHT_LAUNCH)).sum())

    return dict(
        n_gt=n_gt, n_gt_det=n_gt_det, n_gt_launch=n_gt_launch,
        n_packets=n_packets, n_tp_packets=n_tp_packets, n_fp_packets=n_fp_packets,
        n_gt_covered=n_gt_covered, n_gt_missed=n_gt_missed,
        covered_det=covered_det, covered_launch=covered_launch,
        total_obama_count=int(ob_counts.sum()),
        pearson=pearson,
        coverage_ci=wilson_ci(n_gt_covered, n_gt),
        coverage_det_ci=wilson_ci(covered_det, n_gt_det),
        coverage_launch_ci=wilson_ci(covered_launch, n_gt_launch),
        packet_precision_ci=wilson_ci(n_tp_packets, n_packets),
        delta=best_ob, delta_unc=delta_unc,
    )


def write_metrics_report(sc_m, ob_m, path):
    W = 65

    def h(title):
        return f"\n{'─' * W}\n  {title}\n{'─' * W}"

    def row(label, value, ci=None, indent=4):
        ci_str = f"  [95 % CI: {ci[0]:.2f} – {ci[1]:.2f}]" if ci else ""
        return f"{' ' * indent}{label:<36}{value}{ci_str}"

    lines = [
        "═" * W,
        "  CANSAT EVENT DETECTION METRICS",
        "═" * W,
        "",
        row("Ground truth local events:", f"{sc_m['n_gt']}"),
        row("  — DET events:", f"{int(sc_m['n_gt'] - sc_m['fn_det'] - sc_m['tp_launch'] - sc_m['fn_launch'] + sc_m['tp_det'] + sc_m['fn_det'])}"),
        "",
    ]

    # recompute total det/launch from ob_m which has them directly
    n_gt_det    = ob_m["n_gt_det"]
    n_gt_launch = ob_m["n_gt_launch"]
    lines[4] = row("Ground truth local events:", f"{sc_m['n_gt']}  (DET: {n_gt_det}, LAUNCH: {n_gt_launch})")
    lines.pop(5)

    lines += [
        h("SCAMSAT (Probe 6) — spike-train matching  "
          f"[δ = {sc_m['delta']:.1f} s ± {sc_m['delta_unc']:.1f} s]"),
        "",
        row("Total ScamSat detections:", f"{sc_m['n_sc']}"),
        row("True positives (TP):",
            f"{sc_m['tp']}  (DET: {sc_m['tp_det']}, LAUNCH: {sc_m['tp_launch']})"),
        row("False positives (FP):", f"{sc_m['fp']}"),
        row("False negatives (FN):",
            f"{sc_m['fn']}  (DET: {sc_m['fn_det']}, LAUNCH: {sc_m['fn_launch']})"),
        "",
        row("Precision  (TP / n_sc):", f"{sc_m['precision']:.3f}",
            ci=sc_m['precision_ci']),
        row("Recall — all GT  (TP / n_gt):", f"{sc_m['recall']:.3f}",
            ci=sc_m['recall_ci']),
        row("Recall — DET only:", f"{sc_m['tp_det'] / n_gt_det:.3f}",
            ci=sc_m['recall_det_ci']),
        row("Recall — LAUNCH only:", f"{sc_m['tp_launch'] / n_gt_launch:.3f}",
            ci=sc_m['recall_launch_ci']),
        row("F1 score:", f"{sc_m['f1']:.3f}"),
        "",
        "    Recall sensitivity to matching tolerance:",
    ]
    for t, r in sc_m["sensitivity"].items():
        lines.append(f"      ±{t:.1f} s  →  {r:.3f}")

    lines += [
        "",
        "    Confidence caveats:",
        "      · Alignment offset uncertainty propagates directly into",
        f"        matching: ±{sc_m['delta_unc']:.1f} s on δ means TP/FP counts",
        "        could shift by a few events.",
        "      · Acoustic travel time from explosion to CanSat varies",
        "        with altitude (~±0.6 s over the drop), absorbed into",
        "        the tolerance window but not explicitly corrected.",
        "      · Small sample (N={}) → wide CIs.".format(sc_m['n_sc']),

        h("OBAMA (Probe 5) — interval count matching  "
          f"[δ = {ob_m['delta']:.1f} s ± {ob_m['delta_unc']:.1f} s]"),
        "",
        row("Total telemetry packets:", f"{ob_m['n_packets']}"),
        row("  TP packets (GT events present):", f"{ob_m['n_tp_packets']}",
            ci=ob_m['packet_precision_ci']),
        row("  FP packets (no GT events):", f"{ob_m['n_fp_packets']}"),
        "",
        row("GT events covered by any packet:", f"{ob_m['n_gt_covered']} / {ob_m['n_gt']}",
            ci=ob_m['coverage_ci']),
        row("  — DET covered:", f"{ob_m['covered_det']} / {n_gt_det}",
            ci=ob_m['coverage_det_ci']),
        row("  — LAUNCH covered:", f"{ob_m['covered_launch']} / {n_gt_launch}",
            ci=ob_m['coverage_launch_ci']),
        row("GT events outside all packets:", f"{ob_m['n_gt_missed']}"),
        "",
        row("Total OBAMA reported count:", f"{ob_m['total_obama_count']}"),
        row("Pearson corr. (count vs GT weight):", f"{ob_m['pearson']:.3f}"),
        "",
        "    Confidence caveats:",
        "      · Exact timing within each interval unknown; only",
        "        interval-level matching is possible.",
        "      · 'FP packets' may be real events not in ground truth",
        "        (e.g. events at sites not labelled in mic recordings).",
        "      · Very small sample (N={} packets) → CIs are wide.".format(ob_m["n_packets"]),
        "      · OBAMA transmits events in bursts; counts per packet",
        "        aggregate events from varying window sizes.",

        "\n" + "═" * W,
        "  COMPARISON SUMMARY",
        "═" * W,
        "",
        f"{'Metric':<38} {'ScamSat':>10} {'OBAMA':>10}",
        "─" * W,
        f"{'Recall (all GT events)':<38} {sc_m['recall']:>10.3f} "
        f"{'—':>10}",
        f"{'Coverage (GT events in any window)':<38} {'—':>10} "
        f"{ob_m['n_gt_covered'] / ob_m['n_gt']:>10.3f}",
        f"{'Precision / TP-packet rate':<38} {sc_m['precision']:>10.3f} "
        f"{ob_m['n_tp_packets'] / ob_m['n_packets']:>10.3f}",
        f"{'F1 score':<38} {sc_m['f1']:>10.3f} {'—':>10}",
        f"{'False positives (count / packets)':<38} {sc_m['fp']:>10} "
        f"{ob_m['n_fp_packets']:>10}",
        f"{'Alignment δ uncertainty [s]':<38} {sc_m['delta_unc']:>10.1f} "
        f"{ob_m['delta_unc']:>10.1f}",
        "",
        "  Note: OBAMA metrics are interval-level; ScamSat metrics are",
        "  event-level. Direct comparison is indicative only.",
        "═" * W,
    ]

    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    print(text)
    print(f"Metrics → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    gt      = load_ground_truth(GT_FILE)
    sc      = load_scamsat(SCAMSAT_FILE)
    obama   = load_obama(OBAMA_FILE)

    print(f"Ground truth local events : {len(gt)}")
    print(f"ScamSat detections        : {len(sc)}")
    print(f"OBAMA telemetry packets   : {len(obama)}")

    sc_scores = scamsat_scores(gt, sc, SC_DELTA)
    ob_scores = obama_scores(gt, obama, OB_DELTA)

    best_sc = SC_DELTA[np.argmax(sc_scores)]
    best_ob = OB_DELTA[np.argmax(ob_scores)]

    report_scamsat(best_sc, gt, sc)
    report_obama(best_ob, gt, obama)

    sc_m = scamsat_metrics(best_sc, sc_scores, gt, sc)
    ob_m = obama_metrics(best_ob, ob_scores, gt, obama)
    write_metrics_report(sc_m, ob_m, f"{OUTPUT_DIR}/cansat_metrics.txt")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    fig.suptitle("CanSat timeline alignment — score vs. offset δ", fontsize=13)

    ax = axes[0]
    ax.plot(SC_DELTA, sc_scores, color="#1565C0", lw=1.2)
    ax.axvline(best_sc, color="red", ls="--", lw=1.2,
               label=f"best δ = {best_sc:.1f} s")
    ax.set_title("ScamSat — spike-train cross-correlation (weighted)")
    ax.set_xlabel("δ  [s]  (t_wall = t_scamsat + δ)")
    ax.set_ylabel("Weighted matched GT score")
    ax.legend()
    ax.grid(ls=":", alpha=0.4)

    ax = axes[1]
    ax.plot(OB_DELTA, ob_scores, color="#BF360C", lw=1.2)
    ax.axvline(best_ob, color="red", ls="--", lw=1.2,
               label=f"best δ = {best_ob:.1f} s")
    ax.set_title("OBAMA — interval count-matching (dot-product)")
    ax.set_xlabel("δ  [s]  (t_wall = t_obama + δ)")
    ax.set_ylabel("Count × GT-weight score")
    ax.legend()
    ax.grid(ls=":", alpha=0.4)

    out = f"{OUTPUT_DIR}/cansat_alignment.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot → {out}")

    # ── event comparison plot ──────────────────────────────────────────────────
    plot_event_comparison(best_sc, best_ob, gt, sc, obama)


def plot_event_comparison(best_sc, best_ob, gt_events, sc_events, obama_packets):
    from matplotlib.lines import Line2D

    gt_t  = np.array([t for t, _ in gt_events])
    gt_w  = np.array([w for _, w in gt_events])
    sc    = np.array(sc_events)
    sc_shifted = sc + best_sc

    # ScamSat recording window in wall-clock (use actual event span + margin)
    sc_wall_start = sc_shifted.min() - 2
    sc_wall_end   = sc_shifted.max() + 2

    # OBAMA recording window in wall-clock (first packet start → last packet end)
    ob_wall_start = min(a for a, _, _ in obama_packets) + best_ob
    ob_wall_end   = max(b for _, b, _ in obama_packets) + best_ob

    # shared x range: cover GT events + both CanSat windows
    x_min = min(gt_t.min(), sc_wall_start, ob_wall_start) - 5
    x_max = max(gt_t.max(), sc_wall_end,   ob_wall_end)   + 5

    # which ScamSat events are matched
    matched_sc = set()
    for t_gt in gt_t:
        diffs = np.abs(sc_shifted - t_gt)
        idx   = int(np.argmin(diffs))
        if diffs[idx] <= MATCH_TOL:
            matched_sc.add(idx)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True, sharex=True)
    fig.suptitle("CanSat events vs ground truth (aligned)", fontsize=13)

    # ── panel 1: ScamSat timeline ─────────────────────────────────────────────
    ax = axes[0]

    # ScamSat recording window
    ax.axvspan(sc_wall_start, sc_wall_end, color="#A5D6A7", alpha=0.15,
               label="ScamSat detection window")
    ax.axvline(sc_wall_start, color="#2E7D32", lw=1, ls="--", alpha=0.6)
    ax.axvline(sc_wall_end,   color="#2E7D32", lw=1, ls="--", alpha=0.6)

    # GT events: upward stems — DET dark blue, LAUNCH light blue
    for t, w in gt_events:
        col = "#1565C0" if w == WEIGHT_DET else "#90CAF9"
        ax.vlines(t, 0, w, color=col, lw=1.8, zorder=3)
        ax.plot(t, w, "o", color=col, ms=4, zorder=4)

    # ScamSat events: downward stems — green matched, red unmatched
    for i, t_sc in enumerate(sc_shifted):
        col = "#2E7D32" if i in matched_sc else "#C62828"
        ax.vlines(t_sc, 0, -1.5, color=col, lw=1.8, zorder=3)
        ax.plot(t_sc, -1.5, "v", color=col, ms=6, zorder=4)

    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax.set_ylim(-2.5, 3.0)
    ax.set_yticks([-1.5, 0, 1, 2])
    ax.set_yticklabels(["ScamSat", "0", "LAUNCH", "DET"])
    ax.set_title(f"ScamSat vs ground truth  (δ = {best_sc:.1f} s,  tol = ±{MATCH_TOL} s)")
    ax.legend(handles=[
        Line2D([0], [0], color="#1565C0", lw=2, label="GT — DET  (weight 2)"),
        Line2D([0], [0], color="#90CAF9", lw=2, label="GT — LAUNCH  (weight 1)"),
        Line2D([0], [0], color="#2E7D32", lw=2, marker="v", ms=6, label="ScamSat matched"),
        Line2D([0], [0], color="#C62828", lw=2, marker="v", ms=6, label="ScamSat unmatched"),
        Line2D([0], [0], color="#2E7D32", lw=1, ls="--", label="ScamSat window"),
    ], fontsize=8, loc="upper right")
    ax.grid(axis="x", ls=":", alpha=0.4)

    # ── panel 2: OBAMA interval comparison ────────────────────────────────────
    ax = axes[1]

    ob_counts = np.array([c for _, _, c in obama_packets], dtype=float)
    gt_mission = gt_t - best_ob

    # Split GT weight per interval into DET and LAUNCH
    gt_det_sums = np.array([
        gt_w[(gt_mission >= a) & (gt_mission < b) & (gt_w == WEIGHT_DET)].sum()
        for a, b, _ in obama_packets
    ])
    gt_launch_sums = np.array([
        gt_w[(gt_mission >= a) & (gt_mission < b) & (gt_w == WEIGHT_LAUNCH)].sum()
        for a, b, _ in obama_packets
    ])
    gt_sums = gt_det_sums + gt_launch_sums

    norm = max(ob_counts.max(), gt_sums.max())
    ob_norm        = ob_counts     / norm
    gt_det_norm    = gt_det_sums   / norm
    gt_launch_norm = gt_launch_sums / norm

    # OBAMA recording window
    ax.axvspan(ob_wall_start, ob_wall_end, color="#FFCCBC", alpha=0.20)
    ax.axvline(ob_wall_start, color="#BF360C", lw=1, ls="--", alpha=0.6)
    ax.axvline(ob_wall_end,   color="#BF360C", lw=1, ls="--", alpha=0.6)

    # Bars spanning each interval's true wall-clock width
    MIN_BAR_W = 3.0   # minimum bar width [s] so narrow intervals stay legible
    for i, (a, b, _) in enumerate(obama_packets):
        t_start_wall = a + best_ob
        t_end_wall   = b + best_ob
        w_full = max(t_end_wall - t_start_wall, MIN_BAR_W)
        w_half = w_full * 0.42
        mid    = (t_start_wall + t_end_wall) / 2

        # OBAMA bar (left)
        ax.bar(mid - w_half / 2, ob_norm[i], width=w_half,
               color="#BF360C", alpha=0.80)
        ax.text(mid - w_half / 2, ob_norm[i] + 0.02, f"{int(ob_counts[i])}",
                ha="center", va="bottom", fontsize=7, color="#BF360C")

        # GT bar (right) — stacked: DET (dark blue) bottom, LAUNCH (light blue) on top
        ax.bar(mid + w_half / 2, gt_det_norm[i], width=w_half,
               color="#1565C0", alpha=0.80)
        ax.bar(mid + w_half / 2, gt_launch_norm[i], width=w_half,
               bottom=gt_det_norm[i], color="#90CAF9", alpha=0.80)
        if gt_sums[i] > 0:
            ax.text(mid + w_half / 2, gt_det_norm[i] + gt_launch_norm[i] + 0.02,
                    f"{gt_sums[i]:.0f}", ha="center", va="bottom",
                    fontsize=7, color="#1565C0")

    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Wall-clock time  [s from S1 start]")
    ax.set_ylabel("Normalised value")
    ax.set_title(f"OBAMA reported count vs bucketed GT weight  (δ = {best_ob:.1f} s)")
    ax.legend(handles=[
        Line2D([0], [0], color="#BF360C", lw=8, alpha=0.80, label="OBAMA count"),
        Line2D([0], [0], color="#1565C0", lw=8, alpha=0.80, label="GT — DET weight"),
        Line2D([0], [0], color="#90CAF9", lw=8, alpha=0.80, label="GT — LAUNCH weight"),
        Line2D([0], [0], color="#BF360C", lw=1, ls="--",    label="OBAMA window"),
    ], fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.3)
    ax.grid(axis="y", ls=":", alpha=0.4)

    out2 = f"{OUTPUT_DIR}/cansat_alignment_events.png"
    plt.savefig(out2, dpi=150)
    print(f"Plot → {out2}")


if __name__ == "__main__":
    main()
