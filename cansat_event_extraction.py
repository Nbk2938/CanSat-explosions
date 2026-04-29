#!/usr/bin/env python3
"""Extract acoustic event detections from onboard probe data.

OBAMA (Probe 5): events from Excel telemetry (SNR-based detection).
ScamSat (Probe 6): events from binary FFT amplitude file (gain-based detection).
"""

import csv
import numpy as np
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

OUTPUT_DIR = "outputs"


# ── OBAMA ─────────────────────────────────────────────────────────────────────

def extract_obama():
    excel_file = "Data/Probe_5_OBAMA/OBAMA_data_decoded.xlsx"
    csv_out    = f"{OUTPUT_DIR}/OBAMA_data.csv"
    plot_out   = f"{OUTPUT_DIR}/OBAMA_events.png"
    label_out  = f"{OUTPUT_DIR}/OBAMA_events_labels_OOSync.txt"

    wb   = openpyxl.load_workbook(excel_file)
    ws   = wb["decoded"]
    rows = list(ws.iter_rows(values_only=True))
    hdrs = rows[0]
    data = rows[1:]

    def col(name):
        return hdrs.index(name)

    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdrs)
        w.writerows(data)
    print(f"CSV → {csv_out}  ({len(data)} rows)")

    # Collect all valid packet times in order so we can bracket each detection
    packet_times = sorted({
        float(r[col("Time_s")])
        for r in data
        if isinstance(r[col("Time_s")], (int, float))
    })
    prev_time = {t: packet_times[i - 1] if i > 0 else t for i, t in enumerate(packet_times)}

    events = []
    for r in data:
        rt = r[col("Time_s")]
        if not isinstance(rt, (int, float)):
            continue
        for i in range(1, 6):
            snr = r[col(f"evt{i}_snr_dB")]
            if isinstance(snr, (int, float)) and snr != 0:
                events.append((float(rt), float(snr), i))

    print(f"\nOBAMA acoustic event detections: {len(events)}")
    for rt, snr, idx in events:
        print(f"  t={prev_time[rt]:7.1f}–{rt:.1f} s   evt{idx}   SNR={snr:.1f} dB")

    # Label spans [prev_packet, current_packet] — detonation occurred somewhere in that interval
    with open(label_out, "w") as f:
        for rt, snr, idx in events:
            t_start = prev_time[rt]
            f.write(f"{t_start:.6f}\t{rt:.6f}\tOBAMA_evt{idx} SNR={snr:.1f}dB\n")
    print(f"Labels → {label_out}")

    COLORS = ["#1565C0", "#BF360C", "#2E7D32", "#6A1B9A", "#E65100"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    fig.suptitle("OBAMA probe — acoustic event detections", fontsize=13)

    ax = axes[0]
    for i in range(1, 6):
        sub = [(rt, snr) for rt, snr, idx in events if idx == i]
        if sub:
            xs, ys = zip(*sub)
            ax.scatter(xs, ys, s=100, color=COLORS[i - 1], label=f"evt{i}", zorder=3)
            ax.vlines(xs, 0, ys, color=COLORS[i - 1], lw=1.2, alpha=0.5)
    ax.set_ylabel("SNR [dB]")
    ax.set_title("Event SNR over mission timeline")
    ax.legend()
    ax.grid(ls=":", alpha=0.4)

    ax = axes[1]
    cnt = Counter(rt for rt, _, _ in events)
    xs2 = sorted(cnt)
    ys2 = [cnt[x] for x in xs2]
    ax.bar(xs2, ys2, width=2, color="#455A64", alpha=0.8)
    ax.set_ylabel("Events per telemetry packet")
    ax.set_xlabel("Mission time [s]")
    ax.set_title("Event count per telemetry packet")
    ax.grid(ls=":", alpha=0.4)

    plt.savefig(plot_out, dpi=150)
    print(f"Plot → {plot_out}")


# ── ScamSat ───────────────────────────────────────────────────────────────────

def extract_scamsat():
    fft_file  = "Data/Probe_6_ScamSat/fft.txt"
    plot_out  = f"{OUTPUT_DIR}/ScamSat_events.png"
    label_out = f"{OUTPUT_DIR}/ScamSat_events_labels_OOSync.txt"

    # Notebook constants
    TOTAL_DURATION = 3650   # [s] total recording length
    BMP_DURATION   = 3550   # [s] BMP/altitude reference duration
    FFT_OFFSET     = 36     # [s] FFT data lags BMP data by this amount
    T0_GAIN        = 841    # [s] start of gain-detection window (mission time)
    TF_GAIN        = 1002   # [s] end of gain-detection window
    DROP_START     = 829    # [s] drop begins
    DROP_END       = 995    # [s] drop ends / landing
    THRESHOLD      = 1.5    # gain threshold for explosion detection

    audio = np.fromfile(fft_file, dtype="float32")
    N = len(audio)

    def mission_time(i, n_ampls):
        """Convert gain-window index i to mission time [s]."""
        return (((T0_GAIN + 0.7 + i / n_ampls * (TF_GAIN - T0_GAIN)) - FFT_OFFSET)
                * N // BMP_DURATION) * TOTAL_DURATION / N

    # Compute gain in the detection window
    i0 = T0_GAIN * N // BMP_DURATION
    i1 = TF_GAIN * N // BMP_DURATION
    ampls = audio[1 + i0:i1] / audio[i0:i1 - 1]
    n_ampls = len(ampls)

    explosions = []  # (mission_time_s, gain)
    last = False
    for i, g in enumerate(ampls):
        xi = mission_time(i, n_ampls)
        if g > THRESHOLD and DROP_START < xi < DROP_END:
            if not last:
                explosions.append((xi, float(g)))
            last = True
        else:
            last = False

    print(f"\nScamSat explosion detections: {len(explosions)}")
    for t, g in explosions:
        print(f"  t={t:7.1f} s   gain={g:.3f}")

    with open(label_out, "w") as f:
        for k, (t, g) in enumerate(explosions, 1):
            f.write(f"{t:.6f}\t{t + 0.5:.6f}\tScamSat_{k} gain={g:.2f}\n")
    print(f"Labels → {label_out}")

    # ── plot ──
    # Full FFT trace (drop window)
    t0_plot, tf_plot = 815, 1020
    time_full = np.linspace(1, TOTAL_DURATION, N)
    i0p = (t0_plot - FFT_OFFSET) * N // BMP_DURATION
    i1p = (tf_plot - FFT_OFFSET) * N // BMP_DURATION
    t_plot   = time_full[i0p:i1p]
    amp_plot = audio[t0_plot * N // BMP_DURATION:tf_plot * N // BMP_DURATION]
    n = min(len(t_plot), len(amp_plot))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    fig.suptitle("ScamSat (Probe 6) — FFT amplitude & gain-based explosion detection", fontsize=13)

    ax = axes[0]
    ax.plot(t_plot[:n], amp_plot[:n], lw=0.6, color="#1565C0", label="FFT amplitude")
    for t, _ in explosions:
        ax.axvline(t, color="red", lw=1.2, alpha=0.7)
    ax.axvline(DROP_START, color="green",  lw=1, ls="--", alpha=0.5, label="drop start/end")
    ax.axvline(DROP_END,   color="orange", lw=1, ls="--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("FFT amplitude")
    ax.set_title("FFT amplitude (5–10 kHz integral)")
    ax.legend(fontsize=8)
    ax.grid(ls=":", alpha=0.4)

    # Gain trace in detection window
    time_gain = np.linspace(1, TOTAL_DURATION, N)
    i0g = (T0_GAIN - FFT_OFFSET) * N // BMP_DURATION
    t_gain = time_gain[i0g:i0g + n_ampls]
    n2 = min(len(t_gain), n_ampls)

    ax = axes[1]
    ax.plot(t_gain[:n2], ampls[:n2], lw=0.6, color="#37474F", label="gain")
    ax.axhline(THRESHOLD, color="red", ls="--", lw=1, label=f"threshold ({THRESHOLD})")
    for t, _ in explosions:
        ax.axvline(t, color="red", lw=1.2, alpha=0.7)
    ax.axvline(DROP_START, color="green",  lw=1, ls="--", alpha=0.5, label="drop start/end")
    ax.axvline(DROP_END,   color="orange", lw=1, ls="--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Gain  [sample i+1 / sample i]")
    ax.set_xlabel("Mission time [s]")
    ax.set_title("Consecutive-sample gain — explosion spikes exceed threshold")
    ax.legend(fontsize=8)
    ax.grid(ls=":", alpha=0.4)

    plt.savefig(plot_out, dpi=150)
    print(f"Plot → {plot_out}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extract_obama()
    extract_scamsat()
