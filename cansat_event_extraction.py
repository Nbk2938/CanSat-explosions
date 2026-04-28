#!/usr/bin/env python3
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import csv

EXCEL_FILE  = "Data/Probe_5_OBAMA/OBAMA_data_decoded.xlsx"
CSV_OUT     = "outputs/OBAMA_data.csv"
PLOT_OUT    = "outputs/OBAMA_events.png"

# ── load ──────────────────────────────────────────────────────────────────────
wb   = openpyxl.load_workbook(EXCEL_FILE)
ws   = wb["decoded"]
rows = list(ws.iter_rows(values_only=True))
hdrs = rows[0]
data = rows[1:]

def col(name): return hdrs.index(name)

# ── export CSV ────────────────────────────────────────────────────────────────

with open(CSV_OUT, "w", newline="") as f:
    csv.writer(f).writerow(hdrs)
    csv.writer(f).writerows(data)
print(f"CSV → {CSV_OUT}  ({len(data)} rows)")

# ── extract acoustic events ───────────────────────────────────────────────────
events = []  # (mission_time_s, snr_dB, evt_index)
for r in data:
    rt = r[col("Time_s")]
    if not isinstance(rt, (int, float)):
        continue
    for i in range(1, 6):
        snr = r[col(f"evt{i}_snr_dB")]
        if isinstance(snr, (int, float)) and snr != 0:
            events.append((float(rt), float(snr), i))

print(f"\nAcoustic event detections: {len(events)}")
for rt, snr, idx in events:
    print(f"  t={rt:7.1f} s   evt{idx}   SNR={snr:.1f} dB")

# ── plot ──────────────────────────────────────────────────────────────────────
COLORS = ["#1565C0", "#BF360C", "#2E7D32", "#6A1B9A", "#E65100"]

fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
fig.suptitle("OBAMA probe — acoustic event detections", fontsize=13)

ax = axes[0]
for i in range(1, 6):
    sub = [(rt, snr) for rt, snr, idx in events if idx == i]
    if sub:
        xs, ys = zip(*sub)
        ax.scatter(xs, ys, s=100, color=COLORS[i-1], label=f"evt{i}", zorder=3)
        ax.vlines(xs, 0, ys, color=COLORS[i-1], lw=1.2, alpha=0.5)
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

plt.savefig(PLOT_OUT, dpi=150)
print(f"\nPlot → {PLOT_OUT}")




"""
TODO: 

- add the scamsat events extraction
- see if I get something out of the gopro footage of BARBIE
"""
