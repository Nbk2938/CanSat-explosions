#!/usr/bin/env python3
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL_FILE  = "outputs/site_event_labels_moast_probable.txt"
OUTPUT_FILE = "outputs/event_timeline.png"

CROSS_RE = re.compile(r"S([123])([123])_+(\d+)")
LOCAL_RE  = re.compile(r"\bS([123])_(\d+)\b")

sites = {1: [], 2: [], 3: []}

with open(LABEL_FILE) as f:
    for line in f:
        parts = line.strip().split("\t", 2)
        if len(parts) < 3:
            continue
        t, label = float(parts[0]), parts[2]
        if "?" in label:
            continue

        m = CROSS_RE.search(label)
        if m and int(m.group(1)) != int(m.group(2)):
            mic = int(m.group(1))
            sites[mic].append((t, label))
            continue

        m = LOCAL_RE.search(label)
        if m:
            mic = int(m.group(1))
            sites[mic].append((t, label))

# ── plot ──────────────────────────────────────────────────────────────────────
COLORS = {1: "#1565C0", 2: "#BF360C", 3: "#2E7D32"}
LABELS = {1: "S1 (32000_iOS)", 2: "S2 (TimeVideo)", 3: "S3 (18000_iOS)"}

fig, ax = plt.subplots(figsize=(16, 4))

for site, evs in sites.items():
    ts = [t for t, _ in evs]
    ax.vlines(ts, site - 0.35, site + 0.35,
              color=COLORS[site], lw=1.5, label=LABELS[site])

ax.set_yticks([1, 2, 3])
ax.set_yticklabels([LABELS[i] for i in [1, 2, 3]])
ax.set_xlabel("Time [s]")
ax.set_title("Known events per site  (uncertain inter-site events excluded)")
ax.grid(axis="x", ls=":", alpha=0.4)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150)
print(f"Saved → {OUTPUT_FILE}")
