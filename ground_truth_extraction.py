"""Extract ground truth data from Data/Explosion videos
    - extract audio from all videos
    - each video is time stamped (date and hours:minutes:seconds.milliseconds), align the 3 audios with
    - plt the 3 audios aligned according the unique timeline.
"""

import os
import re
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks, spectrogram

from moviepy import VideoFileClip

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "Data", "Explosion Videos")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filename-derived timestamps are wrong for these files (edited in post).
# Each entry is (frame_index, timestamp) — the wall-clock time of that specific frame.
# The true start is back-computed as: timestamp - frame_index / video_fps.
MANUAL_FRAME_REFS: dict[str, tuple[int, datetime]] = {
    "20260205_140018000_iOS.MOV":      (0, datetime(2026, 2, 5, 15,  1, 57, 120_000)),
    "20260205_150632000_iOS.MP4":      (1, datetime(2026, 2, 5, 15,  0, 51, 577_000)),
    "TimeVideo_20260205_150217~3.mp4": (0, datetime(2026, 2, 5, 15,  3, 59, 891_000)),
}

# Detection tuning
RMS_WINDOW_MS   = 50    # RMS envelope window [ms]
MIN_GAP_S       = 1.0   # minimum seconds between two detected events
SIGMA_THRESHOLD = 3.0   # how many std above mean to flag as an event
BAND_HZ         = (20, 2000)  # frequency band for energy detector [Hz]


# ---------------------------------------------------------------------------
# Timestamp / audio helpers
# ---------------------------------------------------------------------------

def parse_timestamp(filename: str) -> datetime:
    """Parse the recording start time from a video filename.

    Supported patterns:
      YYYYMMDD_HHMMSSmmm_iOS.*   e.g. 20260205_140018000_iOS.MOV
      TimeVideo_YYYYMMDD_HHMMSS~N.*  e.g. TimeVideo_20260205_150217~3.mp4
    """
    base = os.path.basename(filename)

    # Pattern 1: YYYYMMDD_HHMMSSmmm
    m = re.search(r"(\d{8})_(\d{6})(\d{3})", base)
    if m:
        date_str, time_str, ms_str = m.group(1), m.group(2), m.group(3)
        return datetime.strptime(f"{date_str}{time_str}{ms_str}", "%Y%m%d%H%M%S%f")

    # Pattern 2: TimeVideo_YYYYMMDD_HHMMSS
    m = re.search(r"(\d{8})_(\d{6})", base)
    if m:
        date_str, time_str = m.group(1), m.group(2)
        return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")

    raise ValueError(f"Cannot parse timestamp from filename: {filename}")


def extract_audio(video_path: str):
    """Return (samples, sample_rate) as mono float32 array."""
    clip = VideoFileClip(video_path)
    audio = clip.audio
    sr = int(audio.fps)
    samples = audio.to_soundarray(fps=sr)
    clip.close()
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples.astype(np.float32), sr


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def rms_envelope(samples: np.ndarray, sr: int, window_ms: float = RMS_WINDOW_MS):
    """Return (times_s, rms) arrays."""
    win = max(1, int(sr * window_ms / 1000))
    n_frames = len(samples) // win
    frames = samples[: n_frames * win].reshape(n_frames, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    times = (np.arange(n_frames) * win + win // 2) / sr
    return times, rms


def band_energy_envelope(samples: np.ndarray, sr: int,
                         fmin: float = BAND_HZ[0], fmax: float = BAND_HZ[1],
                         window_ms: float = 100):
    """Return (times_s, energy) summed over [fmin, fmax] Hz."""
    win = max(64, int(sr * window_ms / 1000))
    freqs, times, Sxx = spectrogram(samples, fs=sr, nperseg=win, noverlap=win // 2)
    band = (freqs >= fmin) & (freqs <= fmax)
    energy = Sxx[band, :].sum(axis=0)
    return times, energy


def detect_events(samples: np.ndarray, sr: int) -> list[dict]:
    """
    Detect candidate explosion events with two methods:
      - 'RMS'  : peak in the RMS amplitude envelope
      - 'Band' : peak in the band-limited spectral energy
    Returns list of dicts sorted by time_s.
    """
    min_gap_frames_rms  = max(1, int(MIN_GAP_S / (RMS_WINDOW_MS / 1000)))
    events = []

    # --- RMS peaks ---
    t_rms, rms = rms_envelope(samples, sr)
    thresh_rms = np.mean(rms) + SIGMA_THRESHOLD * np.std(rms)
    peaks_rms, _ = find_peaks(rms, height=thresh_rms, distance=min_gap_frames_rms)
    for idx in peaks_rms:
        events.append({
            "time_s":    float(t_rms[idx]),
            "method":    "RMS",
            "strength":  float(rms[idx] / thresh_rms),
        })

    # --- Band energy peaks ---
    t_band, energy = band_energy_envelope(samples, sr)
    thresh_band = np.mean(energy) + SIGMA_THRESHOLD * np.std(energy)
    dt_band = float(t_band[1] - t_band[0]) if len(t_band) > 1 else 0.05
    min_gap_frames_band = max(1, int(MIN_GAP_S / dt_band))
    peaks_band, _ = find_peaks(energy, height=thresh_band, distance=min_gap_frames_band)
    for idx in peaks_band:
        events.append({
            "time_s":    float(t_band[idx]),
            "method":    f"Band {BAND_HZ[0]}-{BAND_HZ[1]} Hz",
            "strength":  float(energy[idx] / thresh_band),
        })

    return sorted(events, key=lambda e: e["time_s"])


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def export_audacity_labels(events: list[dict], label_path: str, duration_s: float = 0.5):
    """Write an Audacity label file (tab-separated: start end label)."""
    with open(label_path, "w") as f:
        for ev in events:
            t = ev["time_s"]
            f.write(f"{t:.6f}\t{t + duration_s:.6f}\t{ev['method']} x{ev['strength']:.2f}\n")
    print(f"  Audacity labels → {label_path}")


def print_event_table(all_events: list[dict]):
    """Print a combined table of all detected events across all videos."""
    col = "{:<35} {:<25} {:>12} {:>12} {:>10}"
    sep = "-" * 97
    print("\n" + sep)
    print(col.format("File", "Method", "Local time", "Wall-clock", "Strength"))
    print(sep)
    for ev in sorted(all_events, key=lambda e: e["wall_clock"]):
        local = _fmt_mmss(ev["time_s"])
        wall  = ev["wall_clock"].strftime("%H:%M:%S.%f")[:-3]
        print(col.format(ev["label"][:35], ev["method"], local, wall, f"x{ev['strength']:.2f}"))
    print(sep)


def _fmt_mmss(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Verified label parsing + ground truth plot
# ---------------------------------------------------------------------------

SITE_NAMES = {                       # MIX OF SITES NUMBERS --- CHECK!!
    "20260205_140018000_iOS.MOV":      "S3 — 18000_iOS",
    "20260205_150632000_iOS.MP4":      "S1 — 32000_iOS  (reference)",
    "TimeVideo_20260205_150217~3.mp4": "S2 — TimeVideo",
}

# Video filename → user-convention site number (S1/S2/S3)
FILE_TO_SITE = {
    "20260205_140018000_iOS.MOV":      3,
    "20260205_150632000_iOS.MP4":      1,
    "TimeVideo_20260205_150217~3.mp4": 2,
}

MANUAL_LABEL_FILE = os.path.join(OUTPUT_DIR, "site_event_labels_verified.txt")


def load_manual_labels(path: str) -> dict:
    """
    Parse the manually verified combined label file.
    Returns {site_int: [(t_label_s, arrival_type), ...]}
    Only certain events (no '?') are included.
    """
    cross_re = re.compile(r"S([123])([123])_+(\d+)")
    local_re  = re.compile(r"\bS([123])_(\d+)\b")
    by_site: dict = {1: [], 2: [], 3: []}
    if not os.path.exists(path):
        return by_site
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 2)
            if len(parts) < 3:
                continue
            t, label = float(parts[0]), parts[2]
            if "?" in label:
                continue
            arr = "LAUNCH" if "LAUNCH" in label else "DET"
            if cross_re.search(label):
                continue  # cross-site arrivals handled by load_intersite_labels
            m = local_re.search(label)
            if m:
                by_site[int(m.group(1))].append((t, arr))
    return by_site

def load_intersite_labels(path: str) -> dict:
    """Returns {site_int: [t_label_s, ...]} for certain cross-site arrivals only."""
    cross_re = re.compile(r"S([123])([123])_+(\d+)")
    by_site: dict = {1: [], 2: [], 3: []}
    if not os.path.exists(path):
        return by_site
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 2)
            if len(parts) < 3:
                continue
            t, label = float(parts[0]), parts[2]
            m = cross_re.search(label)
            if m and int(m.group(1)) != int(m.group(2)):
                by_site[int(m.group(1))].append(t)
    return by_site


CATEGORY_STYLE = {
    "S2":        dict(color="#2ca02c", marker="D", label="S2 – local detonation"),
    "DET":       dict(color="#1a6b1a", marker="*", label="DET – launch (manual)"),
    "S1":        dict(color="#1f77b4", marker="o", label="S1 – single spike (inter-site?)"),
    "FAR?":      dict(color="#ff7f0e", marker="^", label="FAR? – far/uncertain"),
    "uncertain": dict(color="#aaaaaa", marker="x", label="? – reflection/uncertain"),
    "confirmed": dict(color="#9467bd", marker="s", label="confirmed (site 1)"),
}


def _label_category(text: str) -> str:
    t = text.strip()
    if "FAR?" in t:
        return "FAR?"
    if t.startswith("DET"):
        return "DET"
    if "S2" in t:
        return "S2"
    if "S1" in t:
        return "S1"
    if t.endswith("?"):
        return "uncertain"
    return "confirmed"


def _label_strength(text: str) -> float:
    m = re.search(r"x([\d.]+)", text)
    return float(m.group(1)) if m else 1.0


def parse_verified_labels(label_path: str) -> list[dict]:
    """Parse an Audacity label file that has been manually verified."""
    events = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            start_s, end_s, label_text = float(parts[0]), float(parts[1]), parts[2]
            events.append(dict(
                start_s=start_s,
                end_s=end_s,
                label_text=label_text,
                category=_label_category(label_text),
                strength=_label_strength(label_text),
            ))
    return events


def plot_ground_truth(records: list[dict], t0: datetime, global_end_s: float):
    """Plot verified detonation events from all sources on a shared timeline."""
    tick_minutes = 1 if global_end_s <= 600 else 2
    t_end = t0 + timedelta(seconds=global_end_s)
    t_plot_start = datetime(t0.year, t0.month, t0.day, 15, 3, 30)
    n = len(records)
    site_names = [SITE_NAMES.get(r["label"], r["label"]) for r in records]

    # Print table
    col = "{:<22} {:<22} {:>12} {:>10}"
    sep = "-" * 70
    print("\n" + sep)
    print(col.format("Site", "Category", "Wall-clock", "Strength"))
    print(sep)
    all_ev = [
        (r, ev)
        for r in records
        for ev in r.get("verified_events", [])
    ]
    for r, ev in sorted(all_ev, key=lambda x: x[1]["start_s"] + (x[0]["start"] - t0).total_seconds()):
        wall = (r["start"] + timedelta(seconds=ev["start_s"])).strftime("%H:%M:%S.%f")[:-3]
        site = SITE_NAMES.get(r["label"], r["label"])
        print(col.format(site[:22], ev["category"], wall, f"x{ev['strength']:.2f}"))
    print(sep)

    # Plot
    _, ax = plt.subplots(figsize=(16, max(4, n * 1.5)))
    plotted = set()

    for row_idx, r in enumerate(records):
        for ev in r.get("verified_events", []):
            wall = r["start"] + timedelta(seconds=ev["start_s"])
            cat = ev["category"]
            style = CATEGORY_STYLE.get(cat, dict(color="black", marker="o", label=cat))
            size = max(60, ev["strength"] * 40)
            kwargs = dict(color=style["color"], marker=style["marker"],
                          s=size, zorder=5, edgecolors="white", linewidths=0.5)
            if cat not in plotted:
                kwargs["label"] = style["label"]
                plotted.add(cat)
            ax.scatter(wall, row_idx, **kwargs)
            ax.axvline(wall, color=style["color"], lw=0.5, alpha=0.2, zorder=1)

    ax.set_yticks(range(n))
    ax.set_yticklabels(site_names, fontsize=9)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlim(t_plot_start, t_end)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=tick_minutes))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax.set_xlabel(t0.strftime("%Y-%m-%d"), fontsize=10)
    ax.set_title("Verified Detonation Events — Ground Truth", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="x", lw=0.5, alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "ground_truth_events.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Ground truth plot saved to {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    video_files = sorted(
        os.path.join(VIDEO_DIR, f)
        for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith((".mov", ".mp4"))
    )

    if not video_files:
        raise FileNotFoundError(f"No video files found in {VIDEO_DIR}")

    print(f"Found {len(video_files)} videos:")
    for v in video_files:
        print(f"  {os.path.basename(v)}")

    records = []
    for path in video_files:
        print(f"\nExtracting audio: {os.path.basename(path)} ...")
        basename = os.path.basename(path)
        samples, sr = extract_audio(path)
        if basename in MANUAL_FRAME_REFS:
            frame_idx, frame_time = MANUAL_FRAME_REFS[basename]
            clip_fps = VideoFileClip(path).fps
            start_dt = frame_time - timedelta(seconds=frame_idx / clip_fps)
        else:
            start_dt = parse_timestamp(path)
        duration_s = len(samples) / sr
        records.append(dict(
            path=path,
            label=basename,
            start=start_dt,
            samples=samples,
            sr=sr,
            duration=duration_s,
        ))
        print(f"  start={start_dt}  duration={duration_s:.2f}s  sr={sr}Hz")

    # Common time axis
    t0 = min(r["start"] for r in records)
    for r in records:
        r["offset_s"] = (r["start"] - t0).total_seconds()
    global_end_s = max(r["offset_s"] + r["duration"] for r in records)
    t_end = t0 + timedelta(seconds=global_end_s)
    tick_minutes = 1 if global_end_s <= 600 else 2

    # --- Detection ---
    all_events = []
    for r in records:
        print(f"\nDetecting events in {r['label']} ...")
        events = detect_events(r["samples"], r["sr"])
        print(f"  {len(events)} candidate(s) found")

        # Attach global wall-clock time and video label
        for ev in events:
            ev["label"] = r["label"]
            ev["wall_clock"] = r["start"] + timedelta(seconds=ev["time_s"])

        # Export Audacity label file for this video
        stem = os.path.splitext(r["label"])[0]
        label_path = os.path.join(OUTPUT_DIR, f"{stem}_labels.txt")
        export_audacity_labels(events, label_path)

        all_events.extend(events)
        r["events"] = events

    print_event_table(all_events)

    # --- Load verified labels and plot ground truth ---
    has_verified = False
    for r in records:
        stem = os.path.splitext(r["label"])[0]
        verified_path = os.path.join(OUTPUT_DIR, f"{stem}_labels_verified.txt")
        if os.path.exists(verified_path):
            r["verified_events"] = parse_verified_labels(verified_path)
            print(f"  Loaded {len(r['verified_events'])} verified events for {r['label']}")
            has_verified = True
        else:
            r["verified_events"] = []

    if has_verified:
        plot_ground_truth(records, t0, global_end_s)

    # --- Waveform plot ---
    t_plot_start = datetime(t0.year, t0.month, t0.day, 15, 3, 30)
    fig, axes = plt.subplots(len(records), 1, figsize=(14, 3 * len(records)), sharex=True)
    if len(records) == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    marker_styles = {"RMS": ("v", "red"), f"Band {BAND_HZ[0]}-{BAND_HZ[1]} Hz": ("^", "#00BCD4")}

    for idx, (ax, r, color) in enumerate(zip(axes, records, colors)):
        t_local = np.arange(len(r["samples"])) / r["sr"]
        t_datetime = [t0 + timedelta(seconds=r["offset_s"] + float(t)) for t in t_local]

        ax.plot(t_datetime, r["samples"], lw=0.4, color=color, alpha=0.8)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_title(r["label"], fontsize=9, loc="left")
        ax.text(0.5, 1.0, f"Site {idx + 1}", transform=ax.transAxes,
                fontsize=12, fontweight="bold", ha="center", va="bottom")
        ax.axvline(r["start"], color=color, lw=1, ls="--", alpha=0.5)
        ax.set_xlim(t_plot_start, t_end)

        # Mark detected events
        for ev in r["events"]:
            marker, mcol = marker_styles.get(ev["method"], ("o", "gray"))
            wall = r["start"] + timedelta(seconds=ev["time_s"])
            ax.axvline(wall, color=mcol, lw=1, ls=":", alpha=0.7)
            ax.plot(wall, 0, marker=marker, color=mcol, ms=8, zorder=5,
                    label=ev["method"])

        # ── overlay manually verified true events ──────────────────────────
        manual_labels = load_manual_labels(MANUAL_LABEL_FILE)
        site_num = FILE_TO_SITE.get(r["label"])
        if site_num:
            launch_added = det_added = False
            for t_lbl, arr in manual_labels.get(site_num, []):
                wall = t0 + timedelta(seconds=t_lbl)
                if arr == "LAUNCH":
                    col_m, lbl_m = "#8E44AD", "True LAUNCH"
                    already = launch_added
                    launch_added = True
                else:
                    col_m, lbl_m = "#C0392B", "True DET"
                    already = det_added
                    det_added = True
                ax.axvline(wall, color=col_m, lw=2.0, ls="-", alpha=0.9,
                           zorder=6, label=None if already else lbl_m)

        # ── overlay inter-site arrivals (light, single legend entry) ──────
        intersite_labels = load_intersite_labels(MANUAL_LABEL_FILE)
        if site_num:
            intersite_added = False
            for t_lbl in intersite_labels.get(site_num, []):
                wall = t0 + timedelta(seconds=t_lbl)
                ax.axvline(wall, color="#555555", lw=1.0, ls="-", alpha=0.4,
                           zorder=4,
                           label=None if intersite_added else "Inter-site detonations")
                intersite_added = True

        # Deduplicate legend entries
        handles, labels_leg = ax.get_legend_handles_labels()
        seen = {}
        for h, lbl in zip(handles, labels_leg):
            seen.setdefault(lbl, h)
        if seen:
            ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper left")

        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=tick_minutes))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)

    axes[-1].set_xlabel(t0.strftime("%Y-%m-%d"), fontsize=10)
    fig.suptitle("Ground Truth Extraction and Verified Detonation Events", fontsize=12, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "ground_truth_extraction.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
