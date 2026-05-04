#!/usr/bin/env python3
"""
Full analysis pipeline — runs all three steps in order:

  Step 1  ground_truth_extraction.py
          Extract audio from video files, run RMS + band-energy detectors,
          overlay manually verified labels, produce waveform plot.

  Step 2  cansat_event_extraction.py
          Extract acoustic event detections from OBAMA (Excel telemetry)
          and ScamSat (binary FFT file), produce label files and plots.

  Step 3  cansat_alignment.py
          Align CanSat timelines to the ground-truth microphone timeline,
          compute detection metrics, produce alignment and event plots.

Usage:
  python run_pipeline.py              # run all steps
  python run_pipeline.py --skip 1    # skip step 1 (slow video extraction)
"""

import sys
import time

import ground_truth_extraction
import cansat_event_extraction
import cansat_alignment

STEPS = {
    1: ("Ground-truth extraction",    ground_truth_extraction.main),
    2: ("CanSat event extraction",    cansat_event_extraction.main),
    3: ("Timeline alignment & metrics", cansat_alignment.main),
}


def run(skip: set[int]):
    total_start = time.time()
    for n, (title, fn) in STEPS.items():
        if n in skip:
            print(f"\n{'─' * 60}")
            print(f"  Step {n}: {title}  [SKIPPED]")
            print(f"{'─' * 60}")
            continue

        print(f"\n{'═' * 60}")
        print(f"  Step {n} / {len(STEPS)}: {title}")
        print(f"{'═' * 60}\n")
        t0 = time.time()
        fn()
        print(f"\n  ✓ Done in {time.time() - t0:.1f} s")

    print(f"\n{'═' * 60}")
    print(f"  Pipeline complete — total {time.time() - total_start:.1f} s")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    skip_steps: set[int] = set()
    args = sys.argv[1:]
    if "--skip" in args:
        idx = args.index("--skip")
        for s in args[idx + 1:]:
            if s.isdigit():
                skip_steps.add(int(s))
            else:
                break
    run(skip_steps)
