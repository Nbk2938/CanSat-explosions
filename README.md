# CanSat-explosions

Analysis of acoustic recordings from a CanSat detonation experiment.
Three microphone sites recorded the same series of pyrotechnic events.
The goal is to reconstruct event timing, source localisation, and
inter-site propagation delays from the raw audio.

---

## Experiment setup

| Site | Recording | GPS coordinates |
|------|-----------|-----------------|
| S1 (reference) | 32000_iOS | 47.40601 N, 8.63522 E |
| S2 | TimeVideo | 47.40453 N, 8.64424 E |
| S3 | 18000_iOS | 47.39651 N, 8.64827 E |

Inter-site distances (Haversine):

| Pair | Distance |
|------|----------|
| S1 – S2 | ≈ 698 m |
| S1 – S3 | ≈ 1443 m |
| S2 – S3 | ≈ 943 m |

Each CanSat event produces **two acoustic spikes** at the local microphone:
1. **LAUNCH** – low-energy propulsion spike at ground level
2. **DET** – high-energy detonation spike from altitude

---

## What was done

### 1. Ground truth extraction (`ground_truth_extraction.py`)
Raw audio was processed to detect energy spikes using two detectors run in
parallel: an RMS detector and a bandpass (20–2000 Hz) detector. Frame
references were used to synchronise audio to video timestamps.

### 2. Manual event labelling
All detected events were manually labelled in Audacity across all three sites
on a shared timeline. Label format:

```
<method> <micSite><sourceSite>_<eventNum> [LAUNCH]
```

Unknown/inter-site events were tagged `S{mic}_FAR?`.  
Labels are in `outputs/site_event_labels_moast_probable.txt`.

### 3. Audio timeline alignment

The three recordings started at different wall-clock times and were placed on
the shared Audacity timeline manually. Two reference events (S3_4 and S3_9)
were identified with high certainty at all three sites and used to compute
GPS-based timing offsets:

| Site | Offset applied |
|------|---------------|
| S1 (32000_iOS) | 0.000 s (reference) |
| S2 (TimeVideo) | +0.803 s |
| S3 (18000_iOS) | +2.428 s |

Residual after correction: ±0.075 s (≈ ±26 m), within GPS accuracy.
GPS-calibrated label file: `outputs/site_event_labels_gps_calibrated.txt`
(also exported as per-site label files under `outputs/Site N - *.txt`).

**Note on GPS vs acoustic distances:** raw measured delays before calibration
were ~37 % longer than GPS distances imply. The distance *ratios* matched to
within 1 %, confirming the GPS geometry is correct. The systematic offset
was caused by the recordings not being synchronised to a common clock before
the GPS correction was applied.

### 4. Calibrated acoustic delays

From the two reference events, after GPS alignment:

| Direction | LAUNCH delay | DET delay |
|-----------|-------------|-----------|
| S3 → S1 | 4.204 ± 0.075 s | 2.948 ± 0.5 s |
| S3 → S2 | 2.746 ± 0.075 s | 1.699 ± 0.5 s |
| S2 → S3 | 2.746 ± 0.075 s | 1.699 ± 0.5 s |
| S2 → S3 (reflected) | 3.552 ± 0.041 s | — |
| S1 ↔ S2 | 2.035 ± 0.15 s (GPS-derived) | — |

A reflected LAUNCH arrival from S2 to S3 was identified empirically:
6 event pairs at S3 always had a second spike ~0.81 s after the direct arrival,
consistent with a ~277 m longer reflected path.

### 5. Inter-site event matching (`inter_site_labeling.py`)

A Bayesian confidence model matched each `FAR?` event to its most probable
source using the calibrated delays above:

```
P(match | Δt) = N(Δt; μ, σ) / (N(Δt; μ, σ) + ρ_bg)
```

where `ρ_bg` is the local event density [events/s] acting as the background
noise hypothesis. Threshold: 75 % confidence.

Key design choices:
- All delay modes (LAUNCH, DET, REFL) are tried for every source event
  regardless of how it was labelled at the source mic, because the source mic
  sometimes misses the LAUNCH spike while the far mic still detects it.
- Self-calibration for S1 ↔ S2 (no reference events) was attempted via a
  sliding-window vote on pairwise Δt histograms but failed — too few data points.

Result (GPS-calibrated labels): **14 / 41 FAR? events matched** at ≥ 75 %
confidence. The remaining 27 are unresolved, mostly S3_FAR? events whose
source could not be confirmed with the available calibration.

### 6. Arrival-time consistency check (`arrival_time_comparison.py`)

For each site pair, measured travel times in both directions were plotted
against the calibrated reference lines. Key findings:

- **S3 → S1** LAUNCH: mean 6.35 s (cal. 6.63 s), 4 events, consistent.
- **S3 → S2** LAUNCH: mean 3.92 s, systematically ~0.45 s below calibrated —
  suggests a residual ~0.3 s alignment offset between S2 and S3.
- **S2 → S3** DET (matched as LAUNCH arrivals): mean 4.57 s, ~0.2 s above
  calibrated — consistent with the same residual offset seen from the other
  direction.
- **S1 → S3**: only 3 uncertain events, one outlier at 4.2 s (likely
  mis-labelled). Too few points to draw conclusions.
- **S1 ↔ S2**: single event (S21_4?), inconclusive.

### 7. Event timeline (`event_timeline.py`)

Simple timeline plot of all *certain* events per site (uncertain FAR? events
excluded). Output: `outputs/event_timeline.png`.

---

## Open questions / next steps

- The S2–S3 residual offset (~0.3 s) was not fully resolved; the GPS
  uncertainty for S2 (±50 m) can explain at most ±0.15 s per direction.
- S1 ↔ S2 delays remain uncalibrated — no reference event was identified
  with certainty at both S1 and S2.
- Many S3_FAR? events could not be matched. They may be reflections within
  S3's local environment or events not labelled in S1/S2.
- Detonation height per event was not estimated; this would require
  triangulation from at least two calibrated inter-site DET delays.

---

## File overview

```
outputs/
  site_event_labels_moast_probable.txt   — manually labelled combined timeline
  site_event_labels_gps_calibrated.txt   — after GPS timing correction
  site-event-labels-matched.txt          — after algorithmic FAR? matching
  Site {1,2,3} - *.txt                   — per-site Audacity label files
  event_timeline.png                     — timeline of certain events
  arrival_time_comparison.png            — travel-time consistency plot
  aligned_audio_waveforms.png            — waveform overview

ground_truth_extraction.py   — audio processing pipeline
event_timeline.py            — timeline visualisation
```
