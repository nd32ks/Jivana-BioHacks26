# JIVANA

Early-warning dashboard that turns camera-trap timestamps into a disturbance
signal: when human activity rises, many species shift activity toward night,
and JIVANA detects that shift **before** populations decline.

## What to download (LILA BC)

1. Go to https://lila.science and agree to the dataset terms.
   Recommended: **SWG Camera Traps (Vietnam/Laos)** — it has a "person"
   category, timestamps, and many camera locations. Any dataset with a human
   category, timestamps, and ≥20 locations works; adjust `DATASET_NAME`,
   `COCO_JSON_PATH`, and `HUMAN_CATEGORIES` in `config.py`.
2. Download the **COCO Camera Traps metadata JSON** only (not the images)
   and place it at:
   `data/swg_camera_traps.json`
3. For the triage demo, download ~200 sample images (a mix of empty and
   non-empty frames) into `data/sample_images/`.

GPS coordinates are withheld by the data provider. JIVANA never invents
them — locations appear as a ranked table.

## Install & run order

```bash
python3.13 -m venv .venv && source .venv/bin/activate   # any Python 3.10+
pip install -r requirements.txt
python activity.py        # sanity checks for the overlap stats (~10 s)
python occupancy.py       # sanity check: known psi/p recovered (~5 s)
python triage.py          # MegaDetector v5 on the ~200 sample images
streamlit run app.py
```

## What we actually downloaded (verified 2026-10-03)

- Metadata: `https://lilawildlife.blob.core.windows.net/lila-wildlife/swg-camera-traps/swg_camera_traps.zip`
  (1 GB unzipped → `data/swg_camera_traps.json`, 2.04 M images, 982 locations).
- 186 sample images (~270 MB) into `data/sample_images/`, manifest in
  `data/sample_manifest.json`.
- Note: the provider removed the human images for privacy (labels remain),
  so most locations have zero human events — see the honest-data note on the
  Activity-shift page.

Expected CPU runtimes (measured on an Apple-silicon laptop):
- `python activity.py` / `occupancy.py`: seconds.
- `python triage.py`: ~3 min for 186 images incl. model download (first run
  downloads the ~280 MB MDv5 weights).
- First `streamlit run app.py` load: ~3 min (1 GB JSON parse + bootstrap CIs,
  then cached). Page-to-page navigation is fast afterwards.

## 90-second demo script (Activity-shift page)

1. **Overview (15 s):** "X independent events, Y locations, Z species. Events
   merge repeats within 30 minutes; MegaDetector triaged ~N% of the sample
   as empty with precision/recall shown."
2. **Activity shift (45 s):** pick a well-sampled species (top of the
   table). Point at the two curves: "At low-disturbance cameras this species
   is active by day; at high-disturbance cameras the mass moves after 18:00.
   Δ = 0.62, 95% CI 0.55–0.68 — meaningfully less overlap. Nocturnality rises
   from 31% to 67%, and the CI on the difference excludes zero, so it is
   flagged 'shifting toward night'. This is clock time, not sun time."
3. **Patrol priority (20 s):** "Locations ranked by standardised human rate
   plus standardised local nocturnality shift — both columns shown, so the
   ranking is explainable. Below: occupancy (probability a site is used,
   accounting for imperfect detection — not abundance) per disturbance group."
4. Close with the sidebar **About** box: proxy measure, no real-time claims,
   manual health tagging only.

## Honesty commitments

- Occupancy = probability a site is used, corrected for imperfect detection.
  Never called abundance.
- Disturbance is a proxy: independent human detections per camera-day.
- Camera-days are approximated from first-to-last image date.
- Nocturnality uses clock time, not sun time.
- Data update after each data collection (SD cards), never real-time.
- Every model output carries a 95% CI (bootstrap or Hessian).
- Health tags are manual; JIVANA does not auto-diagnose.
