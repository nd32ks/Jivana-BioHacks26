"""JIVANA configuration.

All user-tunable settings live here. Paths are relative to the project root
(the folder that contains app.py). See README.md for what to download and
where to put it.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths ----
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_IMAGE_DIR = DATA_DIR / "sample_images"       # ~200 images for triage demo
TRIAGE_RESULTS_PATH = DATA_DIR / "triage_results.json"
HEALTH_DB_PATH = DATA_DIR / "health_tags.db"

# ------------------------------------------------------------- dataset -----
# Recommended: the SWG Camera Traps dataset (Vietnam / Laos) on LILA BC
# (https://lila.science/datasets/swg-camera-traps). It has a "person"
# category, timestamps, and well over 20 camera locations. If you cannot use
# it, set DATASET_NAME and COCO_JSON_PATH to any LILA dataset that has
# (a) a human category, (b) timestamps, (c) >= 20 locations, and add the
# human category names it uses to HUMAN_CATEGORIES below.
DATASET_NAME = "SWG Camera Traps (Vietnam/Laos) - LILA BC"
COCO_JSON_PATH = DATA_DIR / "swg_camera_traps.json"  # COCO Camera Traps format

# Category names (case-insensitive) that count as HUMAN in this dataset.
# "person" covers SWG; add "human" etc. if you switch datasets.
HUMAN_CATEGORIES = {"person", "human"}

# Category names that count as EMPTY frames (no animal / nothing).
EMPTY_CATEGORIES = {"empty", "none", "background", "unlabeled"}

# QC / non-species labels to exclude from the species analysis entirely.
EXCLUDE_CATEGORIES = {"blurred", "ignore", "problem"}

# --------------------------------------------------------- study design ----
# Two detections of the same species at the same location within this window
# count as ONE independent event (standard camera-trap practice; default 30
# minutes, configurable here).
INDEPENDENCE_WINDOW_MINUTES = 30

# Minimum independent detections a species needs IN EACH disturbance group
# before we estimate its activity shift. Below this: "insufficient data".
MIN_DETECTIONS_PER_GROUP = 50

# Disturbance split: locations ranked by human detection rate are split into
# terciles. "high" = top TERCILE_HIGH fraction, "low" = bottom TERCILE_LOW.
# Middle locations are excluded from the shift comparison.
TERCILE_HIGH = 1.0 / 3.0
TERCILE_LOW = 1.0 / 3.0

# ------------------------------------------------------- activity stats ----
BOOTSTRAP_ITERATIONS = 1000     # bootstrap replicates for every CI we show
RANDOM_SEED = 42                # fixed so results are reproducible
# Ridout & Linkie (2009): use Delta1 when the smaller group sample < 75,
# otherwise Delta4 (the more robust overlap estimator).
DELTA1_MAX_SMALLER_N = 75

# -------------------------------------------------------------- triage -----
# MegaDetector confidence threshold for keeping a detection (default 0.2 as
# recommended by the MegaDetector authors for general use).
MEGADETECTOR_THRESHOLD = 0.2
MEGADETECTOR_MODEL = "MDV5A"    # MegaDetector v5, "A" general model
