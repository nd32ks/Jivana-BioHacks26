"""Load and tidy the camera-trap dataset.

Parses a COCO Camera Traps JSON file (the format LILA BC distributes) into a
tidy pandas DataFrame with one row per image:

    image_id, location, datetime, category

where category is a species name, "HUMAN", or "EMPTY".

Honesty notes:
- Missing/invalid timestamps are DROPPED and the count is reported, never
  silently imputed.
- Independence: repeated detections of the same category at the same location
  within INDEPENDENCE_WINDOW_MINUTES are merged into one event. Only
  independent events are used for analysis; raw image counts are still shown
  on the Overview page for context.
"""

import json
from pathlib import Path

import pandas as pd

import config


class DatasetMissingError(FileNotFoundError):
    """Raised when the COCO JSON (or sample images) cannot be found."""


def _missing_message(path: Path) -> str:
    return (
        f"Dataset file not found: {path}\n\n"
        f"JIVANA is configured for: {config.DATASET_NAME}\n"
        "Download the COCO Camera Traps metadata JSON from LILA BC "
        "(https://lila.science) — you must agree to the dataset terms — "
        f"and place it at:\n    {path}\n"
        "Only the metadata JSON is needed for the analysis. For the triage "
        f"demo, also download ~200 sample images into:\n    {config.SAMPLE_IMAGE_DIR}"
    )


def load_coco(path: Path = config.COCO_JSON_PATH) -> pd.DataFrame:
    """Parse the COCO Camera Traps JSON into a tidy DataFrame."""
    if not Path(path).exists():
        raise DatasetMissingError(_missing_message(Path(path)))

    with open(path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # --- category lookup: id -> name -----------------------------------
    cat_id_to_name = {}
    for c in coco.get("categories", []):
        # COCO Camera Traps uses either 'name' or 'common_name'
        name = c.get("common_name") or c.get("name") or str(c.get("id"))
        cat_id_to_name[c["id"]] = str(name).strip()

    # --- images: id -> metadata -----------------------------------------
    images = {}
    for img in coco.get("images", []):
        images[img["id"]] = img

    # --- annotations: image_id -> list of category names -----------------
    # One image can carry several annotations; we keep every category found
    # (an image with both a person and a deer yields one row each), plus an
    # EMPTY row for images with no animal annotation.
    anns_by_image: dict = {}
    for a in coco.get("annotations", []):
        anns_by_image.setdefault(a["image_id"], []).append(a)

    rows = []
    for img_id, img in images.items():
        # Location: prefer explicit 'location' field, fall back to
        # directory-style ids or 'camera' metadata. Never invent GPS.
        loc = (
            img.get("location")
            or img.get("camera")
            or (str(img.get("file_name", "")).split("/")[0] or "unknown")
        )
        dt_raw = img.get("datetime") or img.get("date_captured") or ""
        cats = []
        for a in anns_by_image.get(img_id, []):
            cats.append(cat_id_to_name.get(a.get("category_id"), "unknown"))
        if not cats:
            cats = ["empty"]
        for name in set(cats):  # de-duplicate repeated labels on one image
            rows.append(
                {
                    "image_id": img_id,
                    "location": str(loc),
                    "datetime_raw": str(dt_raw),
                    "category_raw": name,
                }
            )

    df = pd.DataFrame(rows)

    # --- timestamps: parse, drop invalid, REPORT -------------------------
    df["datetime"] = pd.to_datetime(df["datetime_raw"], errors="coerce")
    n_bad = int(df["datetime"].isna().sum())
    if n_bad:
        print(f"[load_data] Dropped {n_bad} rows with missing/invalid timestamps.")
    df = df.dropna(subset=["datetime"]).drop(columns=["datetime_raw"])

    # --- normalise category ---------------------------------------------
    def normalise(name: str) -> str:
        low = name.strip().lower()
        if low in config.HUMAN_CATEGORIES:
            return "HUMAN"
        if low in config.EMPTY_CATEGORIES or low == "":
            return "EMPTY"
        if low in config.EXCLUDE_CATEGORIES:
            return None          # QC label: drop entirely
        # turn "eurasian_wild_pig" into "Eurasian wild pig" for display
        return name.strip().replace("_", " ").title()

    df["category"] = df["category_raw"].map(normalise)
    df = df[df["category"].notna()].drop(columns=["category_raw"])

    # --- independence window --------------------------------------------
    n_before = len(df)
    df = df.sort_values(["location", "category", "datetime"])
    gap = df.groupby(["location", "category"])["datetime"].diff()
    keep = gap.isna() | (
        gap.dt.total_seconds() > config.INDEPENDENCE_WINDOW_MINUTES * 60
    )
    df = df[keep].copy()
    df["independent"] = True
    n_merged = n_before - len(df)
    print(
        f"[load_data] {n_before} category-rows -> {len(df)} independent events "
        f"(merged {n_merged} records within "
        f"{config.INDEPENDENCE_WINDOW_MINUTES} min)."
    )
    return df.reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    """Numbers for the Overview page. All computed from the loaded data."""
    species = sorted(c for c in df["category"].unique() if c not in ("HUMAN", "EMPTY"))
    return {
        "n_rows": int(len(df)),
        "n_locations": int(df["location"].nunique()),
        "n_species": int(len(species)),
        "species": species,
        "date_min": df["datetime"].min(),
        "date_max": df["datetime"].max(),
        "n_human_events": int((df["category"] == "HUMAN").sum()),
        "n_empty": int((df["category"] == "EMPTY").sum()),
    }


if __name__ == "__main__":
    d = load_coco()
    print(dataset_summary(d))
