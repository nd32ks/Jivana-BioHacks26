"""Download JIVANA's data on first run (for cloud deployment).

Downloads, from the public LILA Azure mirror:
  1. The SWG Camera Traps COCO metadata JSON (~59 MB zipped, ~1 GB
     unzipped) — required for all analysis pages.
  2. A small sample of images for the triage/live demos (~60 images,
     ~100 MB). Skipped if SKIP_IMAGES=1.

Only files that are missing are downloaded. Safe to re-run.
"""

import os
import subprocess
import sys
from pathlib import Path

import config

METADATA_URL = ("https://lilawildlife.blob.core.windows.net/lila-wildlife/"
                "swg-camera-traps/swg_camera_traps.zip")
# ~60 images: enough for the triage demo and the live page, small enough
# for a free cloud instance to fetch at startup.
SAMPLE_MANIFEST_URL = ""  # fill in if you host one; empty -> skip images


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_metadata() -> bool:
    """Return True if the COCO JSON is present (or was just downloaded)."""
    if config.COCO_JSON_PATH.exists():
        return True
    zip_path = config.DATA_DIR / "swg_camera_traps.zip"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print("Downloading SWG metadata (59 MB)...")
        _run(["curl", "-sfL", "-o", str(zip_path), METADATA_URL])
    print("Unzipping...")
    _run(["unzip", "-o", "-q", str(zip_path), "-d", str(config.DATA_DIR)])
    # The archive extracts to swg_camera_traps.1.1.json; normalise the name.
    for p in config.DATA_DIR.glob("swg_camera_traps*.json"):
        if p.name != config.COCO_JSON_PATH.name:
            p.rename(config.COCO_JSON_PATH)
        break
    zip_path.unlink(missing_ok=True)
    return config.COCO_JSON_PATH.exists()


def main() -> int:
    if not ensure_metadata():
        print("ERROR: metadata download failed.")
        return 1
    if os.environ.get("SKIP_IMAGES"):
        print("SKIP_IMAGES set — image demos will show 'no images' messages.")
    print("Data ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
