"""Health-issue tagging for JIVANA (SQLite).

Biologists manually tag suspected health issues on camera-trap images. JIVANA
does NOT auto-diagnose health; these tags build a labelled dataset for a
FUTURE model. Tags: injury / snare / emaciation / lesion / other / none.
"""

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import config

VALID_TAGS = {"injury", "snare", "emaciation", "lesion", "other", "none"}


def _connect(db_path: Path = config.HEALTH_DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS health_tags(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL,
            location TEXT,
            species TEXT,
            tag TEXT NOT NULL CHECK(tag IN
                ('injury','snare','emaciation','lesion','other','none')),
            notes TEXT DEFAULT '',
            reviewer TEXT DEFAULT '',
            created_at TEXT NOT NULL)"""
    )
    return con


def add_tag(image_id: str, tag: str, location: str = "", species: str = "",
            notes: str = "", reviewer: str = "",
            db_path: Path = config.HEALTH_DB_PATH) -> None:
    if tag not in VALID_TAGS:
        raise ValueError(f"tag must be one of {sorted(VALID_TAGS)}")
    con = _connect(db_path)
    con.execute(
        "INSERT INTO health_tags(image_id,location,species,tag,notes,reviewer,created_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (image_id, location, species, tag, notes, reviewer,
         datetime.now(timezone.utc).isoformat()))
    con.commit()
    con.close()


def get_tags(db_path: Path = config.HEALTH_DB_PATH):
    import pandas as pd
    con = _connect(db_path)
    df = pd.read_sql("SELECT * FROM health_tags ORDER BY id DESC", con)
    con.close()
    return df


def export_csv(out_path: Path = config.DATA_DIR / "health_tags_export.csv",
               db_path: Path = config.HEALTH_DB_PATH) -> Path:
    """Export all tags as CSV — this is the training data for a future health model."""
    df = get_tags(db_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return Path(out_path)
