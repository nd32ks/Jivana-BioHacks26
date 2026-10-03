"""Disturbance proxy per camera location.

Disturbance here is a PROXY: the rate of independent human detections per
camera-day. It is not a direct measure of hunting pressure or habitat change.

Camera-days are APPROXIMATED as (last image date - first image date) per
location. Gaps in deployment are not visible in the metadata, so rates during
active periods may be over- or under-stated; we say this plainly in the UI.
"""

import numpy as np
import pandas as pd

import config


def human_rate_per_location(df: pd.DataFrame) -> pd.DataFrame:
    """Independent human events / camera-days for each location."""
    rows = []
    for loc, g in df.groupby("location"):
        days = max((g["datetime"].max() - g["datetime"].min()).days, 1)
        n_human = int((g["category"] == "HUMAN").sum())
        rows.append(
            {"location": loc, "camera_days": days, "human_events": n_human,
             "human_rate": n_human / days}
        )
    return pd.DataFrame(rows).sort_values("human_rate", ascending=False).reset_index(drop=True)


def split_locations(rates: pd.DataFrame) -> dict:
    """Split locations into high / middle / low disturbance terciles by human rate."""
    n = len(rates)
    n_high = max(1, int(round(n * config.TERCILE_HIGH)))
    n_low = max(1, int(round(n * config.TERCILE_LOW)))
    ranked = rates.sort_values("human_rate", ascending=False)
    high = set(ranked.head(n_high)["location"])
    low = set(ranked.tail(n_low)["location"])
    return {"high": high, "low": low,
            "high_rate_min": float(ranked.iloc[n_high - 1]["human_rate"]),
            "low_rate_max": float(ranked.iloc[-n_low]["human_rate"]),
            "n_zero_rate": int((rates["human_rate"] == 0).sum()),
            "n_locations": n}


def assign_groups(df: pd.DataFrame, split: dict) -> pd.DataFrame:
    df = df.copy()
    df["disturbance"] = np.where(
        df["location"].isin(split["high"]), "high",
        np.where(df["location"].isin(split["low"]), "low", "middle"))
    return df
