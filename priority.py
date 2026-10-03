"""Patrol-priority ranking of camera locations.

Priority per location = z(human detection rate) + z(local nocturnality shift),
where the shift is the average, across species with enough data AT THAT
LOCATION, of (nocturnality at this location - nocturnality in the low-
disturbance group). Both components are shown so the ranking is explainable.
"""

import numpy as np
import pandas as pd

import activity


def _z(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    return (s - s.mean()) / std if std > 0 else s * 0.0


def local_nocturnality_shift(df, species_with_data) -> pd.Series:
    """Per location: mean nocturnality shift vs the low-disturbance baseline."""
    night = ((df["datetime"].dt.hour >= 18) | (df["datetime"].dt.hour < 6))
    df = df.copy()
    df["night"] = night
    low_night = df[(df["disturbance"] == "low") & df["category"].isin(species_with_data)]
    baseline = low_night["night"].mean() if len(low_night) else np.nan
    rows = []
    for loc, g in df[df["category"].isin(species_with_data)].groupby("location"):
        rows.append({"location": loc, "local_night_share": g["night"].mean()})
    s = pd.DataFrame(rows).set_index("location")["local_night_share"]
    return s - baseline


def patrol_priority(df, rates: pd.DataFrame, species_with_data) -> pd.DataFrame:
    """Ranked, explainable patrol-priority table."""
    out = rates.copy()
    shift = local_nocturnality_shift(df, species_with_data)
    out["local_night_shift"] = out["location"].map(shift).fillna(0.0)
    out["z_human"] = _z(out["human_rate"])
    out["z_shift"] = _z(out["local_night_shift"])
    out["priority"] = out["z_human"] + out["z_shift"]
    out = out.sort_values("priority", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1

    def reasons(r):
        parts = []
        if r["z_human"] > 0:
            parts.append(f"elevated human activity ({r['human_events']} events in "
                         f"~{r['camera_days']} camera-days)")
        else:
            parts.append("human activity at or below average")
        if r["local_night_shift"] > 0:
            parts.append(f"wildlife more nocturnal than baseline "
                         f"(+{r['local_night_shift']:.0%} of detections at night)")
        else:
            parts.append("no clear local nocturnality shift")
        return "; ".join(parts)

    out["reasons"] = out.apply(reasons, axis=1)
    return out[["rank", "location", "human_events", "camera_days", "human_rate",
                "local_night_shift", "priority", "reasons"]]
