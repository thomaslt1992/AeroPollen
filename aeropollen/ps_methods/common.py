from typing import Any

import numpy as np
import pandas as pd

RESULT_COLUMNS = [
    "pollen_type",
    "season",
    "ps_start",
    "ps_end",
    "peak_date",
    "ps_start_doy",
    "ps_end_doy",
    "peak_doy",
    "ps_length",
    "pollen_season_integral",
    "total_pollen_integral",
    "day_threshold",
    "days_above_threshold",
]


def empty_result(
    pollen_type: str,
    season: int,
    *,
    day_threshold: float,
    peak_date: Any = pd.NaT,
    total_integral: float = np.nan,
    days_above_threshold: int = 0,
) -> dict[str, Any]:
    return {
        "pollen_type": pollen_type,
        "season": int(season),
        "ps_start": pd.NaT,
        "ps_end": pd.NaT,
        "peak_date": peak_date,
        "ps_start_doy": np.nan,
        "ps_end_doy": np.nan,
        "peak_doy": peak_date.dayofyear if pd.notna(peak_date) else np.nan,
        "ps_length": np.nan,
        "pollen_season_integral": np.nan,
        "total_pollen_integral": total_integral,
        "day_threshold": float(day_threshold),
        "days_above_threshold": int(days_above_threshold),
    }


def format_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(results, columns=RESULT_COLUMNS)
    for column in (
        "ps_start_doy",
        "ps_end_doy",
        "peak_doy",
        "ps_length",
        "days_above_threshold",
    ):
        frame[column] = frame[column].astype("Int64")
    for column in ("ps_start", "ps_end", "peak_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    return frame
