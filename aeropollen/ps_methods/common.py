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
    "peak_concentration",
    "days_from_season_start_to_peak",
    "pollen_integral_through_peak",
    "days_through_season_end",
    "pollen_integral_after_peak",
]


def empty_result(
    pollen_type: str,
    season: int,
    *,
    day_threshold: float,
    peak_date: Any = pd.NaT,
    peak_concentration: float = np.nan,
    total_integral: float = np.nan,
    days_above_threshold: int = 0,
) -> dict[str, Any]:
    return {
        "pollen_type": pollen_type,
        "season": int(season),
        "ps_start": pd.NaT,
        "ps_end": pd.NaT,
        "peak_date": peak_date,
        "peak_concentration": peak_concentration,
        "ps_start_doy": np.nan,
        "ps_end_doy": np.nan,
        "peak_doy": peak_date.dayofyear if pd.notna(peak_date) else np.nan,
        "ps_length": np.nan,
        "pollen_season_integral": np.nan,
        "total_pollen_integral": total_integral,
        "day_threshold": int(days_above_threshold),
    }


def peak_summary(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    start_idx: int,
    end_idx: int,
    peak_idx: int,
) -> dict[str, float | int]:
    """Include the peak day in the first period, and exclude it from the second."""
    summary = {
        "peak_concentration": float(values[peak_idx]),
        "days_from_season_start_to_peak": np.nan,
        "pollen_integral_through_peak": np.nan,
        "days_through_season_end": np.nan,
        "pollen_integral_after_peak": np.nan,
    }
    if not start_idx <= peak_idx <= end_idx:
        return summary
    summary.update(
        {
            "days_from_season_start_to_peak": int(
                (dates[peak_idx] - dates[start_idx]).days + 1
            ),
            "pollen_integral_through_peak": float(
                np.nansum(values[start_idx : peak_idx + 1])
            ),
            "days_through_season_end": int((dates[end_idx] - dates[peak_idx]).days),
            "pollen_integral_after_peak": float(
                np.nansum(values[peak_idx + 1 : end_idx + 1])
            ),
        }
    )
    return summary


def format_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(results, columns=RESULT_COLUMNS)
    for column in (
        "ps_start_doy",
        "ps_end_doy",
        "peak_doy",
        "ps_length",
        "days_from_season_start_to_peak",
        "days_through_season_end",
        "day_threshold",
    ):
        frame[column] = frame[column].astype("Int64")
    for column in ("ps_start", "ps_end", "peak_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    return frame
