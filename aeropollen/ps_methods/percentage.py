import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


def calculate_ps_percentage(
    dates: pd.Series,
    pollen_df: pd.DataFrame,
    perc: float,
    th_sum: int,
    day_threshold: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Percentage-based pollen-season definition (per year).

    Defines the season between 5% and `perc`% of the annual cumulative sum,
    for each pollen type and each year. Skips years with total pollen below
    `th_sum` or seasons shorter than `day_threshold` (if provided).
    """

    dates = pd.to_datetime(dates)

    if len(dates) == 0 or pollen_df.empty:
        return []

    if len(dates) != len(pollen_df):
        raise ValueError("Length of dates and pollen_df must match.")

    results: List[Dict[str, Any]] = []

    for p_type in pollen_df.columns:
        tmp = pd.DataFrame({"date": dates, "value": pd.to_numeric(pollen_df[p_type], errors="coerce")})
        tmp = tmp.set_index("date").sort_index()

        for year, g in tmp.groupby(tmp.index.year):
            vals = g["value"].to_numpy(dtype="float64")
            year_dates = g.index

            if len(vals) == 0:
                results.append(_empty_result(p_type, year))
                continue

            total_integral = float(np.nansum(vals))
            if not np.isfinite(total_integral) or total_integral < th_sum:
                results.append(_empty_result(p_type, year, total_integral))
                continue

            cumsum = np.cumsum(np.nan_to_num(vals, nan=0.0))
            start_threshold = total_integral * 0.05
            end_threshold = total_integral * (perc / 100.0)

            start_idx = int(np.argmax(cumsum >= start_threshold))
            end_idx = int(np.argmax(cumsum >= end_threshold))

            start_date = year_dates[start_idx]
            end_date = year_dates[end_idx]
            ps_length = int((end_date - start_date).days + 1)

            if day_threshold is not None and ps_length < day_threshold:
                # season too short → treat as missing
                results.append(_empty_result(p_type, year, total_integral))
                continue

            peak_idx = int(np.nanargmax(vals)) if np.any(~np.isnan(vals)) else None
            peak_date = year_dates[peak_idx] if peak_idx is not None else pd.NaT

            season_integral = float(np.nansum(vals[start_idx : end_idx + 1]))

            results.append(
            {
                "pollen_type": p_type,
                "season": int(year),
                "ps_start": start_date,
                "ps_end": end_date,
                "peak_date": peak_date,
                "ps_start_doy": int(start_date.dayofyear),
                "ps_end_doy": int(end_date.dayofyear),
                "peak_doy": int(peak_date.dayofyear) if pd.notna(peak_date) else np.nan,
                "ps_length": ps_length,
                "pollen_integral": season_integral,
                "total_pollen_integral": np.nan if total_integral == 0 else total_integral,
                "day_threshold": day_threshold if day_threshold is not None else np.nan,
            })

    return_df = pd.DataFrame(results)

    date_cols = ["ps_start", "ps_end", "peak_date"]
    int_cols = ["ps_start_doy", "ps_end_doy", "peak_doy", "ps_length", "day_threshold"]

    for col in int_cols:
        return_df[col] = return_df[col].astype("Int64")

    for col in date_cols:
        return_df[col] = pd.to_datetime(return_df[col], errors="coerce").dt.strftime("%Y-%m-%d")

    return return_df


def _empty_result(p_type: str, year: int, total_integral: float = np.nan) -> Dict[str, Any]:
    """Return an empty-row dictionary for consistency with output schema."""
    return {
        "pollen_type": p_type,
        "season": int(year),
        "ps_start": pd.NaT,
        "ps_end": pd.NaT,
        "peak_date": pd.NaT,
        "ps_start_doy": np.nan,
        "ps_end_doy": np.nan,
        "peak_doy": np.nan,
        "ps_length": np.nan,
        "pollen_integral": np.nan,
        "total_pollen_integral": total_integral,
        "day_threshold": np.nan,
    }
