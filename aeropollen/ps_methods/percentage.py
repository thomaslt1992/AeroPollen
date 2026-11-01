from typing import Any

import numpy as np
import pandas as pd

from ..preprocessing.interpolation import _maybe_interpolate
from .common import RESULT_COLUMNS, empty_result, format_results


def calculate_ps_percentage(
    dates: pd.Series,
    pollen_df: pd.DataFrame,
    perc: float,
    day_threshold: float = 100,
    interpolation: bool = True,
    int_method: str = "linear",
) -> pd.DataFrame:
    """Calculate seasons containing the central ``perc`` percent of pollen."""
    parsed_dates = pd.to_datetime(dates, errors="raise")
    if len(parsed_dates) != len(pollen_df):
        raise ValueError("Length of dates and pollen_df must match.")
    if parsed_dates.isna().any():
        raise ValueError("dates must not contain missing values.")
    if parsed_dates.duplicated().any():
        raise ValueError("dates must contain unique values.")
    if len(parsed_dates) == 0 or pollen_df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    results: list[dict[str, Any]] = []
    tail_fraction = (100.0 - float(perc)) / 200.0

    for pollen_type in pollen_df.columns:
        values = pd.to_numeric(pollen_df[pollen_type], errors="coerce")
        values.index = parsed_dates
        values = _maybe_interpolate(
            values,
            interpolation=interpolation,
            int_method=int_method,
        )
        time_series = pd.DataFrame(
            {"date": values.index, "value": values.to_numpy()}
        ).sort_values("date")
        time_series = time_series.set_index("date")

        for season_year, group in time_series.groupby(time_series.index.year):
            series = group["value"].to_numpy(dtype="float64")
            year_dates = group.index
            finite = np.isfinite(series)
            days_above = int(np.sum(finite & (series >= float(day_threshold))))

            if not finite.any():
                results.append(
                    empty_result(
                        pollen_type,
                        int(season_year),
                        day_threshold=float(day_threshold),
                    )
                )
                continue

            total_integral = float(np.nansum(series))
            peak_idx = int(np.nanargmax(series))
            peak_date = year_dates[peak_idx]
            if total_integral <= 0:
                results.append(
                    empty_result(
                        pollen_type,
                        int(season_year),
                        day_threshold=float(day_threshold),
                        peak_date=peak_date,
                        total_integral=total_integral,
                        days_above_threshold=days_above,
                    )
                )
                continue

            cumulative = np.cumsum(np.nan_to_num(series, nan=0.0))
            start_idx = int(np.argmax(cumulative >= total_integral * tail_fraction))
            end_candidates = np.flatnonzero(
                cumulative > total_integral * (1.0 - tail_fraction)
            )
            end_idx = int(end_candidates[0]) if end_candidates.size else len(series) - 1
            start_date = year_dates[start_idx]
            end_date = year_dates[end_idx]

            results.append(
                {
                    "pollen_type": pollen_type,
                    "season": int(season_year),
                    "ps_start": start_date,
                    "ps_end": end_date,
                    "peak_date": peak_date,
                    "ps_start_doy": int(start_date.dayofyear),
                    "ps_end_doy": int(end_date.dayofyear),
                    "peak_doy": int(peak_date.dayofyear),
                    "ps_length": int((end_date - start_date).days + 1),
                    "pollen_season_integral": float(
                        np.nansum(series[start_idx : end_idx + 1])
                    ),
                    "total_pollen_integral": total_integral,
                    "day_threshold": float(day_threshold),
                    "days_above_threshold": days_above,
                }
            )

    return format_results(results)
