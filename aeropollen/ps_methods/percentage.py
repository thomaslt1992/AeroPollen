from typing import Any

import numpy as np
import pandas as pd

from ..preprocessing.interpolation import preprocess_pollen_timeseries
from .common import RESULT_COLUMNS, format_results
from .reference_summary import reference_summary


def calculate_ps_percentage(
    dates: pd.Series,
    pollen_df: pd.DataFrame,
    perc: float,
    day_threshold: float = 100,
    interpolation: bool = True,
    int_method: str = "linear",
) -> pd.DataFrame:
    """Calculate seasons containing the central ``perc`` percent of pollen."""
    if not np.isfinite(perc) or not 0 <= perc <= 100:
        raise ValueError("perc must be between 0 and 100.")
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

    for pollen_type in pollen_df.columns:
        values = pd.to_numeric(pollen_df[pollen_type], errors="coerce")
        time_series = pd.DataFrame(
            {"date": parsed_dates.to_numpy(), "value": values.to_numpy()}
        )
        if interpolation:
            time_series = preprocess_pollen_timeseries(
                time_series,
                date_col="date",
                int_method=int_method,
                extrapolate=False,
            )
        time_series = time_series.sort_values("date").set_index("date")

        for season_year, group in time_series.groupby(time_series.index.year):
            observed = group["value"].dropna()
            series = observed.to_numpy(dtype="float64")
            year_dates = observed.index
            start_idx = None
            end_idx = None
            if series.size and float(np.sum(series)) != 0:
                limit = float(np.sum(series)) / 100 * ((100 - float(perc)) / 2)
                cumulative = np.cumsum(series)
                remaining = np.cumsum(series[::-1])[::-1]
                starts = np.flatnonzero(cumulative >= limit)
                ends = np.flatnonzero(remaining < limit)
                if starts.size:
                    start_idx = int(starts[0])
                if ends.size:
                    end_date = year_dates[int(ends[0])] - pd.Timedelta(days=1)
                    matches = np.flatnonzero(year_dates == end_date)
                    if matches.size:
                        end_idx = int(matches[0])
            results.append(
                reference_summary(
                    pollen_type,
                    int(season_year),
                    series,
                    year_dates,
                    start_idx,
                    end_idx,
                    day_threshold,
                )
            )

    return format_results(results)
