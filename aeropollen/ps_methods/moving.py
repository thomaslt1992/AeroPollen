from numbers import Integral, Real

import numpy as np
import pandas as pd

from .common import RESULT_COLUMNS, format_results
from .reference_summary import reference_summary


def _validate_moving_params(
    *,
    man: int,
    th_ma: float,
    day_threshold: float,
) -> None:
    if not isinstance(man, Integral) or isinstance(man, bool) or man < 1:
        raise ValueError("man must be a positive integer.")
    for name, value in (("th_ma", th_ma), ("day_threshold", day_threshold)):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number.")


def moving_average(values: np.ndarray, man: int) -> np.ndarray:
    effective_man = int(man) + 1 if int(man) % 2 == 0 else int(man)
    half_window = effective_man // 2
    result = np.empty(values.size, dtype="float64")

    for index in range(values.size):
        start = max(0, index - half_window)
        stop = min(values.size, index + half_window + 1)
        result[index] = float(np.mean(values[start:stop]))
    return result


def _moving_bounds(
    values: np.ndarray,
    *,
    man: int,
    th_ma: float,
) -> tuple[int | None, int | None]:
    if values.size == 0:
        return None, None

    averages = moving_average(values, man)
    if float(np.max(averages)) < float(th_ma):
        return None, None

    moving_peak = int(np.argmax(averages))
    before_peak = np.flatnonzero(averages[:moving_peak] < float(th_ma))
    after_peak = np.flatnonzero(averages[moving_peak + 1 :] < float(th_ma))

    start_idx = int(before_peak[-1] + 1) if before_peak.size else None
    end_idx = int(moving_peak + after_peak[0]) if after_peak.size else None
    return start_idx, end_idx


def calculate_ps_moving(
    dates: pd.Series,
    pollen_df: pd.DataFrame,
    *,
    man: int = 11,
    th_ma: float = 5,
    day_threshold: float = 100,
) -> pd.DataFrame:
    """Calculate pollen seasons around the peak of a centered moving average."""
    _validate_moving_params(
        man=man,
        th_ma=th_ma,
        day_threshold=day_threshold,
    )

    parsed_dates = pd.to_datetime(dates, errors="raise")
    if len(parsed_dates) != len(pollen_df):
        raise ValueError("Length of dates and pollen_df must match.")
    if parsed_dates.isna().any():
        raise ValueError("dates must not contain missing values.")
    if parsed_dates.duplicated().any():
        raise ValueError("dates must contain unique values.")
    if len(parsed_dates) == 0 or pollen_df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    results: list[dict[str, object]] = []
    for pollen_type in pollen_df.columns:
        values = pd.to_numeric(pollen_df[pollen_type], errors="coerce")
        time_series = pd.DataFrame(
            {"date": parsed_dates, "value": values.to_numpy()}
        ).sort_values("date")
        time_series = time_series.set_index("date")

        for season_year, group in time_series.groupby(time_series.index.year):
            observed = group["value"].dropna()
            if observed.empty:
                year_dates = pd.DatetimeIndex([])
                values = np.array([], dtype="float64")
            else:
                year_dates = pd.date_range(observed.index.min(), observed.index.max())
                values = observed.reindex(year_dates, fill_value=0).to_numpy(
                    dtype="float64"
                )
            start_idx, end_idx = _moving_bounds(values, man=int(man), th_ma=th_ma)
            results.append(
                reference_summary(
                    pollen_type,
                    int(season_year),
                    values,
                    year_dates,
                    start_idx,
                    end_idx,
                    day_threshold,
                )
            )

    return format_results(results)
