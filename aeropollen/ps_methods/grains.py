from numbers import Integral, Real

import numpy as np
import pandas as pd

from .common import RESULT_COLUMNS, format_results
from .reference_summary import reference_summary


def _validate_grains_params(
    *,
    window_grains: int,
    th_pollen: float,
    day_threshold: float,
) -> None:
    if (
        not isinstance(window_grains, Integral)
        or isinstance(window_grains, bool)
        or window_grains < 1
    ):
        raise ValueError("window_grains must be a positive integer.")

    for name, value in (
        ("th_pollen", th_pollen),
        ("day_threshold", day_threshold),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number.")


def _grains_bounds(
    series: np.ndarray,
    *,
    window_grains: int,
    th_pollen: float,
) -> tuple[int | None, int | None]:
    """Return bounds of qualifying runs in the retained observations."""
    if series.size < window_grains:
        return None, None

    qualifying = np.isfinite(series) & (series >= th_pollen)
    run_totals = np.convolve(
        qualifying.astype(np.int64),
        np.ones(window_grains, dtype=np.int64),
        mode="valid",
    )
    run_starts = np.flatnonzero(run_totals == window_grains)
    if run_starts.size == 0:
        return None, None

    start_idx = int(run_starts[0])
    end_idx = int(run_starts[-1] + window_grains - 1)
    return start_idx, end_idx


def calculate_ps_grains(
    dates: pd.Series,
    pollen_df: pd.DataFrame,
    *,
    window_grains: int = 5,
    th_pollen: float = 10,
    day_threshold: float = 100,
) -> pd.DataFrame:
    """Follow AeRobiology grains rules after dropping missing observations.

    Runs contain ``window_grains`` observations at or above ``th_pollen``.
    Unresolved missing calendar days do not break a run, matching R.
    """
    _validate_grains_params(
        window_grains=window_grains,
        th_pollen=th_pollen,
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
            series = observed.to_numpy(dtype="float64")
            year_dates = observed.index
            start_idx, end_idx = _grains_bounds(
                series,
                window_grains=int(window_grains),
                th_pollen=float(th_pollen),
            )
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
