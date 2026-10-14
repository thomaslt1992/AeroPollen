from numbers import Integral, Real

import numpy as np
import pandas as pd

from .common import RESULT_COLUMNS, empty_result, format_results


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
    """Return the first and last bounds of qualifying consecutive-day runs."""
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
    """Calculate pollen seasons using consecutive threshold-exceeding days.

    The season starts on the first day of the first run of ``window_grains``
    consecutive days at or above ``th_pollen``. It ends on the final day of
    the last qualifying run. Missing values break a run.
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
            # Preserve calendar-day consecutiveness even when this lower-level
            # function is called without the package preprocessing step.
            full_dates = pd.date_range(group.index.min(), group.index.max(), freq="D")
            group = group.reindex(full_dates)
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
            start_idx, end_idx = _grains_bounds(
                series,
                window_grains=int(window_grains),
                th_pollen=float(th_pollen),
            )

            if start_idx is None or end_idx is None:
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
