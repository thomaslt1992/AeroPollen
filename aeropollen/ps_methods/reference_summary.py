"""Season summaries following AeRobiology's calculate_ps.R expressions."""

import numpy as np
import pandas as pd

from .common import RESULT_COLUMNS


def _inclusive_sum(values: np.ndarray, first: int, last: int) -> float:
    # R's a:b includes both endpoints and descends when a > b.
    lower, upper = sorted((first, last))
    if lower < 0 or upper >= len(values):
        return np.nan
    return float(np.sum(values[lower : upper + 1]))


def reference_summary(
    pollen_type: str,
    season: int,
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    start_idx: int | None,
    end_idx: int | None,
    day_threshold: float,
) -> dict[str, object]:
    row = dict.fromkeys(RESULT_COLUMNS, np.nan)
    row.update(pollen_type=pollen_type, season=int(season))
    if not len(values) or float(np.sum(values)) == 0:
        return row

    peak_idx = int(np.argmax(values))
    peak = dates[peak_idx]
    row.update(
        total_pollen_integral=float(np.sum(values)),
        peak_concentration=float(values[peak_idx]),
        peak_date=peak,
        peak_doy=int(peak.dayofyear),
    )
    # R retains annual statistics but stops season summaries at an absent bound.
    if start_idx is None:
        return row
    start = dates[start_idx]
    row.update(ps_start=start, ps_start_doy=int(start.dayofyear))
    if end_idx is None:
        return row
    end = dates[end_idx]
    row.update(
        ps_end=end,
        ps_end_doy=int(end.dayofyear),
        ps_length=int((end - start).days + 1),
        pollen_season_integral=_inclusive_sum(values, start_idx, end_idx),
        days_from_season_start_to_peak=int((peak - start).days + 1),
        pollen_integral_through_peak=_inclusive_sum(values, start_idx, peak_idx),
        days_through_season_end=int((end - peak).days),
        pollen_integral_after_peak=_inclusive_sum(values, peak_idx + 1, end_idx),
        day_threshold=int(np.sum(values >= day_threshold)),
    )
    return row
