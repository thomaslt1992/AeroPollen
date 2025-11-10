import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ClinicalParams:
    n_clinical: int = 5
    window_clinical: int = 7
    th_pollen: float = 10.0
    th_sum: float = 100.0
    th_day: Optional[float] = 100.0


@dataclass(frozen=True)
class ClinicalSeason:
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    high_days: pd.DatetimeIndex


def _params_for_type(clinical_type, n_clinical, window_clinical, th_pollen, th_sum, th_day):
    if clinical_type is None:
        return ClinicalParams(
            n_clinical or 5,
            window_clinical or 7,
            th_pollen or 10.0,
            th_sum or 100.0,
            th_day or 100.0,
        )

    ctype = clinical_type.lower()
    if ctype == "birch":
        return ClinicalParams(5, 7, 10.0, 100.0, 100.0)
    if ctype == "grasses":
        return ClinicalParams(5, 7, 3.0, 30.0, 50.0)
    raise ValueError(f"Unknown clinical type: {clinical_type}")


def clinical_season(series: pd.Series, clinical_type=None, n_clinical=None,
                    window_clinical=None, th_pollen=None, th_sum=None, th_day=None) -> ClinicalSeason:

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")

    series = series.sort_index()
    params = _params_for_type(clinical_type, n_clinical, window_clinical, th_pollen, th_sum, th_day)
    values = series.to_numpy(float)
    dates = series.index
    n = len(values)

    start = end = None
    for i in range(n - params.window_clinical + 1):
        window = values[i:i + params.window_clinical]
        mask = window >= params.th_pollen
        if mask.sum() >= params.n_clinical and window[mask].sum() >= params.th_sum:
            if start is None:
                start = dates[i + np.flatnonzero(mask)[0]]
            end = dates[i + np.flatnonzero(mask)[-1]]

    high_days = pd.DatetimeIndex([], name=series.index.name)
    if params.th_day is not None:
        mask = series >= params.th_day
        high_days = series.index[mask]

    return ClinicalSeason(start, end, high_days)
