import numpy as np
import pandas as pd
import yaml

from typing import Dict, Any, Optional
from pathlib import Path


def load_clinical_cfg(path: str | Path = None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "clinical_thresholds.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _maybe_interpolate(
    s: pd.Series,
    interpolation: bool,
    int_method: str,
) -> pd.Series:
    if not interpolation:
        return s

    method = int_method or "linear"
    if method == "linear":
        return s.interpolate(method="linear")
    if method == "time" and isinstance(s.index, pd.DatetimeIndex):
        return s.interpolate(method="time")

    return s.interpolate(method="linear")


def _get_params_for_pollen(
    p_type: str,
    pollen_params: Optional[Dict[str, Dict[str, Any]]],
    default_n_clinical: int,
    default_window_clinical: int,
    default_th_pollen: float,
    default_th_sum: float,
) -> Dict[str, Any]:
    if pollen_params is None:
        cfg: Dict[str, Any] = {}
    else:
        lower_map = {k.lower(): v for k, v in pollen_params.items()}
        cfg = lower_map.get(p_type.lower(), {})

    return {
        "n_clinical": int(cfg.get("n_clinical", default_n_clinical)),
        "window_clinical": int(cfg.get("window_clinical", default_window_clinical)),
        "th_pollen": float(cfg.get("th_pollen", default_th_pollen)),
        "th_sum": float(cfg.get("th_sum", default_th_sum)),
        "day_threshold": float(
            cfg.get("day_threshold", cfg.get("th_pollen", default_th_pollen))
        ),
    }


def _clinical_bounds(
    series: np.ndarray,
    n_clinical: int,
    window_clinical: int,
    th_pollen: float,
    th_sum: float,
) -> tuple[Optional[int], Optional[int]]:
    n = len(series)
    if n < window_clinical:
        return None, None

    start_idx: Optional[int] = None
    end_idx: Optional[int] = None

    for i in range(0, n - window_clinical + 1):
        window = series[i : i + window_clinical]

        valid = ~np.isnan(window)
        if valid.sum() == 0:
            continue

        high_mask = (window >= th_pollen) & valid
        if high_mask.sum() < n_clinical:
            continue

        high_vals = window[high_mask]
        if np.nansum(high_vals) < th_sum:
            continue

        high_idx = np.flatnonzero(high_mask)
        if start_idx is None:
            start_idx = i + int(high_idx[0])
        end_idx = i + int(high_idx[-1])

    return start_idx, end_idx


def calculate_ps_clinical(
    dates: pd.Series,
    pollen_df: pd.DataFrame,
    *,
    pollen_params: Optional[Dict[str, Dict[str, Any]]] = None,
    clinical_pollen_type : Optional[str] = None,
    n_clinical: int = 5,
    window_clinical: int = 7,
    th_pollen: float = 10.0,
    th_sum: float = 100.0,
    interpolation: bool = True,
    int_method: str = "linear",
):
    if pollen_params is None:
        pollen_params = load_clinical_cfg()

    dates = pd.to_datetime(dates)
    results = []

    for p_type in pollen_df.columns:
        raw = pd.to_numeric(pollen_df[p_type], errors="coerce")
        series = _maybe_interpolate(
            raw, interpolation=interpolation, int_method=int_method
        )

        lookup_key = clinical_pollen_type or p_type

        params = _get_params_for_pollen(
            p_type=lookup_key,
            pollen_params=pollen_params,
            default_n_clinical=n_clinical,
            default_window_clinical=window_clinical,
            default_th_pollen=th_pollen,
            default_th_sum=th_sum,
        )

        vals = series.to_numpy(dtype="float64")
        total_integral = float(np.nansum(vals)) if len(vals) else 0.0
        day_threshold_count = (
            int(np.sum((~np.isnan(vals)) & (vals >= params["day_threshold"])))
            if len(vals)
            else 0
        )

        if len(dates) == 0:
            results.append(
                {
                    "pollen_type": p_type,
                    "season": np.nan,
                    "season_start": pd.NaT,
                    "season_end": pd.NaT,
                    "peak_date": pd.NaT,
                    "season_start_doy": np.nan,
                    "season_end_doy": np.nan,
                    "peak_doy": np.nan,
                    "ps_length": 0,
                    "pollen_integral": 0.0,
                    "total_pollen_integral": total_integral,
                    "day_threshold": day_threshold_count,
                }
            )
            continue

        start_idx, end_idx = _clinical_bounds(
            series=vals,
            n_clinical=params["n_clinical"],
            window_clinical=params["window_clinical"],
            th_pollen=params["th_pollen"],
            th_sum=params["th_sum"],
        )

        if start_idx is None or end_idx is None:
            season_year = int(dates.iloc[0].year)
            results.append(
                {
                    "pollen_type": p_type,
                    "season": season_year,
                    "season_start": pd.NaT,
                    "season_end": pd.NaT,
                    "peak_date": pd.NaT,
                    "season_start_doy": np.nan,
                    "season_end_doy": np.nan,
                    "peak_doy": np.nan,
                    "ps_length": 0,
                    "pollen_integral": 0.0,
                    "total_pollen_integral": total_integral,
                    "day_threshold": day_threshold_count,
                }
            )
            continue

        start_date = dates.iloc[start_idx]
        end_date = dates.iloc[end_idx]
        season_year = int(start_date.year)

        season_slice = vals[start_idx : end_idx + 1]
        if np.all(np.isnan(season_slice)):
            peak_idx = start_idx
        else:
            peak_offset = int(np.nanargmax(season_slice))
            peak_idx = start_idx + peak_offset

        peak_date = dates.iloc[peak_idx]

        start_doy = int(start_date.dayofyear)
        end_doy = int(end_date.dayofyear)
        peak_doy = int(peak_date.dayofyear)
        ps_length = int((end_date - start_date).days + 1)
        season_integral = float(np.nansum(season_slice))

        results.append(
            {
                "pollen_type": p_type,
                "season": season_year,
                "season_start": start_date,
                "season_end": end_date,
                "peak_date": peak_date,
                "season_start_doy": start_doy,
                "season_end_doy": end_doy,
                "peak_doy": peak_doy,
                "ps_length": ps_length,
                "pollen_integral": season_integral,
                "total_pollen_integral": total_integral,
                "day_threshold": day_threshold_count,
            }
        )

    return results
