from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

CLINICAL_POLLEN_ALIASES = {"grass": "grasses"}

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
    "days_above_threshold",
]


def load_clinical_cfg(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    config_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / "config" / "clinical_thresholds.yaml"
    )
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise TypeError("Clinical configuration must contain a mapping.")
    return config


def _canonical_pollen_type(pollen_type: str) -> str:
    normalized = pollen_type.lower().strip()
    return CLINICAL_POLLEN_ALIASES.get(normalized, normalized)


def _validate_clinical_params(params: dict[str, float | int]) -> None:
    n_clinical = params["n_clinical"]
    window_clinical = params["window_clinical"]
    if (
        not isinstance(n_clinical, int)
        or isinstance(n_clinical, bool)
        or n_clinical < 1
    ):
        raise ValueError("n_clinical must be a positive integer.")
    if (
        not isinstance(window_clinical, int)
        or isinstance(window_clinical, bool)
        or window_clinical < 1
    ):
        raise ValueError("window_clinical must be a positive integer.")
    if n_clinical > window_clinical:
        raise ValueError("n_clinical cannot exceed window_clinical.")
    for name in ("th_pollen", "th_sum", "day_threshold"):
        value = params[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative number.")


def _resolve_clinical_params(
    *,
    pollen_params: dict[str, dict[str, Any]],
    clinical_pollen_type: str | None,
    n_clinical: int,
    window_clinical: int,
    th_pollen: float,
    th_sum: float,
    day_threshold: float,
) -> dict[str, float | int]:
    requested_type = clinical_pollen_type
    if requested_type is not None and requested_type.lower().strip() == "none":
        requested_type = None

    if requested_type is None:
        params: dict[str, float | int] = {
            "n_clinical": n_clinical,
            "window_clinical": window_clinical,
            "th_pollen": th_pollen,
            "th_sum": th_sum,
            "day_threshold": day_threshold,
        }
    else:
        canonical_type = _canonical_pollen_type(requested_type)
        normalized_config = {
            key.lower().strip(): value for key, value in pollen_params.items()
        }
        if canonical_type not in normalized_config:
            available = ", ".join(sorted(normalized_config))
            raise ValueError(
                f"Unknown clinical_pollen_type '{clinical_pollen_type}'. "
                f"Available types: {available}."
            )
        preset = normalized_config[canonical_type]
        params = {
            "n_clinical": int(preset["n_clinical"]),
            "window_clinical": int(preset["window_clinical"]),
            "th_pollen": float(preset["th_pollen"]),
            "th_sum": float(preset["th_sum"]),
            "day_threshold": float(preset["day_threshold"]),
        }

    _validate_clinical_params(params)
    return params


def _window_meets_clinical_rule(
    current_value: float,
    window: np.ndarray,
    *,
    n_clinical: int,
    th_pollen: float,
    th_sum: float,
) -> bool:
    if not np.isfinite(current_value) or current_value < th_pollen:
        return False

    qualifying = window[np.isfinite(window) & (window >= th_pollen)]
    if qualifying.size < n_clinical:
        return False

    largest_values = np.sort(qualifying)[-n_clinical:]
    return bool(largest_values.sum() >= th_sum)


def _clinical_bounds(
    series: np.ndarray,
    n_clinical: int,
    window_clinical: int,
    th_pollen: float,
    th_sum: float,
) -> tuple[int | None, int | None]:
    start_idx: int | None = None
    end_idx: int | None = None

    for index, value in enumerate(series):
        forward_window = series[index : index + window_clinical]
        if _window_meets_clinical_rule(
            value,
            forward_window,
            n_clinical=n_clinical,
            th_pollen=th_pollen,
            th_sum=th_sum,
        ):
            start_idx = index
            break

    for index, value in enumerate(series):
        window_start = max(0, index - window_clinical + 1)
        backward_window = series[window_start : index + 1]
        if _window_meets_clinical_rule(
            value,
            backward_window,
            n_clinical=n_clinical,
            th_pollen=th_pollen,
            th_sum=th_sum,
        ):
            end_idx = index

    if start_idx is None or end_idx is None or end_idx < start_idx:
        return None, None
    return start_idx, end_idx


def _empty_result(
    pollen_type: str,
    season: int,
    *,
    day_threshold: float,
    peak_date: Any = pd.NaT,
    total_integral: float = np.nan,
    days_above_threshold: int = 0,
) -> dict[str, Any]:
    return {
        "pollen_type": pollen_type,
        "season": season,
        "ps_start": pd.NaT,
        "ps_end": pd.NaT,
        "peak_date": peak_date,
        "ps_start_doy": np.nan,
        "ps_end_doy": np.nan,
        "peak_doy": peak_date.dayofyear if pd.notna(peak_date) else np.nan,
        "ps_length": np.nan,
        "pollen_season_integral": np.nan,
        "total_pollen_integral": total_integral,
        "day_threshold": day_threshold,
        "days_above_threshold": days_above_threshold,
    }


def calculate_ps_clinical(
    dates: pd.Series,
    pollen_df: pd.DataFrame,
    *,
    pollen_params: dict[str, dict[str, Any]] | None = None,
    clinical_pollen_type: str | None = None,
    n_clinical: int = 5,
    window_clinical: int = 7,
    th_pollen: float = 10,
    th_sum: float = 100,
    day_threshold: float = 100,
) -> pd.DataFrame:
    config = load_clinical_cfg() if pollen_params is None else pollen_params
    params = _resolve_clinical_params(
        pollen_params=config,
        clinical_pollen_type=clinical_pollen_type,
        n_clinical=n_clinical,
        window_clinical=window_clinical,
        th_pollen=th_pollen,
        th_sum=th_sum,
        day_threshold=day_threshold,
    )

    parsed_dates = pd.to_datetime(dates)
    if len(parsed_dates) != len(pollen_df):
        raise ValueError("Length of dates and pollen_df must match.")
    if len(parsed_dates) == 0 or pollen_df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    results: list[dict[str, Any]] = []
    for pollen_type in pollen_df.columns:
        values = pd.to_numeric(pollen_df[pollen_type], errors="coerce")
        time_series = pd.DataFrame({"date": parsed_dates, "value": values}).set_index(
            "date"
        )

        for season_year, group in time_series.groupby(time_series.index.year):
            series = group["value"].to_numpy(dtype="float64")
            year_dates = group.index
            finite = np.isfinite(series)
            days_above = int(
                np.sum(finite & (series >= float(params["day_threshold"])))
            )

            if not finite.any():
                results.append(
                    _empty_result(
                        pollen_type,
                        int(season_year),
                        day_threshold=float(params["day_threshold"]),
                    )
                )
                continue

            total_integral = float(np.nansum(series))
            peak_idx = int(np.nanargmax(series))
            peak_date = year_dates[peak_idx]
            start_idx, end_idx = _clinical_bounds(
                series,
                n_clinical=int(params["n_clinical"]),
                window_clinical=int(params["window_clinical"]),
                th_pollen=float(params["th_pollen"]),
                th_sum=float(params["th_sum"]),
            )

            if start_idx is None or end_idx is None:
                results.append(
                    _empty_result(
                        pollen_type,
                        int(season_year),
                        day_threshold=float(params["day_threshold"]),
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
                    "day_threshold": float(params["day_threshold"]),
                    "days_above_threshold": days_above,
                }
            )

    result = pd.DataFrame(results, columns=RESULT_COLUMNS)
    for column in (
        "ps_start_doy",
        "ps_end_doy",
        "peak_doy",
        "ps_length",
        "days_above_threshold",
    ):
        result[column] = result[column].astype("Int64")
    for column in ("ps_start", "ps_end", "peak_date"):
        result[column] = pd.to_datetime(result[column], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    return result
