import pandas as pd
import numpy as np


def calculate_ps(
    data: pd.DataFrame,
    method: str = "percentage",
    th_day: int = 100,
    perc: float = 95,
    def_season: str = "natural",
    reduction: bool = False,
    red_level: float = 0.9,
    derivative: int = 5,
    man: int = 11,
    th_ma: int = 5,
    n_clinical: int = 5,
    window_clinical: int = 7,
    window_grains: int = 5,
    th_pollen: int = 10,
    th_sum: int = 100,
    type: str = "none",
    interpolation: bool = True,
    int_method: str = "lineal",
    maxdays: int = 30,
    result: str = "table",
    plot: bool = True,
    export_plot: bool = False,
    export_result: bool = False
):
    """
    Core pollen-season calculation.

    Interface aligned with AeRobiology::calculate_ps.
    Currently only 'percentage' is implemented.
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")

    date_col = None
    for col in data.columns:
        if col.lower() == "date":
            date_col = col
            break

    if date_col is None:
        raise ValueError("data must contain a 'Date' column (case-insensitive).")

    dates = pd.to_datetime(data[date_col])
    pollen_cols = [c for c in data.columns if c != date_col]

    if not pollen_cols:
        raise ValueError("data must have at least one pollen column besides 'Date'.")

    method = method.lower()

    if method == "percentage":
        result_obj = _calculate_ps_percentage(
            dates=dates,
            pollen_df=data[pollen_cols],
            perc=perc,
            th_sum=th_sum,
        )

    elif method == "clinical":
        raise NotImplementedError("Method 'clinical' is not implemented yet.")

    elif method == "logistic":
        raise NotImplementedError("Method 'logistic' is not implemented yet.")

    elif method == "moving":
        raise NotImplementedError("Method 'moving' is not implemented yet.")

    elif method == "grains":
        raise NotImplementedError("Method 'grains' is not implemented yet.")

    else:
        raise ValueError(
            "Unknown method. Supported now: 'percentage'. "
            "Planned: 'clinical', 'logistic', 'moving', 'grains'."
        )

    if result == "table":
        return pd.DataFrame(result_obj)
    if result == "list":
        return result_obj
    raise ValueError("result must be 'table' or 'list'")


def _calculate_ps_percentage(
    dates: pd.Series,
    pollen_df: pd.DataFrame,
    perc: float,
    th_sum: int,
):
    results = []

    for p_type in pollen_df.columns:
        series = pd.to_numeric(pollen_df[p_type], errors="coerce").fillna(0).to_numpy(dtype=float)
        total_sum = series.sum()

        if total_sum < th_sum:
            results.append({
                "pollen_type": p_type,
                "season": np.nan,
                "season_start": pd.NaT,
                "season_end": pd.NaT,
                "peak_date": pd.NaT,
                "season_start_doy": np.nan,
                "season_end_doy": np.nan,
                "peak_doy": np.nan,
                "ps_length": np.nan,
                "pollen_integral": float(total_sum),
                "status": "excluded (low total)",
            })
            continue

        cumsum = np.cumsum(series)
        start_threshold = total_sum * 0.05
        end_threshold = total_sum * (perc / 100.0)

        start_idx = int(np.argmax(cumsum >= start_threshold))
        end_idx = int(np.argmax(cumsum >= end_threshold))
        peak_idx = int(np.argmax(series))

        start_date = dates.iloc[start_idx]
        end_date = dates.iloc[end_idx]
        peak_date = dates.iloc[peak_idx]

        season_year = int(start_date.year)
        start_doy = int(start_date.dayofyear)
        end_doy = int(end_date.dayofyear)
        peak_doy = int(peak_date.dayofyear)
        ps_length = int((end_date - start_date).days + 1)

        results.append({
            "pollen_type": p_type,
            "season": season_year,
            "season_start": start_date,
            "season_end": end_date,
            "peak_date": peak_date,
            "season_start_doy": start_doy,
            "season_end_doy": end_doy,
            "peak_doy": peak_doy,
            "ps_length": ps_length,
            "pollen_integral": float(total_sum),
            "status": "ok",
        })

    return results
