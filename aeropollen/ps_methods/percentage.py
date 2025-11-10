import numpy as np
import pandas as pd


def calculate_ps_percentage(
    dates: pd.Series,
    pollen_df: pd.DataFrame,
    perc: float,
    th_sum: int,
    day_threshold_value: int,
):
    results = []

    for p_type in pollen_df.columns:
        series = pd.to_numeric(pollen_df[p_type], errors="coerce").to_numpy(dtype=float)

        if np.all(np.isnan(series)):
            results.append(
                {
                    "pollen_type": p_type,
                    "season": np.nan,
                    "ps_start": pd.NaT,
                    "ps_end": pd.NaT,
                    "peak_date": pd.NaT,
                    "ps_start_doy": np.nan,
                    "ps_end_doy": np.nan,
                    "peak_doy": np.nan,
                    "ps_length": np.nan,
                    "pollen_integral": np.nan,
                    "total_pollen_integral": np.nan,
                    "day_threshold": np.nan,
                }
            )
            continue

        total_sum = float(np.nansum(series))
        if total_sum <= 0:
            results.append(
                {
                    "pollen_type": p_type,
                    "season": np.nan,
                    "ps_start": pd.NaT,
                    "ps_end": pd.NaT,
                    "peak_date": pd.NaT,
                    "ps_start_doy": np.nan,
                    "ps_end_doy": np.nan,
                    "peak_doy": np.nan,
                    "ps_length": np.nan,
                    "pollen_integral": np.nan,
                    "total_pollen_integral": total_sum,
                    "day_threshold": np.nan,
                }
            )
            continue

        clean_series = np.nan_to_num(series, nan=0.0)
        cumsum = np.cumsum(clean_series)

        gap = (100.0 - perc) / 2.0
        start_pct = gap
        end_pct = 100.0 - gap

        start_threshold = total_sum * (start_pct / 100.0)
        end_threshold = total_sum * (end_pct / 100.0)

        start_idx = int(np.argmax(cumsum >= start_threshold))
        end_idx = int(np.argmax(cumsum >= end_threshold))
        peak_idx = int(np.nanargmax(series))

        start_date = dates.iloc[start_idx]
        end_date = dates.iloc[end_idx]
        peak_date = dates.iloc[peak_idx]

        season_year = int(start_date.year)
        start_doy = int(start_date.dayofyear)
        end_doy = int(end_date.dayofyear)
        peak_doy = int(peak_date.dayofyear)
        ps_length = int((end_date - start_date).days + 1)

        season_slice = series[start_idx : end_idx + 1]
        season_sum = float(np.nansum(season_slice))
        days_above_threshold = int((season_slice > day_threshold_value).sum())

        results.append(
            {
                "pollen_type": p_type,
                "season": season_year,
                "ps_start": start_date,
                "ps_end": end_date,
                "peak_date": peak_date,
                "ps_start_doy": start_doy,
                "ps_end_doy": end_doy,
                "peak_doy": peak_doy,
                "ps_length": ps_length,
                "pollen_integral": season_sum,
                "total_pollen_integral": total_sum,
                "day_threshold": int(days_above_threshold),
            }
        )

    return results
