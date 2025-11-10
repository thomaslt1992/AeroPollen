import numpy as np
import pandas as pd


def calculate_ps_percentage(
    dates: pd.Series,
    pollen_df: pd.DataFrame,
    perc: float,
    th_sum: int,
):
    """
    5 percentage % pollen-season definition.
    """
    results = []

    for p_type in pollen_df.columns:
        series = (
            pd.to_numeric(pollen_df[p_type], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=float)
        )
        total_sum = series.sum()

        if total_sum < th_sum:
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
                    "ps_length": np.nan,
                    "pollen_integral": float(total_sum),
                    "status": "excluded (low total)",
                }
            )
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
                "pollen_integral": float(total_sum),
                "status": "ok",
            }
        )

    return results
