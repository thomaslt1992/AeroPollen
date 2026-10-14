import pandas as pd

from ..preprocessing.interpolation import preprocess_pollen_timeseries
from .clinical import calculate_ps_clinical
from .grains import calculate_ps_grains
from .moving import calculate_ps_moving
from .percentage import calculate_ps_percentage


def calculate_ps(
    data: pd.DataFrame,
    method: str = "percentage",
    th_day: float = 100,
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
    th_pollen: float = 10,
    th_sum: float = 100,
    season_type: str = "none",
    interpolation: bool = True,
    int_method: str = "linear",
    maxdays: int = 30,
    result: str = "table",
    plot: bool = True,
    export_plot: bool = False,
    export_result: bool = False,
    clinical_pollen_type: str | None = None,
):
    """
    Estimate main pollen season parameters using different methods.

    Supports 'percentage', 'moving', 'clinical', and 'grains' methods with unified
    interpolation logic.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    method = method.lower().strip()
    if method not in {"percentage", "clinical", "moving", "grains"}:
        raise ValueError(
            "Unknown method. Supported methods: 'percentage', 'moving', "
            "'clinical', and 'grains'."
        )
    if result not in {"table", "list"}:
        raise ValueError("result must be 'table' or 'list'.")

    date_col = None
    for col in data.columns:
        if str(col).lower() == "date":
            date_col = col
            break

    if date_col is None:
        raise ValueError("data must contain a 'Date' column (case-insensitive).")

    if method == "percentage" and (perc <= 5 or perc > 100):
        raise ValueError("perc must be in (5, 100]. Typical value: 95.")

    df = preprocess_pollen_timeseries(
        data=data,
        date_col=date_col,
        interpolation=interpolation,
        int_method=int_method,
        maxdays=maxdays,
    )

    dates = df[date_col]
    pollen_cols = [c for c in df.columns if c != date_col]

    if not pollen_cols:
        raise ValueError("data must have at least one pollen column besides 'Date'.")

    if method == "percentage":
        result_obj = calculate_ps_percentage(
            dates=dates,
            pollen_df=df[pollen_cols],
            perc=perc,
            day_threshold=th_day,
            interpolation=False,
            int_method=int_method,
        )

    elif method == "moving":
        result_obj = calculate_ps_moving(
            dates=dates,
            pollen_df=df[pollen_cols],
            man=man,
            th_ma=th_ma,
            day_threshold=th_day,
        )

    elif method == "clinical":
        result_obj = calculate_ps_clinical(
            dates=dates,
            pollen_df=df[pollen_cols],
            n_clinical=n_clinical,
            window_clinical=window_clinical,
            th_pollen=th_pollen,
            th_sum=th_sum,
            day_threshold=th_day,
            clinical_pollen_type=clinical_pollen_type,
        )

    elif method == "grains":
        result_obj = calculate_ps_grains(
            dates=dates,
            pollen_df=df[pollen_cols],
            window_grains=window_grains,
            th_pollen=th_pollen,
            day_threshold=th_day,
        )

    if result == "table":
        return pd.DataFrame(result_obj)
    if result == "list":
        return result_obj
