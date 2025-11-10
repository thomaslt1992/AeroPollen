import pandas as pd

from .percentage import calculate_ps_percentage
from .clinical import calculate_ps_clinical
from ..preprocessing.interpolation import preprocess_pollen_timeseries


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
    season_type: str = "none",
    interpolation: bool = True,
    int_method: str = "lineal",
    maxdays: int = 30,
    result: str = "table",
    plot: bool = True,
    export_plot: bool = False,
    export_result: bool = False,
    clinical_pollen_type: str | None = None,
):
    """
    Estimate main pollen season parameters
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

    if perc <= 5 or perc > 100:
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

    method = method.lower()

    if method == "percentage":
        if perc <= 5 or perc > 100:
            raise ValueError("perc must be in (5, 100]. Typical value: 95.")
        result_obj = calculate_ps_percentage(
            dates=dates,
            pollen_df=df[pollen_cols],
            perc=perc,
            th_sum=th_sum,
            day_threshold_value=th_day,
        )
    elif method == "clinical":
        result_obj = calculate_ps_clinical(
            dates=dates,
            pollen_df=df[pollen_cols],
            n_clinical=n_clinical,
            window_clinical=window_clinical,
            th_pollen=th_pollen,
            th_sum=th_sum,
            interpolation=interpolation,
            int_method=int_method,
            clinical_pollen_type=clinical_pollen_type,
        )

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

    raise ValueError("result must be 'table' or 'list'.")
