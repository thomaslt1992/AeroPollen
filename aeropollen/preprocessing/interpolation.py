import pandas as pd


def preprocess_pollen_timeseries(
    data: pd.DataFrame,
    date_col: str = "Date",
    interpolation: bool = True,
    int_method: str = "lineal",
    maxdays: int = 30,
) -> pd.DataFrame:
    """
    Ensure daily continuity and optional interpolation for pollen time series.
    """
    if date_col not in data.columns:
        raise ValueError(f"'{date_col}' column not found in data.")

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    value_cols = [c for c in df.columns if c != date_col]

    full_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    df = df.set_index(date_col).reindex(full_range)
    df.index.name = date_col

    if not interpolation:
        return df.reset_index()

    method = int_method.lower()
    if method == "lineal":
        method = "linear"

    if method not in {"linear"}:
        raise ValueError("int_method must be 'lineal' or 'linear' for now.")

    for col in value_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        df[col] = s.interpolate(
            method=method,
            limit=maxdays,
            limit_direction="both",
        )

    return df.reset_index()
