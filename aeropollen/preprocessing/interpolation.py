import pandas as pd

SUPPORTED_INTERPOLATION_METHODS = {"linear", "time"}


def _normalize_interpolation_method(int_method: str) -> str:
    method = int_method.lower().strip()
    if method == "lineal":
        method = "linear"
    if method not in SUPPORTED_INTERPOLATION_METHODS:
        supported = ", ".join(sorted(SUPPORTED_INTERPOLATION_METHODS | {"lineal"}))
        raise ValueError(f"int_method must be one of: {supported}.")
    return method


def _interpolate_short_gaps(
    series: pd.Series,
    *,
    method: str,
    maxdays: int,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if maxdays == 0 or not numeric.isna().any():
        return numeric

    missing = numeric.isna()
    groups = missing.ne(missing.shift(fill_value=False)).cumsum()
    gap_sizes = missing.groupby(groups).transform("sum")
    fillable = missing & gap_sizes.le(maxdays)
    interpolated = numeric.interpolate(method=method, limit_direction="both")

    result = numeric.copy()
    result.loc[fillable] = interpolated.loc[fillable]
    return result


def _maybe_interpolate(
    series: pd.Series,
    interpolation: bool,
    int_method: str,
) -> pd.Series:
    if not interpolation:
        return series
    method = _normalize_interpolation_method(int_method)
    return _interpolate_short_gaps(series, method=method, maxdays=len(series))


def preprocess_pollen_timeseries(
    data: pd.DataFrame,
    date_col: str = "Date",
    interpolation: bool = True,
    int_method: str = "linear",
    maxdays: int = 30,
) -> pd.DataFrame:
    """Return a continuous daily pollen time series with optional gap filling."""
    if date_col not in data.columns:
        raise ValueError(f"'{date_col}' column not found in data.")
    if data.empty:
        return data.copy()
    if not isinstance(maxdays, int) or isinstance(maxdays, bool) or maxdays < 0:
        raise ValueError("maxdays must be a non-negative integer.")

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    if df[date_col].isna().any():
        raise ValueError(f"'{date_col}' must not contain missing dates.")
    if df[date_col].duplicated().any():
        raise ValueError(f"'{date_col}' must contain unique dates.")

    df = df.sort_values(date_col)
    value_cols = [column for column in df.columns if column != date_col]
    full_range = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    df = df.set_index(date_col).reindex(full_range)
    df.index.name = date_col

    if interpolation:
        method = _normalize_interpolation_method(int_method)
        for column in value_cols:
            df[column] = _interpolate_short_gaps(
                df[column],
                method=method,
                maxdays=maxdays,
            )

    return df.reset_index()
