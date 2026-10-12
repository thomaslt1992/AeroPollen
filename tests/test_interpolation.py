import numpy as np
import pandas as pd
import pytest

from aeropollen.preprocessing.interpolation import preprocess_pollen_timeseries


def test_only_gaps_within_maxdays_are_interpolated() -> None:
    data = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=8, freq="D"),
            "Pollen": [0.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0],
        }
    )

    result = preprocess_pollen_timeseries(data, maxdays=2)

    assert result.loc[1:2, "Pollen"].tolist() == [1.0, 2.0]
    assert result.loc[4:6, "Pollen"].isna().all()


def test_time_interpolation_is_supported() -> None:
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "Pollen": [0.0, 2.0],
        }
    )

    result = preprocess_pollen_timeseries(data, int_method="time", maxdays=1)

    assert result.loc[1, "Pollen"] == 1.0


def test_duplicate_dates_are_rejected() -> None:
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-01-01"]),
            "Pollen": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="unique dates"):
        preprocess_pollen_timeseries(data)
