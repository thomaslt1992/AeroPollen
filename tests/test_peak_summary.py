import numpy as np
import pandas as pd
import pytest

from aeropollen import calculate_ps
from aeropollen.ps_methods.common import peak_summary


@pytest.mark.parametrize("method", ["clinical", "grains", "moving", "percentage"])
def test_peak_periods_partition_season(method):
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=9),
            "custom": [0, 0, 10, 20, 30, 20, 10, 0, 0],
        }
    )
    row = calculate_ps(
        data,
        method=method,
        interpolation=False,
        man=3,
        th_ma=5,
        n_clinical=3,
        window_clinical=3,
        th_pollen=10,
        th_sum=30,
        window_grains=3,
    ).iloc[0]
    assert row["peak_concentration"] == 30
    assert row["days_from_season_start_to_peak"] == 3
    assert row["pollen_integral_through_peak"] == 60
    assert row["pollen_integral_after_peak"] == 30
    assert (
        row["days_from_season_start_to_peak"] + row["days_through_season_end"]
        == row["ps_length"]
    )
    assert (
        row["pollen_integral_through_peak"] + row["pollen_integral_after_peak"]
        == row["pollen_season_integral"]
    )


def test_peak_at_season_end_has_empty_post_peak_period():
    summary = peak_summary(
        np.array([10, 20, 30]), pd.date_range("2020-01-01", periods=3), 0, 2, 2
    )
    assert summary["days_through_season_end"] == 0
    assert summary["pollen_integral_after_peak"] == 0
    assert summary["pollen_integral_through_peak"] == 60


def test_peak_outside_season_has_missing_period_summaries():
    summary = peak_summary(
        np.array([100, 0, 10, 20]), pd.date_range("2020-01-01", periods=4), 2, 3, 0
    )
    assert summary["peak_concentration"] == 100
    assert all(pd.isna(v) for k, v in summary.items() if k != "peak_concentration")


@pytest.mark.parametrize(
    "values, expected_peak", [([0, 1, 0], 1), ([np.nan] * 3, np.nan)]
)
def test_absent_season_retains_only_available_peak(values, expected_peak):
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "custom": values,
        }
    )
    row = calculate_ps(data, method="clinical", interpolation=False).iloc[0]
    if pd.isna(expected_peak):
        assert pd.isna(row["peak_concentration"])
    else:
        assert row["peak_concentration"] == expected_peak
    assert pd.isna(row["days_through_season_end"])
    assert pd.isna(row["pollen_integral_through_peak"])
