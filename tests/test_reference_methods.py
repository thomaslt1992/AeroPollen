"""Regression cases derived from AeRobiology's calculate_ps.R and ma.R."""

import numpy as np
import pandas as pd
import pytest

from aeropollen import calculate_ps


def calculate(values, method, **kwargs):
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=len(values)),
            "pollen": values,
        }
    )
    return calculate_ps(data, method=method, **kwargs).iloc[0]


@pytest.mark.parametrize("values", [[10] * 7, [10, 10, 0, 0]])
def test_moving_does_not_invent_a_start_at_the_series_edge(values):
    row = calculate(values, "moving", man=1, th_ma=5, interpolation=False)
    assert pd.isna(row.ps_start)
    assert pd.isna(row.ps_end)
    assert pd.isna(row.ps_length)
    assert pd.isna(row.day_threshold)
    assert row.total_pollen_integral == sum(values)
    assert row.peak_concentration == 10


def test_moving_retains_start_when_end_crossing_is_absent():
    row = calculate([0, 0, 10, 10], "moving", man=1, th_ma=5)
    assert row.ps_start == "2020-01-03"
    assert pd.isna(row.ps_end)
    assert pd.isna(row.pollen_season_integral)


@pytest.mark.parametrize("interpolation", [False, True])
def test_moving_drops_missing_edges_before_smoothing(interpolation):
    row = calculate(
        [np.nan, np.nan, 10, 10, 0, 0, np.nan],
        "moving",
        man=3,
        th_ma=5,
        interpolation=interpolation,
    )
    assert pd.isna(row.ps_start)
    assert row.peak_date == "2020-01-03"
    assert row.total_pollen_integral == 20


def test_moving_reconstructs_internal_gaps_with_zeros():
    row = calculate(
        [0, 0, 9, np.nan, 9, 0, 0],
        "moving",
        man=3,
        th_ma=3,
        th_day=0,
        interpolation=False,
    )
    assert row.ps_start == "2020-01-02"
    assert row.ps_end == "2020-01-06"
    assert row.pollen_season_integral == 18
    assert row.day_threshold == 7


def test_percentage_uses_reverse_sum_and_strict_end_threshold():
    row = calculate([0, 2, 96, 2, 0], "percentage", perc=96)
    assert row.ps_start == "2020-01-02"
    assert row.ps_end == "2020-01-04"
    assert row.pollen_season_integral == 100


@pytest.mark.parametrize("perc", [95, 100])
def test_percentage_does_not_fall_back_to_last_day(perc):
    row = calculate([0, 0, 1, 99], "percentage", perc=perc)
    assert row.ps_start == ("2020-01-04" if perc == 95 else "2020-01-01")
    assert pd.isna(row.ps_end)
    assert pd.isna(row.ps_length)
    assert pd.isna(row.day_threshold)
    assert row.total_pollen_integral == 100


def test_percentage_does_not_map_missing_calendar_boundary_to_observation():
    row = calculate(
        [0, 5, 90, np.nan, 5, 0], "percentage", perc=80, interpolation=False
    )
    assert row.ps_start == "2020-01-03"
    assert pd.isna(row.ps_end)
    assert row.total_pollen_integral == 100


@pytest.mark.parametrize("perc", [0, 1, 5])
def test_percentage_accepts_r_percentage_range(perc):
    row = calculate([0, 25, 50, 25, 0], "percentage", perc=perc)
    assert row.ps_start == "2020-01-03"
    assert row.ps_end == "2020-01-03"


def test_percentage_preserves_r_inclusive_post_peak_slice():
    row = calculate([0, 1, 98, 1, 0], "percentage", perc=95)
    assert row.days_through_season_end == 0
    # In R, (peak + 1):end descends here; it includes the peak and next day.
    assert row.pollen_integral_after_peak == 99


@pytest.mark.parametrize("method", ["moving", "percentage"])
@pytest.mark.parametrize("grouping", ["interannual", "peak", "unknown"])
def test_unsupported_season_grouping_is_not_silently_ignored(method, grouping):
    with pytest.raises(ValueError, match="natural"):
        calculate([0, 10, 0], method, def_season=grouping)


@pytest.mark.parametrize("method", ["moving", "percentage"])
def test_zero_year_has_missing_statistics(method):
    row = calculate([0, 0, 0], method, interpolation=False)
    assert pd.isna(row.peak_concentration)
    assert pd.isna(row.total_pollen_integral)


def test_internal_interpolation_respects_gap_limit():
    short = calculate([0, 0, 9, np.nan, 9, 0, 0], "moving", man=1, th_ma=5, maxdays=2)
    long = calculate(
        [0, 0, 9, np.nan, np.nan, np.nan, 9, 0, 0], "moving", man=1, th_ma=5, maxdays=2
    )
    assert short.pollen_season_integral == 27
    assert short.ps_length == 3
    assert long.pollen_season_integral == 9
    assert long.ps_length == 1
