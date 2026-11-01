import numpy as np
import pandas as pd
import pytest

from aeropollen import calculate_ps


def pollen_frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=len(values), freq="D"),
            "Poaceae": values,
        }
    )


def test_grains_uses_first_and_last_consecutive_runs() -> None:
    data = pollen_frame([0, 2, 2, 2, 0, 2, 2, 2, 0])

    result = calculate_ps(
        data,
        method="grains",
        window_grains=3,
        th_pollen=2,
        th_day=2,
        interpolation=False,
    )

    assert result.loc[0, "ps_start"] == "2020-01-02"
    assert result.loc[0, "ps_end"] == "2020-01-08"
    assert result.loc[0, "ps_length"] == 7
    assert result.loc[0, "pollen_season_integral"] == 12
    assert result.loc[0, "day_threshold"] == 6


def test_grains_threshold_is_inclusive() -> None:
    data = pollen_frame([10, 10, 10])

    result = calculate_ps(
        data,
        method="grains",
        window_grains=3,
        th_pollen=10,
        interpolation=False,
    )

    assert result.loc[0, "ps_start"] == "2020-01-01"
    assert result.loc[0, "ps_end"] == "2020-01-03"


def test_missing_value_is_omitted_before_finding_grains_runs() -> None:
    data = pollen_frame([10, 10, np.nan, 10, 10])

    result = calculate_ps(
        data,
        method="grains",
        window_grains=3,
        th_pollen=10,
        interpolation=False,
    )

    assert result.loc[0, "ps_start"] == "2020-01-01"
    assert result.loc[0, "ps_end"] == "2020-01-05"
    assert result.loc[0, "ps_length"] == 5
    assert result.loc[0, "pollen_season_integral"] == 40


def test_missing_calendar_day_does_not_break_a_grains_run() -> None:
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04"]),
            "Poaceae": [10, 10, 10],
        }
    )

    result = calculate_ps(
        data,
        method="grains",
        window_grains=3,
        th_pollen=10,
        interpolation=False,
    )

    assert result.loc[0, "ps_start"] == "2020-01-01"
    assert result.loc[0, "ps_end"] == "2020-01-04"


def test_year_without_qualifying_run_does_not_reuse_previous_year() -> None:
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2020-12-29",
                    "2020-12-30",
                    "2020-12-31",
                    "2021-01-01",
                    "2021-01-02",
                    "2021-01-03",
                ]
            ),
            "Poaceae": [10, 10, 10, 10, 0, 10],
        }
    )

    result = calculate_ps(
        data,
        method="grains",
        window_grains=3,
        th_pollen=10,
        interpolation=False,
    )

    assert result.loc[result["season"] == 2020, "ps_start"].iloc[0] == "2020-12-29"
    assert pd.isna(result.loc[result["season"] == 2021, "ps_start"].iloc[0])


@pytest.mark.parametrize("window_grains", [0, -1, 2.5, True])
def test_invalid_grains_window_is_rejected(window_grains: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        calculate_ps(
            pollen_frame([10, 10, 10]),
            method="grains",
            window_grains=window_grains,
        )


def test_negative_grains_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        calculate_ps(
            pollen_frame([10, 10, 10]),
            method="grains",
            th_pollen=-1,
        )


def test_grains_does_not_extrapolate_missing_edges() -> None:
    row = calculate_ps(
        pollen_frame([np.nan, 10, 10, np.nan]),
        method="grains",
        window_grains=3,
        th_pollen=10,
    ).iloc[0]
    assert pd.isna(row.ps_start)
    assert pd.isna(row.day_threshold)
    assert row.total_pollen_integral == 20
    assert row.peak_date == "2020-01-02"


def test_grains_interpolates_only_short_interior_gaps() -> None:
    data = pollen_frame([0, 10, np.nan, np.nan, np.nan, 10, 0])
    short = calculate_ps(
        data,
        method="grains",
        window_grains=5,
        th_pollen=10,
        maxdays=3,
    ).iloc[0]
    long = calculate_ps(
        data,
        method="grains",
        window_grains=5,
        th_pollen=10,
        maxdays=2,
    ).iloc[0]
    assert short.ps_start == "2020-01-02"
    assert short.ps_end == "2020-01-06"
    assert short.pollen_season_integral == 50
    assert pd.isna(long.ps_start)
    assert long.total_pollen_integral == 20


def test_grains_reproduces_r_post_peak_sum_at_season_end() -> None:
    row = calculate_ps(
        pollen_frame([0, 10, 20, 30, 1, 0]),
        method="grains",
        window_grains=3,
        th_pollen=10,
    ).iloc[0]
    assert row.days_through_season_end == 0
    assert row.pollen_integral_after_peak == 31


def test_grains_reproduces_r_summary_when_peak_is_outside_season() -> None:
    row = calculate_ps(
        pollen_frame([100, 0, 10, 20, 10, 0]),
        method="grains",
        window_grains=3,
        th_pollen=10,
    ).iloc[0]
    assert row.ps_start == "2020-01-03"
    assert row.days_from_season_start_to_peak == -1
    assert row.pollen_integral_through_peak == 110
    assert row.pollen_integral_after_peak == 40
