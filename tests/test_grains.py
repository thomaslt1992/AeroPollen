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


def test_missing_value_breaks_a_grains_run() -> None:
    data = pollen_frame([10, 10, np.nan, 10, 10])

    result = calculate_ps(
        data,
        method="grains",
        window_grains=3,
        th_pollen=10,
        interpolation=False,
    )

    assert pd.isna(result.loc[0, "ps_start"])
    assert pd.isna(result.loc[0, "ps_end"])


def test_missing_calendar_day_breaks_a_grains_run() -> None:
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

    assert pd.isna(result.loc[0, "ps_start"])


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
