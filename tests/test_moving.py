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


def test_moving_uses_threshold_interval_around_moving_peak() -> None:
    data = pollen_frame([0, 0, 6, 6, 6, 0, 0])

    result = calculate_ps(
        data,
        method="moving",
        man=3,
        th_ma=4,
        th_day=6,
        interpolation=False,
    )

    assert result.loc[0, "ps_start"] == "2020-01-03"
    assert result.loc[0, "ps_end"] == "2020-01-05"
    assert result.loc[0, "peak_date"] == "2020-01-03"
    assert result.loc[0, "ps_length"] == 3
    assert result.loc[0, "pollen_season_integral"] == 18
    assert result.loc[0, "day_threshold"] == 3


def test_even_moving_window_is_promoted_to_next_odd_window() -> None:
    data = pollen_frame([0, 0, 6, 6, 6, 0, 0])

    even = calculate_ps(
        data,
        method="moving",
        man=4,
        th_ma=3,
        interpolation=False,
    )
    odd = calculate_ps(
        data,
        method="moving",
        man=5,
        th_ma=3,
        interpolation=False,
    )

    pd.testing.assert_frame_equal(even, odd)


def test_moving_returns_empty_season_below_threshold() -> None:
    result = calculate_ps(
        pollen_frame([0, 1, 1, 0]),
        method="moving",
        man=3,
        th_ma=2,
        interpolation=False,
    )

    assert pd.isna(result.loc[0, "ps_start"])
    assert pd.isna(result.loc[0, "ps_end"])
    assert result.loc[0, "total_pollen_integral"] == 2


@pytest.mark.parametrize("man", [0, -1, 2.5, True])
def test_invalid_moving_window_is_rejected(man: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        calculate_ps(pollen_frame([1, 2, 1]), method="moving", man=man)


def test_all_methods_return_the_same_columns() -> None:
    data = pollen_frame([0, 10, 10, 10, 10, 10, 0])
    expected = [
        "pollen_type",
        "season",
        "ps_start",
        "ps_end",
        "peak_date",
        "ps_start_doy",
        "ps_end_doy",
        "peak_doy",
        "ps_length",
        "pollen_season_integral",
        "total_pollen_integral",
        "day_threshold",
    ]
    method_arguments = {
        "percentage": {},
        "moving": {"man": 3, "th_ma": 5},
        "clinical": {
            "n_clinical": 3,
            "window_clinical": 3,
            "th_pollen": 10,
            "th_sum": 30,
        },
        "grains": {"window_grains": 3, "th_pollen": 10},
    }

    for method, arguments in method_arguments.items():
        result = calculate_ps(
            data,
            method=method,
            interpolation=False,
            **arguments,
        )
        extras = ["ppp_start_doy", "ppp_end_doy", "ppp_length", "PPP", "total_ppp_seasons"] if method == "clinical" else []
        assert result.columns.tolist() == expected + extras
