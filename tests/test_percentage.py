import pandas as pd

from aeropollen import calculate_ps


def test_percentage_uses_equal_tails_and_reports_threshold_days() -> None:
    data = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "Poaceae": [1, 1, 96, 1, 1],
        }
    )

    result = calculate_ps(
        data,
        method="percentage",
        perc=96,
        th_day=50,
        interpolation=False,
    )

    assert result.loc[0, "ps_start"] == "2020-01-02"
    assert result.loc[0, "ps_end"] == "2020-01-04"
    assert result.loc[0, "pollen_season_integral"] == 98
    assert result.loc[0, "day_threshold"] == 1
