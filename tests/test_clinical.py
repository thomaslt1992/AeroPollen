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


def test_clinical_uses_largest_n_values_for_sum() -> None:
    data = pollen_frame([5.0] * 7)

    result = calculate_ps(
        data,
        method="clinical",
        clinical_pollen_type="grasses",
        interpolation=False,
    )

    assert pd.isna(result.loc[0, "ps_start"])
    assert result.loc[0, "day_threshold"] == 0


def test_grass_alias_matches_grasses_preset() -> None:
    data = pollen_frame([0, 7, 7, 7, 7, 7, 0])

    canonical = calculate_ps(
        data,
        method="clinical",
        clinical_pollen_type="grasses",
        interpolation=False,
    )
    alias = calculate_ps(
        data,
        method="clinical",
        clinical_pollen_type="grass",
        interpolation=False,
    )

    pd.testing.assert_frame_equal(canonical, alias)


def test_manual_clinical_parameters_are_applied_without_preset() -> None:
    data = pollen_frame([0, 10, 10, 10, 10, 10, 0, 0]).rename(columns={"Poaceae": "custom_pollen"})

    result = calculate_ps(
        data,
        method="clinical",
        n_clinical=5,
        window_clinical=7,
        th_pollen=10,
        th_sum=50,
        th_day=10,
        clinical_pollen_type=None,
        interpolation=False,
    )

    assert result.loc[0, "ps_start"] == "2020-01-02"
    assert result.loc[0, "ps_end"] == "2020-01-06"
    assert result.loc[0, "ps_length"] == 5
    assert result.loc[0, "day_threshold"] == 5


def test_preset_intentionally_overrides_manual_parameters() -> None:
    data = pollen_frame([5.0] * 7)

    result = calculate_ps(
        data,
        method="clinical",
        clinical_pollen_type="grasses",
        n_clinical=1,
        window_clinical=1,
        th_pollen=1,
        th_sum=1,
        interpolation=False,
    )

    assert pd.isna(result.loc[0, "ps_start"])


def test_all_missing_year_returns_empty_season() -> None:
    data = pollen_frame([np.nan] * 7)

    result = calculate_ps(data, method="clinical", interpolation=False)

    assert pd.isna(result.loc[0, "ps_start"])
    assert pd.isna(result.loc[0, "peak_date"])
    assert pd.isna(result.loc[0, "total_pollen_integral"])


def test_invalid_clinical_window_is_rejected() -> None:
    data = pollen_frame([10.0] * 7)

    with pytest.raises(ValueError, match="cannot exceed"):
        calculate_ps(
            data,
            method="clinical",
            n_clinical=6,
            window_clinical=5,
        )


@pytest.mark.parametrize(
    ("species", "threshold"),
    [
        ("birch", 100),
        ("grasses", 50),
        ("ragweed", 50),
        ("olive", 100),
        ("cypress", 100),
    ],
)
def test_ppp_uses_yaml_threshold_and_five_consecutive_days(species, threshold):
    data = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=24),
        species: [0] + [threshold] * 4 + [0] + [threshold] * 5
        + [0] * 3 + [threshold] * 6 + [0] * 4,
    })
    result = calculate_ps(
        data,
        method="clinical",
        clinical_pollen_type=species,
        th_day=9999,
        interpolation=False,
    )
    row = result.iloc[0]
    assert row["day_threshold"] == 15
    assert row["ppp_start_doy"] == [7, 15]
    assert row["ppp_end_doy"] == [11, 20]
    assert row["ppp_length"] == [5, 6]
    assert row["PPP"] == 11
    assert row["total_ppp_seasons"] == 2


def test_ppp_auto_resolves_each_species_from_yaml():
    dates = pd.date_range("2020-01-01", periods=15)
    data = pd.DataFrame({
        "Date": dates,
        "birch": [0] + [100] * 10 + [0] * 4,
        "ragweed": [0] + [50] * 10 + [0] * 4,
    })
    result = calculate_ps(data, method="clinical", th_day=5000, interpolation=False)
    assert dict(zip(result["pollen_type"], result["day_threshold"])) == {
        "birch": 10,
        "ragweed": 10,
    }
    assert result["PPP"].tolist() == [10, 10]
