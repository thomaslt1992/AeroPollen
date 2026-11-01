import pandas as pd
import pytest

from aeropollen.cli import main


@pytest.mark.parametrize("method", ["percentage", "moving", "grains", "clinical"])
def test_cli_outputs_expected_schema(tmp_path, method):
    source = tmp_path / "daily.csv"
    destination = tmp_path / "output.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=20),
            "Betula": [0] * 5 + [110] * 10 + [0] * 5,
        }
    ).to_csv(source, index=False)
    assert (
        main(
            [
                "--input",
                str(source),
                "--output",
                str(destination),
                "--method",
                method,
                "--species",
                "Betula",
                "--no-interpolation",
            ]
        )
        == 0
    )
    result = pd.read_csv(destination)
    assert result["pollen_type"].tolist() == ["Betula"]
    assert "day_threshold" in result.columns
    assert "days_above_threshold" not in result.columns
    assert ("ppp_start_doy" in result.columns) == (method == "clinical")
    assert "Year" not in result.columns


def test_cli_missing_species_fails(tmp_path):
    source = tmp_path / "daily.csv"
    pd.DataFrame({"date": ["2020-01-01"], "birch": [1]}).to_csv(source, index=False)
    with pytest.raises(SystemExit) as exc:
        main(["--input", str(source), "--species", "unknown"])
    assert exc.value.code == 2
