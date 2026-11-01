import io
import json

import pandas as pd
import pytest

from aeropollen import calculate_ps


def metadata(path):
    return {
        key: json.loads(value)
        for line in path.read_text().splitlines()
        if line.startswith("# ") and "=" in line
        for key, value in [line[2:].split("=", 1)]
    }


@pytest.mark.parametrize("method", ["clinical", "grains", "moving", "percentage"])
def test_export_preserves_results_and_defaults(tmp_path, method):
    source = tmp_path / "sample.csv"
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=30),
            "Poaceae": [0] * 5 + [100] * 20 + [0] * 5,
        }
    )
    data.to_csv(source, index=False)
    expected = calculate_ps(data, method=method)
    actual = calculate_ps(source, method=method, return_result=True)
    pd.testing.assert_frame_equal(expected, actual)
    output = tmp_path / "sample_ps.csv"
    pd.testing.assert_frame_equal(
        pd.read_csv(output, comment="#"),
        pd.read_csv(io.StringIO(expected.to_csv(index=False))),
    )
    saved = metadata(output)
    assert saved["arguments"]["maxdays"] == 30
    assert saved["arguments"]["method"] == method
    assert saved["pollen_columns"] == ["Poaceae"]
    if method == "clinical":
        assert saved["effective_clinical_parameters"]["Poaceae"]["ppp_threshold"] == 50
        assert (
            saved["effective_clinical_parameters"]["Poaceae"][
                "ppp_min_consecutive_days"
            ]
            == 3
        )


def test_dataframe_export_and_even_window(tmp_path):
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5),
            "pollen": [0, 10, 20, 10, 0],
        }
    )
    source = tmp_path / "custom.csv"
    calculate_ps(data, method="moving", man=4, source_name=source, return_result=True)
    saved = metadata(tmp_path / "custom_ps.csv")
    assert saved["arguments"]["man"] == 4
    assert saved["effective_man"] == 5


def test_export_is_opt_in(tmp_path):
    source = tmp_path / "sample.csv"
    pd.DataFrame({"date": ["2020-01-01"], "pollen": [10]}).to_csv(source, index=False)
    calculate_ps(source)
    assert not (tmp_path / "sample_ps.csv").exists()


@pytest.mark.parametrize("method", ["clinical", "grains", "moving", "percentage"])
def test_dataframe_default_filename(tmp_path, monkeypatch, method):
    monkeypatch.chdir(tmp_path)
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=30),
            "Poaceae": [0] * 5 + [100] * 20 + [0] * 5,
        }
    )
    expected = calculate_ps(data, method=method)
    actual = calculate_ps(data, method=method, return_result=True)
    pd.testing.assert_frame_equal(expected, actual)
    output = tmp_path / "df_ps.csv"
    pd.testing.assert_frame_equal(
        pd.read_csv(output, comment="#"),
        pd.read_csv(io.StringIO(expected.to_csv(index=False))),
    )
    assert metadata(output)["source_name"] == "df.csv"
