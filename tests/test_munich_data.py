from pathlib import Path

import pandas as pd

from aeropollen import calculate_ps

DATA_FILE = (
    Path(__file__).resolve().parents[1] / "aeropollen" / "data" / "munich_pollen.csv"
)


def test_munich_clinical_presets_return_each_year() -> None:
    data = pd.read_csv(DATA_FILE)

    birch = calculate_ps(
        data[["date", "Betula"]],
        method="clinical",
        clinical_pollen_type="birch",
    )
    grasses = calculate_ps(
        data[["date", "Poaceae"]],
        method="clinical",
        clinical_pollen_type="grasses",
    )

    assert birch["season"].tolist() == list(range(2010, 2016))
    assert grasses["season"].tolist() == list(range(2010, 2016))
    assert birch["ps_start"].tolist() == [
        "2010-03-31",
        "2011-04-03",
        "2012-03-29",
        "2013-04-17",
        "2014-03-26",
        "2015-04-11",
    ]
    assert grasses["ps_start"].tolist() == [
        "2010-05-22",
        "2011-04-22",
        "2012-04-29",
        "2013-05-12",
        "2014-04-30",
        "2015-05-05",
    ]
    assert grasses["ps_end"].tolist() == [
        "2010-08-02",
        "2011-08-14",
        "2012-08-02",
        "2013-09-17",
        "2014-08-05",
        "2015-08-14",
    ]
    assert birch["day_threshold"].eq(100).all()
    assert grasses["day_threshold"].eq(50).all()


def test_munich_grains_matches_consecutive_day_definition() -> None:
    data = pd.read_csv(DATA_FILE)

    result = calculate_ps(
        data[["date", "Poaceae"]],
        method="grains",
        window_grains=3,
        th_pollen=2,
    )

    assert result["season"].tolist() == list(range(2010, 2016))
    assert result["ps_start"].tolist() == [
        "2010-05-08",
        "2011-04-22",
        "2012-05-01",
        "2013-05-14",
        "2014-04-24",
        "2015-05-05",
    ]
    assert result["ps_end"].tolist() == [
        "2010-09-08",
        "2011-09-12",
        "2012-09-06",
        "2013-09-17",
        "2014-08-12",
        "2015-09-01",
    ]
