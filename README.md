# AeroPollen

AeroPollen is a Python package for estimating pollen-season characteristics from daily airborne pollen concentrations. It implements four season-definition methods, drawing on the [AeRobiology](https://cran.r-project.org/package=AeRobiology) R package and published aerobiological definitions.

The package is under development. Methods and outputs may change.

## Installation

Python 3.10 or newer is required.

```bash
git clone --branch main https://github.com/thomaslt1992/AeroPollen.git
cd AeroPollen
python -m pip install -e .
```

Input data must contain a date column and at least one pollen taxon column with daily concentrations (grains/m³).

## Notebook usage

```python
import pandas as pd
from aeropollen import calculate_ps
from IPython.display import display

data = pd.read_csv("aeropollen/data/munich_pollen.csv")

clinical = calculate_ps(
    data[["date", "Betula"]],
    method="clinical",
    clinical_pollen_type="birch",
)

percentage = calculate_ps(
    data[["date", "Poaceae"]],
    method="percentage",
    perc=95,
)

grains = calculate_ps(
    data[["date", "Poaceae"]],
    method="grains",
    window_grains=5,
    th_pollen=10,
)

moving = calculate_ps(
    data[["date", "Poaceae"]],
    method="moving",
    man=11,
    th_ma=5,
)

display(clinical)
display(percentage)
display(grains)
display(moving)
```

Run this notebook from the repository root after `git pull origin main`. Restart the kernel if AeroPollen was already imported. The example uses the bundled Munich data (2010–2015).

The default interpolation is linear; use `interpolation=False` to retain missing values. Clinical presets are available for `birch`, `grasses`, `ragweed`, `olive`, and `cypress`. Recognized taxon names automatically select their corresponding YAML settings in `aeropollen/config/clinical_thresholds.yaml`. For custom clinical thresholds, use a taxon name without a matching preset.

## Command-line usage

Following installation, use `aeropollen`, `python -m aeropollen`, or `python main.py` from the repository root.

```bash
aeropollen --input aeropollen/data/munich_pollen.csv --method clinical --species Betula --clinical-pollen-type birch --output birch_season.csv

aeropollen --input aeropollen/data/munich_pollen.csv --method percentage --species Poaceae --perc 95 --output grasses_percentage.csv

aeropollen --input aeropollen/data/munich_pollen.csv --method grains --species Poaceae --window-grains 5 --th-pollen 10 --output grasses_grains.csv

aeropollen --input aeropollen/data/munich_pollen.csv --method moving --species Poaceae --man 11 --th-ma 5 --output grasses_moving.csv
```

Omit `--output` to print CSV results in the terminal and omit `--species` to process all available pollen columns. Use `aeropollen --help` for the full argument list.

## Methods

| Method | Season definition |
| --- | --- |
| **Percentage** | Contains the specified fraction of the annual pollen integral. At 95%, 2.5% of the cumulative integral is excluded at each end. |
| **Moving** | Uses a centered moving average; season boundaries surround its maximum and are defined by the moving-average threshold. |
| **Grains** | Starts with the first qualifying run and ends with the last, requiring a specified number of retained observations at or above a concentration threshold. |
| **Clinical** | Uses a species-specific 5-of-7-day rule with minimum daily concentrations and a cumulative threshold, based on EAACI definitions. |

The clinical method additionally returns Peak Pollen Period (PPP) intervals within the identified season: `ppp_start_doy`, `ppp_end_doy`, `ppp_length`, `PPP` (total days), and `total_ppp_seasons`. The implementation requires **three consecutive days** at or above the species-specific `day_threshold` defined in the YAML file.

All methods return the same season-summary columns, while clinical includes these five additional PPP fields. The output column `day_threshold` reports the *number of days* meeting or exceeding the applicable concentration threshold, not the threshold concentration itself.

The following additional columns are available for all methods:

| Column | Meaning |
| --- | --- |
| `peak_concentration` | Maximum daily concentration in the year. |
| `days_from_season_start_to_peak` | Days from season start through the peak day, inclusive. |
| `pollen_integral_through_peak` | Pollen total from season start through the peak day. |
| `days_through_season_end` | Days after the peak day through season end. |
| `pollen_integral_after_peak` | Pollen total after the peak day through season end. |

The first annual maximum defines the peak when values tie. Clinical leaves period summaries missing when no season is identified or the annual peak falls outside that season. An available annual peak concentration is still returned.

### Moving, percentage and grains compatibility

These methods follow the AeRobiology `calculate_ps.R` and `ma.R` calculations for calendar years (`def_season="natural"`). Other season-grouping options raise an error; they are not silently treated as calendar years. Linear interpolation (`linear` or `lineal`; `time` is equivalent on the daily grid) fills only interior gaps of at most `maxdays` days. Clinical preprocessing is unchanged.

- Moving removes missing observations before reconstructing the daily series between its first and last observations, with zeros in unresolved internal gaps. It uses the first smoothed maximum and strict below-threshold crossings. A missing crossing does not fall back to the first or last day.
- Grains removes unresolved missing observations before finding runs. It starts at the first qualifying run and ends at the last, using `window_grains` retained observations at or above `th_pollen`. With complete daily data these are consecutive days; with gaps a run can span more calendar days.
- Percentage uses a forward cumulative sum for the start and a reverse cumulative sum for the end. The end is the calendar day before the first reverse sum strictly below the tail threshold. An absent end boundary remains missing, including at `perc=100`. Values from 0 to 100 are accepted.
- Annual statistics remain available when a boundary is undefined. Following R's summary assignment order, a missing start leaves both season boundaries missing; a missing end retains an available start. Dependent summaries and the high-day count remain missing. All-zero or all-missing years have missing statistics.
- The pre/post-peak sums reproduce R's inclusive index ranges, including descending ranges. In particular, if the peak equals season end, R's `(peak + 1):end` includes the peak and following observation, rather than an empty interval. Thus `pollen_integral_after_peak` can be nonzero when `days_through_season_end` is zero, or missing when that following observation is absent. A peak outside the selected season can also give signed durations. These quirks apply only to moving, percentage and grains for numerical compatibility.

The regression tests cover these source-defined cases. This implementation has been checked against translated R formulas; it has not been validated by executing the R package in this environment.

## References

1. Pfaar O, et al. (2017). Defining pollen exposure times for clinical trials of allergen immunotherapy for pollen-induced rhinoconjunctivitis: an EAACI position paper. *Allergy*, 72, 713–722. [doi:10.1111/all.13092](https://doi.org/10.1111/all.13092).
2. [AeRobiology R package](https://cran.r-project.org/package=AeRobiology), including the [`calculate_ps` documentation](https://search.r-project.org/CRAN/refmans/AeRobiology/html/calculate_ps.html). Describes the percentage, moving, grains, and clinical season definitions, with underlying methodological references.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

### Save results with calculation parameters

All four methods support `return_result=True`. Pass a CSV path to save
`<input_name>_ps.csv` beside the input file:

```python
from aeropollen import calculate_ps

results = calculate_ps(
    "aeropollen/data/munich_pollen.csv",
    method="grains",
    return_result=True,
)
```

For an existing DataFrame, supply the source filename explicitly:

```python
results = calculate_ps(
    data[["date", "Betula", "Poaceae"]],
    method="clinical",
    source_name="munich_pollen.csv",
    return_result=True,
)
```

The function still returns its usual notebook result. The CSV contains the
result table followed by comment-prefixed parameter rows with JSON values.
These record every supplied or default argument, the selected pollen columns,
resolved clinical thresholds (including PPP), and the effective odd moving
window. Unused compatibility arguments are identified separately.
Read only the result table with `pd.read_csv(path, comment="#")`.

Export is off by default; `export_result=True` is also accepted for compatibility.
Repeating an export to the same source name replaces its `_ps.csv` file.
