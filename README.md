# AeroPollen

AeroPollen is a Python package for estimating pollen-season characteristics from daily airborne pollen concentrations. It implements four season-definition methods, drawing on the [AeRobiology](https://cran.r-project.org/package=AeRobiology) R package and published aerobiological definitions.

The package is under development. Methods and outputs may change.

## Installation

Python 3.10 or newer is required.

```bash
git clone --branch refactor_code https://github.com/thomaslt1992/AeroPollen.git
cd AeroPollen
python -m pip install -e .
```

Input data must contain a date column and at least one pollen taxon column with daily concentrations (grains/m³).

## Notebook usage

```python
import pandas as pd
from aeropollen import calculate_ps

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

clinical.head()
```

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
| **Grains** | Starts with the first qualifying run and ends with the last, requiring a specified number of consecutive days above a concentration threshold. |
| **Clinical** | Uses a species-specific 5-of-7-day rule with minimum daily concentrations and a cumulative threshold, based on EAACI definitions. |

The clinical method additionally returns Peak Pollen Period (PPP) intervals within the identified season: `ppp_start_doy`, `ppp_end_doy`, `ppp_length`, `PPP` (total days), and `total_ppp_seasons`. The present implementation requires **five consecutive days** at or above the species-specific `day_threshold` defined in the YAML file. This duration is a package-specific choice; the published EAACI PPP definition uses **three consecutive days**.

All methods return the same season-summary columns, while clinical includes these five additional PPP fields. The output column `day_threshold` reports the *number of days* meeting or exceeding the applicable concentration threshold, not the threshold concentration itself.

## References

1. Pfaar O, et al. (2017). Defining pollen exposure times for clinical trials of allergen immunotherapy for pollen-induced rhinoconjunctivitis: an EAACI position paper. *Allergy*, 72, 713–722. [doi:10.1111/all.13092](https://doi.org/10.1111/all.13092).
2. [AeRobiology R package](https://cran.r-project.org/package=AeRobiology), including the [`calculate_ps` documentation](https://search.r-project.org/CRAN/refmans/AeRobiology/html/calculate_ps.html). Describes the percentage, moving, grains, and clinical season definitions, with underlying methodological references.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Data and license

Example pollen data originate from AeRobiology; see [data attribution](aeropollen/data/README.md). AeroPollen source code is released under the [Unlicense](LICENSE). Example datasets retain their upstream licensing terms.
