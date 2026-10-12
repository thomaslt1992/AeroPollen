# AeroPollen

AeroPollen is an early-stage Python package for estimating pollen-season
characteristics from daily pollen observations. It is inspired by the
[AeRobiology](https://cran.r-project.org/package=AeRobiology) R package while
providing a smaller Python-focused API.

> [!WARNING]
> AeroPollen is under active development. Its API and results may change before
> the first stable release.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install git+https://github.com/thomaslt1992/AeroPollen.git@refactor_code
```

For development:

```bash
git clone --branch refactor_code --single-branch https://github.com/thomaslt1992/AeroPollen.git
cd AeroPollen
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Clinical method

The available clinical presets are `birch`, `grasses`, `cypress`, `olive`, and
`ragweed`. `grass` remains accepted as an alias for `grasses`.

```python
import pandas as pd

from aeropollen import calculate_ps

data = pd.read_csv("aeropollen/data/munich_pollen.csv")
result = calculate_ps(
    data[["date", "Poaceae"]],
    method="clinical",
    clinical_pollen_type="grasses",
    interpolation=True,
    int_method="linear",
    maxdays=30,
)

print(result)
```

Selecting a preset applies its predefined `n_clinical`, `window_clinical`,
`th_pollen`, `th_sum`, and `th_day` values. To use custom values, omit
`clinical_pollen_type`:

```python
result = calculate_ps(
    data[["date", "Poaceae"]],
    method="clinical",
    n_clinical=5,
    window_clinical=7,
    th_pollen=3,
    th_sum=30,
    th_day=50,
)
```

Run the included example with:

```bash
python examples/clinical_tryout.py
```

## Development checks

```bash
python -m pytest
python -m build
python -m twine check --strict dist/*
```

## Data attribution

The example CSV files were obtained from AeRobiology 2.0.4. Their upstream
sources include the Zentrum Allergie und Umwelt and the Bavarian State Office
for Health and Food Safety. The example data are redistributed under GPL-3;
see `aeropollen/data/README.md`.

## License

AeroPollen source code is released into the public domain under the
[Unlicense](LICENSE). The example datasets retain their upstream GPL-3 terms.
