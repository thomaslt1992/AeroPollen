import pandas as pd
import numpy as np
from aeropollen import calculate_ps

years = range(2015, 2025)  # 10-year test period
dfs = []

# randomly pick 1–2 years to be completely missing
n_missing_years = np.random.choice([1, 2])
missing_years = np.random.choice(list(years), size=n_missing_years, replace=False)

for y in years:
    days = pd.date_range(f"{y}-03-01", f"{y}-09-30")
    n = len(days)

    if y in missing_years:
        # entire year NaN
        df_year = pd.DataFrame({
            "Date": days,
            "Poaceae": np.nan
        })
        dfs.append(df_year)
        continue

    # vary peak timing and strength
    peak_shift = np.random.uniform(-0.3, 0.3)
    intensity = np.random.uniform(10, 50)

    x = np.linspace(-2 + peak_shift, 2 + peak_shift, n)
    curve = np.exp(-0.5 * (x ** 2)) * intensity + np.random.normal(0, 4, n)
    curve = np.clip(curve, 0, None)

    # randomly drop some values (scattered NaNs)
    nan_mode = np.random.choice(["few", "moderate", "many", "none"], p=[0.77, 0.13, 0.09, 0.01])
    if nan_mode == "few":
        nan_fraction = np.random.uniform(0.05, 0.1)
    elif nan_mode == "moderate":
        nan_fraction = np.random.uniform(0.2, 0.4)
    elif nan_mode == "many":
        nan_fraction = np.random.uniform(0.6, 0.9)
    else:
        nan_fraction = 0.0

    nan_idx = np.random.choice(n, size=int(n * nan_fraction), replace=False)
    curve[nan_idx] = np.nan

    df_year = pd.DataFrame({
        "Date": days,
        "Poaceae": curve.astype("float")
    })
    dfs.append(df_year)

# combine all years
df = pd.concat(dfs, ignore_index=True)

# === Run clinical method on long-term series ===
res = calculate_ps(df, method="clinical", clinical_pollen_type="olive")
print(res)
