from pathlib import Path

import pandas as pd

from aeropollen import calculate_ps

data_file = (
    Path(__file__).resolve().parents[1] / "aeropollen" / "data" / "munich_pollen.csv"
)
data = pd.read_csv(data_file)

result = calculate_ps(
    data[["date", "Poaceae"]],
    method="moving",
    man=7,
    th_ma=4,
    interpolation=True,
    int_method="linear",
    maxdays=30,
)

print(result.to_string(index=False))
