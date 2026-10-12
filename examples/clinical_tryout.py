from pathlib import Path

import pandas as pd

from aeropollen import calculate_ps

data_file = (
    Path(__file__).resolve().parents[1] / "aeropollen" / "data" / "munich_pollen.csv"
)
data = pd.read_csv(data_file)

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

print(pd.concat([birch, grasses], ignore_index=True).to_string(index=False))
