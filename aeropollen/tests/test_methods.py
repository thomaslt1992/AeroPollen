import pandas as pd
import numpy as np
from aeropollen import calculate_ps

# fake pollen data
df = pd.DataFrame({
    "Date": pd.date_range("2023-03-01", "2023-06-30"),
    "Poaceae": np.random.randint(0, 100, 122),
    "Betula": np.random.randint(0, 80, 122)
})

res = calculate_ps(df, method="clinical", interpolation=True, int_method="linear", n_clinical=5, window_clinical=7, th_pollen=10.0, th_sum=100.0, th_day=0.001)
print(res)