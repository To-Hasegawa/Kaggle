import pandas as pd

df = pd.read_csv("***.csv")

import pandas_profiling as pdp

pdp.ProfileReport(df)