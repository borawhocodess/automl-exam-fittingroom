# %%
# IMPORTS

from fittingroom.utils import print_datasets_overview

# %%
# DATASETS OVERVIEW
print_datasets_overview("../data")

# %%
# see brazilian houses parquet

import pandas as pd

df = pd.read_parquet("../data/brazilian_houses/1/X_test.parquet")
print(df.head())
# %%
