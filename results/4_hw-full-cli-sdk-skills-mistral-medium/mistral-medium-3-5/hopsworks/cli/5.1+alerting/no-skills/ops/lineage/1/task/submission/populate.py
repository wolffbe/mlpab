import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

rawa_fg = fs.get_feature_group("rawa6913f1", version=1)
rawb_fg = fs.get_feature_group("rawb6913f1", version=1)
derived_fg = fs.get_feature_group("derived6913f1", version=1)

rawa_df = rawa_fg.read()
rawb_df = rawb_fg.read()

merged = pd.merge(rawa_df, rawb_df, on="row_id", how="inner")
merged["col_sum"] = (merged["a_val"] + merged["b_val"]).round(6)
result = merged[["row_id", "col_sum"]]

derived_fg.insert(result)
