"""Verify job 2: online-store read + direct offline parquet check."""
import glob

import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("predictions646af0", version=1)

online_df = fg.read(online=True)
print("ONLINE rows:", len(online_df))
print(online_df.sort_values("row_id").head(3).to_string(index=False))

files = glob.glob(
    "/hopsfs/apps/hive/warehouse/mlpab2ff3f2_featurestore.db/predictions646af0_1/*.parquet"
)
print("OFFLINE parquet files:", files)
total = 0
for f in files:
    df = pd.read_parquet(f)
    total += len(df)
    print(f, "rows:", len(df), "cols:", list(df.columns))
print("OFFLINE total rows:", total)
