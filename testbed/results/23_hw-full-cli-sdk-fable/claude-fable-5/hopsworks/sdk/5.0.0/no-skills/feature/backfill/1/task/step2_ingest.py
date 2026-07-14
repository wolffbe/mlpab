import os

# route platform traffic through the localhost proxy (sandbox only allows localhost)
for _v in ("NO_PROXY", "no_proxy"):
    os.environ.pop(_v, None)

import pandas as pd

import hopsworks

# combine the three out-of-order batches; latest updated_at per row_id wins
batches = [pd.read_csv(f"data/batch_{i}.csv") for i in (1, 2, 3)]
df = pd.concat(batches, ignore_index=True)
df["updated_at"] = df["updated_at"].astype("int64")
df["balance"] = df["balance"].astype("float64")
df = df.sort_values("updated_at").drop_duplicates(subset="row_id", keep="last")
df = df.reset_index(drop=True)
print(f"total rows across batches: {sum(len(b) for b in batches)}")
print(f"unique row_ids (latest revision kept): {len(df)}")

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="accountsd00439",
    version=1,
    description="Accounts table, latest revision per row_id",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,
)

job, _ = fg.insert(df, wait=True)
print("insert done")
print("feature group:", fg.name, "version:", fg.version)
