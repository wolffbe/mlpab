"""Platform job: create rawa8af783, rawb8af783 and derived derived8af783 with lineage."""
import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

local_a = dataset_api.download("Resources/lineage8af783/raw_a.csv", overwrite=True)
local_b = dataset_api.download("Resources/lineage8af783/raw_b.csv", overwrite=True)
df_a = pd.read_csv(local_a)
df_b = pd.read_csv(local_b)
print("raw_a rows:", len(df_a), "raw_b rows:", len(df_b))

fg_a = fs.get_or_create_feature_group(
    name="rawa8af783",
    version=1,
    primary_key=["row_id"],
    description="Raw table A loaded from raw_a.csv",
    online_enabled=False,
)
fg_a.insert(df_a, wait=True)
print("rawa8af783 v1 inserted")

fg_b = fs.get_or_create_feature_group(
    name="rawb8af783",
    version=1,
    primary_key=["row_id"],
    description="Raw table B loaded from raw_b.csv",
    online_enabled=False,
)
fg_b.insert(df_b, wait=True)
print("rawb8af783 v1 inserted")

df_d = df_a.merge(df_b, on="row_id", how="inner")
df_d["col_sum"] = (df_d["a_val"] + df_d["b_val"]).round(6)
df_d = df_d[["row_id", "col_sum"]]
print("derived rows:", len(df_d))

fg_d = fs.get_or_create_feature_group(
    name="derived8af783",
    version=1,
    primary_key=["row_id"],
    description=(
        "Derived from rawa8af783 v1 and rawb8af783 v1: inner join on row_id, "
        "col_sum = round(a_val + b_val, 6)"
    ),
    online_enabled=True,
    parents=[fg_a, fg_b],
)
fg_d.insert(df_d, wait=True)
print("derived8af783 v1 inserted (online enabled)")
print("DONE")
