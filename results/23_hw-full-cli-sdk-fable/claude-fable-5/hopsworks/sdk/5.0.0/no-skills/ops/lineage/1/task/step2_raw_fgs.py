import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

df_a = pd.read_csv("data/raw_a.csv")
df_b = pd.read_csv("data/raw_b.csv")
print("raw_a:", df_a.shape, "raw_b:", df_b.shape)

fg_a = fs.get_or_create_feature_group(
    name="rawa8af783",
    version=1,
    description="Raw table A loaded from raw_a.csv",
    primary_key=["row_id"],
    online_enabled=False,
)

# upsert on primary key: safe to re-insert even if partially loaded before
fg_a.insert(df_a, wait=True)
print("rawa8af783 inserted")

fg_b = fs.get_or_create_feature_group(
    name="rawb8af783",
    version=1,
    description="Raw table B loaded from raw_b.csv",
    primary_key=["row_id"],
    online_enabled=False,
)
fg_b.insert(df_b, wait=True)
print("rawb8af783 inserted")
