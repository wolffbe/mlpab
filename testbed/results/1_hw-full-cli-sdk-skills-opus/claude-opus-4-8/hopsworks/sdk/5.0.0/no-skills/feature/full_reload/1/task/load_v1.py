import hopsworks
import pandas as pd

proj = hopsworks.login()
fs = proj.get_feature_store()

df = pd.read_csv("data/initial_export.csv", dtype={"row_id": str})
df["updated_at"] = df["updated_at"].astype("int64")
print("V1 shape:", df.shape, "cols:", list(df.columns))

fg1 = fs.create_feature_group(
    name="customersa8deb9",
    version=1,
    description="Customers initial export (original schema)",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=False,
)
fg1.insert(df, write_options={"wait_for_job": True})
print("V1 inserted")
