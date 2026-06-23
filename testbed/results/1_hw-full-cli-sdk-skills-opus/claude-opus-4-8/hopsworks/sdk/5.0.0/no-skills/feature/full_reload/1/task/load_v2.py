import hopsworks
import pandas as pd

proj = hopsworks.login()
fs = proj.get_feature_store()

df = pd.read_csv("data/reload/new_export.csv", dtype={"row_id": str, "full_name": str, "currency": str})
df["updated_at"] = df["updated_at"].astype("int64")
df["balance"] = df["balance"].astype("float64")
print("V2 shape:", df.shape, "cols:", list(df.columns))
print(df.dtypes)

fg2 = fs.create_feature_group(
    name="customersa8deb9",
    version=2,
    description="Customers full re-export (new breaking schema)",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,
)
fg2.insert(df, write_options={"wait_for_job": True})
print("V2 inserted")
