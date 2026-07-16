"""Ingest accounts batches into feature group accountsd00439 (latest revision per row_id)."""
import hopsworks
import pandas as pd

project = hopsworks.login()
dataset_api = project.get_dataset_api()

frames = []
for i in (1, 2, 3):
    local = dataset_api.download(
        f"Resources/accountsd00439/batch_{i}.csv", overwrite=True
    )
    frames.append(pd.read_csv(local))

df = pd.concat(frames, ignore_index=True)
print(f"Total revisions read: {len(df)}")

# Keep only the latest revision per row_id (stable sort; last wins).
df = (
    df.sort_values("updated_at", kind="mergesort")
    .drop_duplicates(subset="row_id", keep="last")
    .reset_index(drop=True)
)
print(f"Rows after dedup: {len(df)} (unique row_ids: {df['row_id'].nunique()})")

df["row_id"] = df["row_id"].astype(str)
df["status"] = df["status"].astype(str)
df["balance"] = df["balance"].astype("float64")
df["updated_at"] = df["updated_at"].astype("int64")

fs = project.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="accountsd00439",
    version=1,
    description="Accounts table — latest revision per row_id",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,
)
fg.insert(df, wait=True)
print("Insert complete:", len(df), "rows")
