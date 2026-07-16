import pandas as pd
import hopsworks

# --- Load the three batches (out of order); concat only, no transform ---
frames = [pd.read_csv(f"data/batch_{i}.csv") for i in (1, 2, 3)]
df = pd.concat(frames, ignore_index=True)
df["row_id"] = df["row_id"].astype(str)
df["updated_at"] = df["updated_at"].astype("int64")
print("total rows loaded:", len(df), "| distinct row_id:", df["row_id"].nunique())

proj = hopsworks.login()
fs = proj.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="accounts65d53c",
    version=1,
    description="accounts table, latest revision per row_id",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,          # low-latency / online lookup
    time_travel_format="HUDI",
    hudi_precombine_key="updated_at",  # latest updated_at wins on duplicate keys
    stream=True,
)

# Single commit -> Hudi precombine keeps the row with max(updated_at) per row_id
job, _ = fg.insert(df, operation="upsert", wait=True)
print("insert complete")
