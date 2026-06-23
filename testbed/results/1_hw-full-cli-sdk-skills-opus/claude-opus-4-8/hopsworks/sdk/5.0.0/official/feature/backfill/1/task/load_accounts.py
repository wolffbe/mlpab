import pandas as pd
import hopsworks
from hsfs.feature import Feature

# 1. Read the three out-of-order batches and combine
frames = [pd.read_csv(f"data/batch_{i}.csv") for i in (1, 2, 3)]
df = pd.concat(frames, ignore_index=True)
df["updated_at"] = df["updated_at"].astype("int64")
print("combined rows:", len(df), "distinct row_id:", df["row_id"].nunique())

# 2. Keep only each row_id's LATEST revision (max updated_at) -> exactly one row per row_id
df = (
    df.sort_values("updated_at")
      .drop_duplicates(subset="row_id", keep="last")
      .reset_index(drop=True)
)
print("after dedup rows:", len(df))
assert df["row_id"].is_unique, "row_id not unique after dedup"

# 3. Connect to the platform
project = hopsworks.login()
fs = project.get_feature_store()

# 4. Register the feature group: online-enabled, record key row_id, event-time updated_at
fg = fs.get_or_create_feature_group(
    name="accounts65d53c",
    version=1,
    description="Accounts table: latest revision per row_id, backfilled from three out-of-order batches.",
    primary_key=["row_id"],
    event_time="updated_at",
    features=[
        Feature("row_id", "string", description="Unique account record key"),
        Feature("status", "string", description="Account status: active | dormant | closed"),
        Feature("balance", "double", description="Account balance"),
        Feature("updated_at", "bigint", description="Revision time, epoch milliseconds (event time)"),
    ],
    online_enabled=True,   # low-latency / real-time lookups
    stream=True,           # required with online_enabled
)

# 5. Insert the deduped data; block until online + offline ingestion complete
fg.insert(df, wait=True)
print("inserted. fg.id =", fg.id)
