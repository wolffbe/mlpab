import pandas as pd
import hopsworks

# Read both exports and present each record key exactly once.
df1 = pd.read_csv("data/transactions_export_1.csv")
df2 = pd.read_csv("data/transactions_export_2.csv")
combined = pd.concat([df1, df2], ignore_index=True)
# Each row_id is the unique record key; overlapping rows are identical re-deliveries.
deduped = combined.drop_duplicates(subset="row_id", keep="last").reset_index(drop=True)
print("Combined:", len(combined), "-> deduped (one per row_id):", len(deduped))

deduped["row_id"] = deduped["row_id"].astype(str)
deduped["account_id"] = deduped["account_id"].astype(str)
deduped["event_time"] = deduped["event_time"].astype("int64")
deduped["amount"] = deduped["amount"].astype("float64")
deduped["category"] = deduped["category"].astype(str)

proj = hopsworks.login()
fs = proj.get_feature_store()

# Drop the previous version that contained duplicate rows.
try:
    old = fs.get_feature_group("transactions3cd0a6", version=1)
    if old is not None:
        old.delete()
        print("Deleted previous feature group with duplicate rows.")
except Exception as e:
    print("No previous FG to delete or delete failed:", e)

fg = fs.get_or_create_feature_group(
    name="transactions3cd0a6",
    version=1,
    description="Transactions table ingested from two overlapping exports (deduped on row_id)",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)

fg.insert(deduped)
print("Inserted", len(deduped), "rows. online_enabled:", fg.online_enabled, "id:", fg.id)
