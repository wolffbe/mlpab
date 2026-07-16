import json
import math
import pandas as pd
import hopsworks
from hsfs.feature import Feature

ALLOWED = {"grocery", "travel", "salary", "rent", "other"}

# Read everything as string first so we can detect empty/missing amounts reliably.
raw = pd.read_csv("data/events.csv", dtype=str, keep_default_na=False)
print("total rows:", len(raw))

valid_rows = []
rejected = []

for _, r in raw.iterrows():
    row_id = r["row_id"]
    amount_str = r["amount"].strip()
    category = r["category"].strip()

    ok = True
    # Rule 1: amount present
    if amount_str == "":
        ok = False
    else:
        try:
            amt = float(amount_str)
        except ValueError:
            amt = None
            ok = False
        if amt is not None and (math.isnan(amt) or math.isinf(amt)):
            ok = False
        # Rule 2: amount within [0, 10000] inclusive
        elif amt is not None and not (0.0 <= amt <= 10000.0):
            ok = False
    # Rule 3: category allowed
    if category not in ALLOWED:
        ok = False

    if ok:
        valid_rows.append({
            "row_id": row_id,
            "account_id": r["account_id"].strip(),
            "event_time": int(r["event_time"]),
            "amount": float(amount_str),
            "category": category,
        })
    else:
        rejected.append(row_id)

print("valid:", len(valid_rows), "rejected:", len(rejected))

df = pd.DataFrame(valid_rows, columns=["row_id", "account_id", "event_time", "amount", "category"])
df["event_time"] = df["event_time"].astype("int64")
df["amount"] = df["amount"].astype("float64")
print(df.dtypes)
print(df.head())

# Write answers file
with open("submission/answers.json", "w") as f:
    json.dump({"rejected": rejected}, f)
print("wrote submission/answers.json")

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="events5b591e",
    version=1,
    description="Events that satisfy the data contract (amount present & in [0,10000], category in allowed set).",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    stream=True,
    features=[
        Feature("row_id", "string", description="Unique record key"),
        Feature("account_id", "string", description="Account identifier"),
        Feature("event_time", "bigint", description="Event time, epoch milliseconds"),
        Feature("amount", "double", description="Transaction amount"),
        Feature("category", "string", description="Event category"),
    ],
)

fg.insert(df, wait=True)
print("insert complete. fg.id =", fg.id)
