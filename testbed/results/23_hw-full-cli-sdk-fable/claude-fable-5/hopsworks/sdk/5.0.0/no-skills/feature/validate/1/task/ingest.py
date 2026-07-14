import hopsworks
import csv
import json
import os

VALID_CATEGORIES = {"grocery", "travel", "salary", "rent", "other"}

rows = []
rejected = []
with open("data/events.csv", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        amount_raw = (r["amount"] or "").strip()
        category = r["category"] if r["category"] is not None else ""
        ok = True
        if amount_raw == "":
            ok = False
        else:
            try:
                amount = float(amount_raw)
                if not (0.0 <= amount <= 10000.0):
                    ok = False
            except ValueError:
                ok = False
        if category not in VALID_CATEGORIES:
            ok = False
        if ok:
            rows.append(
                {
                    "row_id": r["row_id"],
                    "account_id": r["account_id"],
                    "event_time": int(r["event_time"]),
                    "amount": float(amount_raw),
                    "category": category,
                }
            )
        else:
            rejected.append(r["row_id"])

print(f"total={len(rows) + len(rejected)} valid={len(rows)} rejected={len(rejected)}")

import pandas as pd

df = pd.DataFrame(rows)
df["row_id"] = df["row_id"].astype(str)
df["account_id"] = df["account_id"].astype(str)
df["event_time"] = df["event_time"].astype("int64")
df["amount"] = df["amount"].astype("float64")
df["category"] = df["category"].astype(str)
print(df.dtypes)

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="eventsee881b",
    version=1,
    description="Contract-valid events export",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)
job, validation = fg.insert(df, wait=True)
print("insert done:", job)

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"rejected": rejected}, f, indent=2)
print("wrote submission/answers.json with", len(rejected), "rejected ids")
