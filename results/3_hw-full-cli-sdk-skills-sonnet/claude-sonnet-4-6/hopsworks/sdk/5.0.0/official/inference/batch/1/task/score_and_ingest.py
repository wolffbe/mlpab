import csv
import json
import math
import hopsworks

T = 1773219600000

with open("data/feature_history.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# For each account, find the most recent revision at or before T
accounts = {}
for row in rows:
    acct = row["account_id"]
    et = int(row["event_time"])
    if et <= T:
        if acct not in accounts or et > accounts[acct]["event_time"]:
            accounts[acct] = {
                "event_time": et,
                "f1": float(row["f1"]),
                "f2": float(row["f2"]),
                "f3": float(row["f3"]),
            }

with open("data/model.json") as f:
    model = json.load(f)

w_f1 = model["weights"]["f1"]
w_f2 = model["weights"]["f2"]
w_f3 = model["weights"]["f3"]
bias = model["bias"]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

records = []
for acct, feat in sorted(accounts.items()):
    z = w_f1 * feat["f1"] + w_f2 * feat["f2"] + w_f3 * feat["f3"] + bias
    score = round(sigmoid(z), 6)
    records.append({"account_id": acct, "score": score})

print(f"Total accounts to score: {len(records)}")
print("Sample:", records[:3])

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

import pandas as pd
df = pd.DataFrame(records)
print(df.dtypes)
print(df.head())

# Create feature group
fg = fs.get_or_create_feature_group(
    name="scores43f1c2",
    version=1,
    primary_key=["account_id"],
    online_enabled=True,
    description="Batch scores as of T=1773219600000",
)

fg.insert(df)
print("Insert complete.")
print("Feature group:", fg.name, "version:", fg.version)
