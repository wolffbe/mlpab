import csv
import json
import os

# Read and validate the CSV
VALID_CATEGORIES = {"grocery", "travel", "salary", "rent", "other"}

rows = []
rejected = []
valid_rows = []

with open("data/events.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)
        row_id = row["row_id"]
        amount_str = row.get("amount", "").strip()
        category = row.get("category", "").strip()

        # Rule 1: amount must be present (non-null, non-empty)
        if not amount_str:
            rejected.append(row_id)
            continue

        # Rule 2: amount within [0, 10000]
        try:
            amount = float(amount_str)
        except ValueError:
            rejected.append(row_id)
            continue

        if amount < 0 or amount > 10000:
            rejected.append(row_id)
            continue

        # Rule 3: category must be valid
        if category not in VALID_CATEGORIES:
            rejected.append(row_id)
            continue

        valid_rows.append(row)

print(f"Total rows: {len(rows)}")
print(f"Valid rows: {len(valid_rows)}")
print(f"Rejected rows: {len(rejected)}")
print(f"Rejected IDs: {rejected}")

# Connect to Hopsworks
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

# Build a pandas DataFrame from valid rows
import pandas as pd

df = pd.DataFrame(valid_rows)
# Cast types
df["event_time"] = df["event_time"].astype("int64")
df["amount"] = df["amount"].astype("float64")

print(df.dtypes)
print(df.head())

# Create the feature group
fg = fs.get_or_create_feature_group(
    name="events2cfe18",
    version=1,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Events feature table v1",
)

fg.insert(df)
print("Insert complete")

# Write submission/answers.json
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"rejected": rejected}, f)

print("Done. answers.json written.")
