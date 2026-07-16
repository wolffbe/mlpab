import csv
import json
import os

# --- 1. Load and validate CSV ---
data_path = "data/events.csv"
valid_categories = {"grocery", "travel", "salary", "rent", "other"}

valid_rows = []
rejected_ids = []

with open(data_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        row_id = row["row_id"]
        amount_raw = row.get("amount", "").strip()
        category = row.get("category", "").strip()

        # Rule 1: amount present
        if not amount_raw:
            rejected_ids.append(row_id)
            continue

        # Rule 2: amount in [0, 10000]
        try:
            amount = float(amount_raw)
        except ValueError:
            rejected_ids.append(row_id)
            continue

        if amount < 0 or amount > 10000:
            rejected_ids.append(row_id)
            continue

        # Rule 3: category must be valid
        if category not in valid_categories:
            rejected_ids.append(row_id)
            continue

        valid_rows.append({
            "row_id": row["row_id"],
            "account_id": row["account_id"],
            "event_time": int(row["event_time"]),
            "amount": float(row["amount"]),
            "category": row["category"],
        })

print(f"Valid rows: {len(valid_rows)}, Rejected: {len(rejected_ids)}")

# --- 2. Connect to Hopsworks ---
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

# --- 3. Create feature group ---
import hsfs.feature as feature_module

fg = fs.get_or_create_feature_group(
    name="events2cfe18",
    version=1,
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    description="Events feature table with contract-validated rows",
)

# --- 4. Insert valid rows ---
import pandas as pd

df = pd.DataFrame(valid_rows)
print(df.dtypes)
print(df.head(3))

fg.insert(df, wait=True)
print("Insert complete.")

# --- 5. Write answers.json ---
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"rejected": rejected_ids}, f)

print(f"answers.json written with {len(rejected_ids)} rejected ids.")
