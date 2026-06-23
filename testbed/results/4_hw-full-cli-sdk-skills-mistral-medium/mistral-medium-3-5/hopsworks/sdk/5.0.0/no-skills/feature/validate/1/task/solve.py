import hopsworks
import csv
import json
import os
import pandas as pd

# Connect to Hopsworks
hopsworks.login()

# Read the CSV file
with open('data/events.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

valid_categories = {'grocery', 'travel', 'salary', 'rent', 'other'}
rejected = []
valid_rows = []

for row in rows:
    row_id = row['row_id']
    amount_str = row['amount'].strip()
    category = row['category'].strip()
    
    # Check amount is present
    if amount_str == '':
        rejected.append(row_id)
        continue
    
    # Check amount is numeric
    try:
        amount = float(amount_str)
    except ValueError:
        rejected.append(row_id)
        continue
    
    # Check amount is in [0, 10000]
    if amount < 0 or amount > 10000:
        rejected.append(row_id)
        continue
    
    # Check category
    if category not in valid_categories:
        rejected.append(row_id)
        continue
    
    # If we get here, the row is valid
    valid_rows.append(row)

# Write the answers.json
os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as f:
    json.dump({"rejected": sorted(rejected)}, f)

print(f"Total rows: {len(rows)}")
print(f"Valid rows: {len(valid_rows)}")
print(f"Rejected rows: {len(rejected)}")

# Now register the feature table with valid rows
project = hopsworks.get_current_project()
fs = project.get_feature_store()

# Convert valid rows to pandas DataFrame with proper types
df = pd.DataFrame(valid_rows)
df['amount'] = df['amount'].astype(float)
df['event_time'] = df['event_time'].astype('int64')

# Register the feature group (feature table)
# In Hopsworks, feature tables are called "feature groups"
feature_group = fs.create_feature_group(
    name="events4b3862",
    version=1,
    description="Filtered events data satisfying the contract",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True  # Enable for low-latency lookup
)

# Insert the data into the feature group
feature_group.insert(df)

print("Feature group registered successfully")
print(f"Rejected IDs written to submission/answers.json")
