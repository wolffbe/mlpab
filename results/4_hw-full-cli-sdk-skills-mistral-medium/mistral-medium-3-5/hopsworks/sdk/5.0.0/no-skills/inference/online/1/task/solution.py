import hopsworks
import json
import os
import pandas as pd
import time

# Connect to Hopsworks
print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Define feature group name and version
FEATURE_GROUP_NAME = "profiles37db55"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "profiles37db55"
FEATURE_VIEW_VERSION = 1

# Clean up if they already exist
try:
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)
    print(f"Feature group {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION} already exists, deleting...")
    fg.delete()
except:
    print(f"Feature group {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION} does not exist")

try:
    fv = fs.get_feature_view(FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION)
    print(f"Feature view {FEATURE_VIEW_NAME} v{FEATURE_VIEW_VERSION} already exists, deleting...")
    fv.delete()
except:
    print(f"Feature view {FEATURE_VIEW_NAME} v{FEATURE_VIEW_VERSION} does not exist")

# Read the CSV file
print("Reading CSV file...")
df = pd.read_csv("data/features.csv")
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Create feature group with online enabled
print("Creating feature group...")
fg = fs.create_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
    description="Account feature profiles",
    primary_key=["account_id"],
    online_enabled=True,
)

# Save the data to the feature group
print("Saving data to feature group...")
fg.save(df, wait=True)

# Wait for online ingestion to complete
print("Waiting for online ingestion...")
time.sleep(10)

# Create feature view
print("Creating feature view...")
fv = fs.create_feature_view(
    name=FEATURE_VIEW_NAME,
    version=FEATURE_VIEW_VERSION,
    description="Account feature profiles view",
    query=fg.select_all(),
)

# Wait a bit more for the feature view to be ready
print("Waiting for feature view to be ready...")
time.sleep(5)

# Read lookup keys
print("Reading lookup keys...")
with open("data/lookup_keys.txt", "r") as f:
    lookup_keys = [line.strip() for line in f.readlines()]

print(f"Found {len(lookup_keys)} lookup keys: {lookup_keys}")

# Retrieve feature vectors through online/low-latency read path
print("Retrieving feature vectors from online store...")
vectors = {}

for key in lookup_keys:
    print(f"  Looking up {key}...")
    # Get the feature vector from online store
    feature_vector = fv.get_feature_vector(entry={"account_id": key}, return_type="list")
    # The feature vector includes account_id as first element, then f1, f2, f3, f4
    # Skip the first element (account_id) and take f1, f2, f3, f4
    vectors[key] = [float(x) for x in feature_vector[1:]]
    print(f"    {key}: {vectors[key]}")

# Write results to submission/answers.json
print("Writing results to submission/answers.json...")
os.makedirs("submission", exist_ok=True)
result = {"vectors": vectors}

with open("submission/answers.json", "w") as f:
    json.dump(result, f, indent=2)

print("Done!")
print(f"Result: {json.dumps(result, indent=2)}")
