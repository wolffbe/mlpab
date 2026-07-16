#!/usr/bin/env python3
import hopsworks
import json
import os

# Login to Hopsworks
hopsworks.login()

# Get the feature store
proj = hopsworks.project.Project()
fs = proj.get_feature_store()

print(f"Feature store: {fs.name}")

# First, upload the CSV file to the dataset API
print("Uploading CSV file...")
dataset_api = proj.get_dataset_api()
upload_path = dataset_api.upload(
    local_path="data/features.csv",
    upload_path="Resources/drift_data/features.csv",
    overwrite=True
)
print(f"CSV uploaded to: {upload_path}")

# Create feature group from the uploaded CSV
fg_name = "drift_detection_fg"
fg_version = 1

try:
    fg = fs.get_feature_group(fg_name, version=fg_version)
    print(f"Found existing feature group: {fg.name} v{fg.version}")
except Exception as e:
    print(f"Creating feature group from CSV: {e}")
    fg = fs.create_feature_group(
        name=fg_name,
        version=fg_version,
        description="Feature group for drift detection",
        online_enabled=False,
        path=upload_path,
        time_travel_format="HUDI",
        event_time="event_time"
    )
    print(f"Created feature group: {fg.name} v{fg.version}")

print("Done!")
