#!/usr/bin/env python3
import hopsworks
import hsfs

# Login to Hopsworks
hopsworks.login()

# Get the feature store
proj = hopsworks.project.Project()
fs = proj.get_feature_store()

# Get the feature group
fg_name = "drift_detection_fg"
fg_version = 1

try:
    fg = fs.get_feature_group(fg_name, version=fg_version)
    print(f"Found feature group: {fg.name} v{fg.version}")
    print(f"Feature group path: {fg.path}")
    print(f"Feature group ID: {fg.id}")
except Exception as e:
    print(f"Error getting feature group: {e}")
    exit(1)

# Try to insert data from the uploaded CSV
# First, let's see what methods are available
print([x for x in dir(fg) if not x.startswith('_') and callable(getattr(fg, x))])

# Try to get the data as a dataframe using the dataset API
dataset_api = proj.get_dataset_api()
try:
    # Read the uploaded CSV
    local_path = "data/features.csv"
    # Try to insert using the path
    job = fg.insert(
        features=local_path,
        overwrite=True
    )
    print(f"Insert job: {job}")
except Exception as e:
    print(f"Error inserting: {e}")

print("Done!")
