#!/usr/bin/env python3
"""
Script to detect data drift using Hopsworks feature store.
"""
import hopsworks
import json
import os

def main():
    # Login to Hopsworks
    print("Logging in to Hopsworks...")
    h = hopsworks.login()
    
    # Get feature store
    fs = h.get_feature_store()
    
    # Upload the data to a dataset
    print("Uploading data...")
    dataset_api = h.get_dataset_api()
    
    # Create a directory for the data
    data_path = "Resources/drift_data"
    dataset_api.mkdir(data_path)
    
    # Upload the CSV file
    local_path = "data/features.csv"
    remote_path = f"{data_path}/features.csv"
    dataset_api.upload(local_path, remote_path)
    
    # Create a feature group
    print("Creating feature group...")
    fg_name = "drift_detection_fg"
    
    # Check if feature group already exists
    try:
        fg = fs.get_feature_group(fg_name)
        print(f"Feature group {fg_name} already exists")
    except:
        # Create new feature group
        fg = fs.create_feature_group(
            name=fg_name,
            version=1,
            description="Feature group for drift detection",
            online_enabled=False,
            time_travel_format="HUDI"
        )
        print(f"Created feature group: {fg.name}")
    
    # Get the feature group
    fg = fs.get_feature_group(fg_name)
    
    # Upload the data to the feature group
    print("Uploading data to feature group...")
    dataset_api.upload_feature_group(
        local_path=local_path,
        feature_group=fg,
        overwrite=True
    )
    
    print("Data uploaded successfully!")
    
    # Now let's analyze the data for drift
    # We'll use the feature store's SQL capabilities to query the data
    print("\nAnalyzing data for drift...")
    
    # Get the feature view or query the feature group
    # First, let's see what features are available
    features = fg.get_features()
    print(f"Features in group: {[f.name for f in features]}")
    
    # Query the data to get statistics over time
    # We'll split the data into two periods and compare
    query = fg.select_all()
    
    # Get the data as a dataframe (this should work on the platform)
    # But we can't use pandas locally, so we need to use the platform's capabilities
    
    # Let's try to get statistics using the feature store
    # Check if there's a statistics method
    print(f"\nFeature group methods: {[m for m in dir(fg) if not m.startswith('_')]}")
    
    # Try to get the underlying storage and query it
    # Let's check what's available
    
    h.logout()

if __name__ == "__main__":
    main()
