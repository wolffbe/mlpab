#!/usr/bin/env python3
"""
Script to ingest daily increments into the feature group `incremental97a30c`.
This script is designed to run as a Hopsworks job.
"""

import hopsworks
import pandas as pd
import os

def ingest_increment():
    # Connect to Hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    # Get the feature group
    fg = fs.get_feature_group("incremental97a30c", version=1)
    
    # List all increment files in Resources
    resources_dir = "/Projects/{}/Resources/".format(project.projectname)
    increment_files = [f for f in os.listdir(resources_dir) if f.startswith("increment_") and f.endswith(".csv")]
    
    # Sort files to ensure chronological order
    increment_files.sort()
    
    # Read and insert each file
    for file in increment_files:
        file_path = os.path.join(resources_dir, file)
        df = pd.read_csv(file_path)
        print(f"Inserting {file} with {len(df)} rows...")
        fg.insert(df, write_options={"start_offline_materialization": True})
        print(f"Inserted {file} successfully.")

if __name__ == "__main__":
    ingest_increment()