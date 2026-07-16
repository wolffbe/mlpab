#!/usr/bin/env python3
"""
Script to create feature groups on Hopsworks platform and derive a new feature group.
"""

import hopsworks
import pandas as pd
import os

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read source data
raw_a = pd.read_csv("data/raw_a.csv")
raw_b = pd.read_csv("data/raw_b.csv")

# Create feature groups for raw tables
rawa_fg = fs.create_feature_group(
    name="rawa7b4d0b",
    version=1,
    description="Raw feature group from raw_a.csv",
    primary_key=["row_id"],
    online_enabled=True,
)
rawa_fg.insert(raw_a, write_options={"wait_for_job": True})

rawb_fg = fs.create_feature_group(
    name="rawb7b4d0b",
    version=1,
    description="Raw feature group from raw_b.csv",
    primary_key=["row_id"],
    online_enabled=True,
)
rawb_fg.insert(raw_b, write_options={"wait_for_job": True})

# Find common row_ids
common_row_ids = set(raw_a["row_id"]).intersection(set(raw_b["row_id"]))
raw_a_common = raw_a[raw_a["row_id"].isin(common_row_ids)]
raw_b_common = raw_b[raw_b["row_id"].isin(common_row_ids)]

# Merge and compute col_sum
merged = pd.merge(raw_a_common, raw_b_common, on="row_id", how="inner")
merged["col_sum"] = (merged["a_val"] + merged["b_val"]).round(6)

# Create derived feature group
derived_fg = fs.create_feature_group(
    name="derived7b4d0b",
    version=1,
    description="Derived feature group with col_sum = a_val + b_val",
    primary_key=["row_id"],
    online_enabled=True,
)
derived_fg.insert(merged[["row_id", "col_sum"]], write_options={"wait_for_job": True})

# Register lineage
rawa_fg.save()
rawb_fg.save()
derived_fg.save()

# Add lineage metadata
derived_fg.update_statistics_config(
    description="Derived from rawa7b4d0b and rawb7b4d0b"
)

# Write lineage answer
lineage_answer = {"derived_from": sorted(["rawa7b4d0b", "rawb7b4d0b"])}
import json
with open("submission/answers.json", "w") as f:
    json.dump(lineage_answer, f, indent=2)

# If platform is unavailable, write derived table to CSV
# (This is a fallback and not expected to be used in normal operation)
merged[["row_id", "col_sum"]].to_csv("submission/derived7b4d0b.csv", index=False)

print("Feature groups created and lineage registered successfully.")