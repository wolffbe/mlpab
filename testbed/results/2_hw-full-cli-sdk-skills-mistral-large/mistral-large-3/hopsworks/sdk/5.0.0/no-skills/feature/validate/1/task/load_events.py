#!/usr/bin/env python3
import os
import json
import pandas as pd
import hopsworks

# Contract rules
VALID_CATEGORIES = {"grocery", "travel", "salary", "rent", "other"}
MIN_AMOUNT = 0
MAX_AMOUNT = 10000


def validate_row(row):
    """Validate a single row against the contract rules."""
    # Rule 1: amount is present
    if pd.isna(row["amount"]) or row["amount"] == "":
        return False
    
    # Rule 2: amount is within [0, 10000]
    try:
        amount = float(row["amount"])
        if not (MIN_AMOUNT <= amount <= MAX_AMOUNT):
            return False
    except (ValueError, TypeError):
        return False
    
    # Rule 3: category is one of the allowed values
    if row["category"] not in VALID_CATEGORIES:
        return False
    
    return True


def main():
    # Read the data
    df = pd.read_csv("data/events.csv")
    
    # Validate rows
    df["is_valid"] = df.apply(validate_row, axis=1)
    valid_df = df[df["is_valid"]].copy()
    rejected_ids = df[~df["is_valid"]]["row_id"].tolist()
    
    # Write rejected row IDs to submission/answers.json
    os.makedirs("submission", exist_ok=True)
    with open("submission/answers.json", "w") as f:
        json.dump({"rejected": rejected_ids}, f)
    
    # Connect to Hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    # Register feature table
    event_feature_group = fs.create_feature_group(
        name="events6149f0",
        version=1,
        description="Validated events data",
        primary_key=["row_id"],
        event_time="event_time",
        online_enabled=True,
    )
    
    # Load valid data
    event_feature_group.insert(valid_df.drop(columns=["is_valid"]), write_options={"wait_for_job": True})


if __name__ == "__main__":
    main()