#!/usr/bin/env python3
"""
Task: Register feature table eventsd693d3 in Databricks, load valid rows, enable online access.
"""

import csv
import json
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    TableType,
    DataSourceFormat,
    ColumnInfo,
    ColumnTypeName,
)

# Environment
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # e.g. workspace.mlpab394daa
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]   # e.g. mlpab394daa
CATALOG, SCHEMA_NAME = SCHEMA.split(".", 1)

TABLE_NAME = "eventsd693d3"
FULL_TABLE = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"

DATA_FILE = "data/events.csv"
SUBMISSION_DIR = "submission"

VALID_CATEGORIES = {"grocery", "travel", "salary", "rent", "other"}

def validate_row(row):
    """Return True if row passes all contract rules."""
    amount_str = row.get("amount", "").strip()
    category = row.get("category", "").strip()

    # Rule 1: amount must be present
    if not amount_str or amount_str.lower() == "null":
        return False, "null/empty amount"

    # Rule 2: amount in [0, 10000]
    try:
        amount = float(amount_str)
    except ValueError:
        return False, f"non-numeric amount: {amount_str}"

    if amount < 0 or amount > 10000:
        return False, f"amount out of range: {amount}"

    # Rule 3: valid category
    if category not in VALID_CATEGORIES:
        return False, f"invalid category: {category}"

    return True, None

def read_and_filter_csv():
    valid_rows = []
    rejected_ids = []

    with open(DATA_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ok, reason = validate_row(row)
            if ok:
                valid_rows.append(row)
            else:
                rejected_ids.append(row["row_id"])
                print(f"  REJECTED {row['row_id']}: {reason}")

    return valid_rows, rejected_ids

def main():
    print("=== Starting task ===")
    print(f"Schema: {SCHEMA}")
    print(f"Table: {FULL_TABLE}")

    # Step 1: Filter data
    print("\n--- Step 1: Reading and filtering CSV ---")
    valid_rows, rejected_ids = read_and_filter_csv()
    print(f"Valid rows: {len(valid_rows)}, Rejected: {len(rejected_ids)}")
    print(f"Rejected IDs: {rejected_ids}")

    # Write submission/answers.json
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    answers_path = os.path.join(SUBMISSION_DIR, "answers.json")
    with open(answers_path, "w") as f:
        json.dump({"rejected": rejected_ids}, f, indent=2)
    print(f"Written: {answers_path}")

    # Step 2: Connect to Databricks
    print("\n--- Step 2: Connecting to Databricks ---")
    w = WorkspaceClient()
    print(f"Connected: {w.config.host}")

    # Step 3: Explore Databricks SDK for feature engineering APIs
    print("\n--- Step 3: Exploring SDK capabilities ---")
    print("Available in w:", [x for x in dir(w) if not x.startswith('_')])

    # Step 4: Check if feature engineering is available
    print("\n--- Step 4: Checking feature engineering ---")
    if hasattr(w, 'feature_store'):
        print("feature_store available:", dir(w.feature_store))
    else:
        print("No feature_store attribute")

    # Check online tables
    if hasattr(w, 'online_tables'):
        print("online_tables available:", dir(w.online_tables))
    else:
        print("No online_tables attribute")

    # Step 5: Create Delta table in Unity Catalog
    print("\n--- Step 5: Creating Delta table ---")

    # First, create the table using SQL via statement execution
    # Build CSV data as SQL values
    # We need to upload the data somehow

    # Check what SQL execution looks like
    print("Checking statement execution API...")
    if hasattr(w, 'statement_execution'):
        print("statement_execution available")

    # Check for warehouses
    if hasattr(w, 'warehouses'):
        warehouses = list(w.warehouses.list())
        print(f"Warehouses: {[wh.name for wh in warehouses]}")

    # Check volumes
    if hasattr(w, 'volumes'):
        print("volumes API available")

    # Check files API
    if hasattr(w, 'files'):
        print("files API available:", [x for x in dir(w.files) if not x.startswith('_')])

    return valid_rows, rejected_ids

if __name__ == "__main__":
    valid_rows, rejected_ids = main()
