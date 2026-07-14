"""
Create an online table for the scores4f5893 feature table.
"""
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab69a58e
FULL_TABLE_NAME = f"{SCHEMA}.scores4f5893"

w = WorkspaceClient()
print(f"Connected to Databricks: {w.config.host}")
print(f"Creating online table for: {FULL_TABLE_NAME}")

# Delete existing online table if any
try:
    existing = w.online_tables.get(name=FULL_TABLE_NAME)
    print(f"Existing online table found, deleting...")
    w.online_tables.delete(name=FULL_TABLE_NAME)
    time.sleep(15)
    print("Deleted.")
except Exception as e:
    err_str = str(e).lower()
    if "does not exist" in err_str or "not found" in err_str or "404" in err_str or "not an online table" in err_str:
        print(f"No existing online table.")
    else:
        print(f"Note during delete check: {e}")

# Create the online table with triggered scheduling (snapshot mode)
spec = OnlineTableSpec(
    source_table_full_name=FULL_TABLE_NAME,
    primary_key_columns=["account_id"],
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
    perform_full_copy=True,
)

try:
    online_table_obj = OnlineTable(
        name=FULL_TABLE_NAME,
        spec=spec,
    )
    result = w.online_tables.create(table=online_table_obj)
    print(f"Online table creation initiated.")
    print(f"Result: {result}")
except Exception as e:
    print(f"Online table creation error: {e}")
    raise

# Wait for it to be provisioned
print("Waiting for online table to be provisioned...")
max_wait = 600  # 10 minutes
start = time.time()
while time.time() - start < max_wait:
    try:
        ot = w.online_tables.get(name=FULL_TABLE_NAME)
        status = ot.status
        print(f"Status: {status}")
        if status and status.detailed_state:
            state = str(status.detailed_state).upper()
            if "ONLINE" in state or "ACTIVE" in state or "PROVISIONED" in state:
                print(f"Online table is ready!")
                break
            elif "FAILED" in state or "ERROR" in state:
                print(f"Online table failed: {status}")
                break
        time.sleep(20)
    except Exception as e:
        print(f"Status check error: {e}")
        time.sleep(20)

print("\nFinal status:")
try:
    ot = w.online_tables.get(name=FULL_TABLE_NAME)
    print(ot)
except Exception as e:
    print(f"Could not get final status: {e}")

print("\nDone!")
