import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable,
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

w = WorkspaceClient()

schema_full = os.environ["MLPAB_DATABRICKS_SCHEMA"]
catalog, schema_name = schema_full.split(".", 1)
ot_full_name = f"{catalog}.{schema_name}.featuresb1ea93_online"

print(f"Creating online table: {ot_full_name}")

try:
    existing = w.online_tables.get(ot_full_name)
    print(f"Online table already exists: {existing.name}, status: {existing.status}")
except Exception as e:
    if "does not exist" in str(e) or "NOT_FOUND" in str(e) or "NotFound" in str(e).__class__.__name__:
        spec = OnlineTableSpec(
            source_table_full_name=f"{catalog}.{schema_name}.featuresb1ea93",
            primary_key_columns=["row_id"],
            timeseries_key="event_time",
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
        )
        table = OnlineTable(name=ot_full_name, spec=spec)
        waiter = w.online_tables.create(table=table)
        print(f"Online table creation started, waiting...")
        # The create returns a waiter - wait for it
        try:
            result = waiter.result(timeout=600)
            print(f"Online table ready: {result.name}")
            print(f"Status: {result.status}")
        except Exception as wait_err:
            print(f"Wait error (table may still be provisioning): {wait_err}")
            # Check status
            ot = w.online_tables.get(ot_full_name)
            print(f"Current status: {ot.status}")
    else:
        print(f"Unexpected error: {e}")
        raise

print("\nFinal check...")
ot = w.online_tables.get(ot_full_name)
print(f"Online table: {ot.name}")
print(f"Status: {ot.status}")
print(f"Table serving URL: {ot.table_serving_url}")
