# Databricks notebook source
import json, time

schema = "workspace.mlpabcef85c"
table_name = "profilesaa70e4"
full_table_name = f"{schema}.{table_name}"
prefix = "mlpabcef85c"

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

import databricks.sdk.service.catalog as cat_module

# Find available classes
online_classes = sorted([x for x in dir(cat_module) if 'nline' in x or 'Triggered' in x or 'Policy' in x])
print("Online/Triggered classes:", online_classes)

results = {"online_classes": online_classes}

# COMMAND ----------
# Try to create online table using SDK
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec

spec_kwargs = {
    "source_table_full_name": full_table_name,
    "primary_key_columns": ["account_id"],
}

# Build triggered policy if available
if hasattr(cat_module, 'OnlineTableSpecTriggeredSchedulingPolicy'):
    spec_kwargs["run_triggered"] = cat_module.OnlineTableSpecTriggeredSchedulingPolicy()
    print("Using OnlineTableSpecTriggeredSchedulingPolicy")
else:
    print("No OnlineTableSpecTriggeredSchedulingPolicy - trying without")

try:
    spec = OnlineTableSpec(**spec_kwargs)
    ot_result = w.online_tables.create(name=full_table_name, spec=spec)
    results["create_success"] = True
    results["create_result"] = str(ot_result)[:500]
    print(f"Create SUCCESS: {ot_result}")
except Exception as e:
    results["create_error"] = str(e)[:500]
    print(f"Create FAILED: {e}")

# COMMAND ----------
# Check current state
try:
    ot = w.online_tables.get(name=full_table_name)
    results["ot_status"] = str(ot.status)[:300]
    print(f"Online table status: {ot.status}")
except Exception as e:
    results["ot_get_error"] = str(e)[:300]
    print(f"Get failed: {e}")

dbutils.notebook.exit(json.dumps(results))
