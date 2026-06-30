# Databricks notebook source

import json

# Check SDK classes
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as sdk_catalog

# Find available classes
online_classes = [attr for attr in dir(sdk_catalog) if 'online' in attr.lower() or 'Online' in attr]
trigger_classes = [attr for attr in dir(sdk_catalog) if 'trigger' in attr.lower() or 'Trigger' in attr]

result = {
    "online_classes": online_classes,
    "trigger_classes": trigger_classes
}

# Try to create online table
catalog_name = "workspace"
schema_name = "mlpab312fe6"
source_table = f"{catalog_name}.{schema_name}.incremental3526e9"
online_name = f"{catalog_name}.{schema_name}.incremental3526e9_online"

w = WorkspaceClient()

try:
    # Try with just run_triggered as empty dict or None
    spec = sdk_catalog.OnlineTableSpec(
        source_table_full_name=source_table,
        primary_key_columns=["row_id"],
        run_triggered=sdk_catalog.OnlineTableSpecTriggeredSchedulingPolicy() if hasattr(sdk_catalog, 'OnlineTableSpecTriggeredSchedulingPolicy') else {}
    )
    online_table = w.online_tables.create(name=online_name, spec=spec)
    result['success'] = str(online_table)
except Exception as e1:
    result['error_1'] = str(e1)
    try:
        # Try without run_triggered
        spec2 = sdk_catalog.OnlineTableSpec(
            source_table_full_name=source_table,
            primary_key_columns=["row_id"]
        )
        online_table = w.online_tables.create(name=online_name, spec=spec2)
        result['success'] = str(online_table)
    except Exception as e2:
        result['error_2'] = str(e2)

dbutils.notebook.exit(json.dumps(result, indent=2))
