# Databricks notebook source
# COMMAND ----------
import json
results = {}

# Check available classes
from databricks.sdk.service import catalog as sdk_catalog
results['online_synced_classes'] = [c for c in dir(sdk_catalog) if 'Online' in c or 'Synced' in c]

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
results['has_online_tables'] = hasattr(w, 'online_tables')
results['has_synced_tables'] = hasattr(w, 'synced_tables')
results['workspace_client_attrs'] = [a for a in dir(w) if not a.startswith('_')]

# Try creating online table
try:
    from databricks.sdk.service.catalog import OnlineTableSpec
    online_table = w.online_tables.create(
        name="workspace.mlpabf1452c.featuresb1ea93_online",
        spec=OnlineTableSpec(
            source_table_full_name="workspace.mlpabf1452c.featuresb1ea93",
            primary_key_columns=["row_id"],
            timeseries_key="event_time"
        )
    )
    results['online_table_created'] = str(online_table)
except Exception as e:
    results['online_table_error'] = f"{type(e).__name__}: {str(e)}"

dbutils.notebook.exit(json.dumps(results))
