import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as c

w = WorkspaceClient()

try:
    result = w.online_tables.create(
        name="workspace.mlpabbb38f1.derivedd05474_online",
        spec=c.OnlineTableSpec(
            source_table_full_name="workspace.mlpabbb38f1.derivedd05474",
            primary_key_columns=["row_id"],
            run_triggered=c.OnlineTableSpecTriggeredSchedulingPolicy()
        )
    )
    dbutils.notebook.exit(f"SUCCESS: {result}")
except Exception as e:
    dbutils.notebook.exit(f"ERROR: {type(e).__name__}: {e}")
