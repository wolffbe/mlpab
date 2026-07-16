import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as c

w = WorkspaceClient()

# Check what's in catalog for online tables
online_items = [x for x in dir(c) if 'online' in x.lower() or 'Online' in x]
print(f"Online-related: {online_items}")

try:
    result = w.online_tables.create(
        name="workspace.mlpabbb38f1.derivedd05474_online",
        spec=c.OnlineTableSpec(
            source_table_full_name="workspace.mlpabbb38f1.derivedd05474",
            primary_key_columns=["row_id"],
            run_triggered=c.OnlineTableSpecTriggeredSchedulingPolicy() if hasattr(c, 'OnlineTableSpecTriggeredSchedulingPolicy') else {}
        )
    )
    dbutils.notebook.exit(f"SUCCESS: {result}")
except Exception as e:
    # Try with dict
    try:
        result2 = w.online_tables.create(
            name="workspace.mlpabbb38f1.derivedd05474_online",
            spec=c.OnlineTableSpec(
                source_table_full_name="workspace.mlpabbb38f1.derivedd05474",
                primary_key_columns=["row_id"]
            )
        )
        dbutils.notebook.exit(f"SUCCESS2: {result2}")
    except Exception as e2:
        dbutils.notebook.exit(f"ERROR1: {e}\nERROR2: {e2}\nOnline items: {online_items}")
