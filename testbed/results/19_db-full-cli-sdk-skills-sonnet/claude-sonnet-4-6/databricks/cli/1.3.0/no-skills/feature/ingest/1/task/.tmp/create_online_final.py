# Create online table using Databricks SDK
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec
import json

w = WorkspaceClient()

results = {}
source_table = "workspace.mlpabcbef07.transactions9dd1da"
online_table_name = "workspace.mlpabcbef07.transactions9dd1da_online"

# Check OnlineTableSpec fields
import inspect
results['spec_fields'] = [f for f in dir(OnlineTableSpec) if not f.startswith('_')]

# Try creating with minimal spec
try:
    spec = OnlineTableSpec(
        source_table_full_name=source_table,
        primary_key_columns=["row_id"],
        timeseries_key="event_time"
    )

    # Try run_triggered as dict
    spec.run_triggered = {}

    online_table = w.online_tables.create(
        name=online_table_name,
        spec=spec
    )
    results['success'] = True
    results['online_table'] = str(online_table)
except Exception as e:
    results['error'] = str(e)
    results['error_type'] = type(e).__name__

# Write results
output_path = '/Volumes/workspace/mlpabcbef07/csvdata/online_final_result.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Results:", json.dumps(results, indent=2))
