# Create online table using Databricks SDK from within the workspace
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
import json

w = WorkspaceClient()

results = {}
source_table = "workspace.mlpabcbef07.transactions9dd1da"
online_table_name = "workspace.mlpabcbef07.transactions9dd1da_online"

try:
    # Try creating the online table (Synced Table in newer Databricks)
    spec = OnlineTableSpec(
        source_table_full_name=source_table,
        primary_key_columns=["row_id"],
        timeseries_key="event_time",
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
    )

    online_table = w.online_tables.create(
        name=online_table_name,
        spec=spec
    )
    results['success'] = True
    results['name'] = online_table.name
    results['spec'] = str(online_table.spec)
    results['status'] = str(online_table.status)
except Exception as e:
    results['error'] = str(e)
    results['error_type'] = type(e).__name__

    # Try the raw API
    try:
        response = w.api_client.do(
            'POST',
            '/api/2.0/online-tables',
            body={
                'name': online_table_name,
                'spec': {
                    'source_table_full_name': source_table,
                    'primary_key_columns': ['row_id'],
                    'timeseries_key': 'event_time',
                    'run_triggered': {}
                }
            }
        )
        results['raw_api_response'] = str(response)
    except Exception as e2:
        results['raw_api_error'] = str(e2)

# Write results
output_path = '/Volumes/workspace/mlpabcbef07/csvdata/create_online_result.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Results:", json.dumps(results, indent=2))
