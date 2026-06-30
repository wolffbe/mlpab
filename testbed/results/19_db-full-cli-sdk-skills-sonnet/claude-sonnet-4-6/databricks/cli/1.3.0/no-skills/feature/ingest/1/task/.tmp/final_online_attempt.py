# Final attempt to create online/synced table using all available methods
from databricks.sdk import WorkspaceClient
import json, requests

w = WorkspaceClient()
results = {}

# Method 1: Try with perform_full_copy flag (might bypass deprecation for initial sync)
try:
    from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec
    spec = OnlineTableSpec(
        source_table_full_name="workspace.mlpabcbef07.transactions9dd1da",
        primary_key_columns=["row_id"],
        timeseries_key="event_time",
        perform_full_copy=True
    )
    result = w.online_tables.create(
        name="workspace.mlpabcbef07.transactions9dd1da_online",
        spec=spec
    )
    results['online_with_full_copy'] = str(result)
except Exception as e:
    results['online_with_full_copy_error'] = str(e)

# Method 2: Try GET on catalog/synced-tables endpoint
for path in [
    '/api/2.0/catalog/synced-tables',
    '/api/2.0/synced-tables',
    '/api/2.1/online-tables',
    '/api/2.1/catalog/online-tables',
    '/api/2.0/online-tables/v2',
]:
    try:
        response = w.api_client.do('POST', path, body={
            'name': 'workspace.mlpabcbef07.transactions9dd1da_online',
            'spec': {
                'source_table_full_name': 'workspace.mlpabcbef07.transactions9dd1da',
                'primary_key_columns': ['row_id'],
                'timeseries_key': 'event_time',
                'run_triggered': {}
            }
        })
        results[f'POST {path}'] = {'success': True, 'response': str(response)[:300]}
    except Exception as e:
        results[f'POST {path}'] = {'error': str(e)[:200]}

output_path = '/Volumes/workspace/mlpabcbef07/csvdata/final_online_result.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Done:", json.dumps(results, indent=2))
