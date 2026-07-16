# Check Databricks SDK capabilities for online/synced tables
import json

results = {}

try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()

    # Check what APIs are available
    api_list = [attr for attr in dir(w) if not attr.startswith('_')]
    results['workspace_client_apis'] = api_list

    # Check online tables API
    if hasattr(w, 'online_tables'):
        results['online_tables_methods'] = [m for m in dir(w.online_tables) if not m.startswith('_')]

    # Check feature serving
    if hasattr(w, 'feature_serving'):
        results['feature_serving_methods'] = [m for m in dir(w.feature_serving) if not m.startswith('_')]

    # Try to find synced tables
    synced_attrs = [attr for attr in dir(w) if 'sync' in attr.lower()]
    results['synced_related'] = synced_attrs

except Exception as e:
    results['sdk_error'] = str(e)

# Write results
output_path = '/Volumes/workspace/mlpabcbef07/csvdata/sdk_check3.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Results written. Keys:", list(results.keys()))
