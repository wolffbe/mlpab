# Simple test to create online table
import json

results = {}

try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    results['sdk_ok'] = True

    # Try to list online table methods
    results['methods'] = [m for m in dir(w.online_tables) if not m.startswith('_')]

    # Try catalog module
    from databricks.sdk.service import catalog as catalog_module
    results['catalog_items'] = [x for x in dir(catalog_module) if 'Online' in x or 'Synced' in x or 'Sync' in x]

except Exception as e:
    results['error'] = str(e)

# Write results
output_path = '/Volumes/workspace/mlpabcbef07/csvdata/online_simple_result.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Done:", json.dumps(results, indent=2))
