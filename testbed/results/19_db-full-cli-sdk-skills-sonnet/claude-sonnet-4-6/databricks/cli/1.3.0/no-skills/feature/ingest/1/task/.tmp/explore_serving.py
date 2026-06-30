# Explore serving endpoints and feature serving options
from databricks.sdk import WorkspaceClient
import json, requests

w = WorkspaceClient()

results = {}

# Check serving endpoints API
results['serving_endpoint_methods'] = [m for m in dir(w.serving_endpoints) if not m.startswith('_')]

# Check if there's a feature_specs API
results['workspace_api_list'] = [m for m in dir(w) if not m.startswith('_')]

# Try creating a feature serving endpoint
try:
    # Look at what parameters create takes
    import inspect
    sig = inspect.signature(w.serving_endpoints.create)
    results['create_signature'] = str(sig)
except Exception as e:
    results['sig_error'] = str(e)

# Check if there's a specific feature serving type
try:
    from databricks.sdk.service import serving
    results['serving_module_items'] = [x for x in dir(serving) if 'Feature' in x or 'feature' in x]
except Exception as e:
    results['serving_error'] = str(e)

# Try to create online table via direct HTTP using workspace token
try:
    import os
    host = os.environ.get('DATABRICKS_HOST', '')
    token = os.environ.get('DATABRICKS_TOKEN', '')

    if not host or not token:
        # Try getting from workspace
        host = spark.conf.get('spark.databricks.workspaceUrl', '')
        token = dbutils.secrets.get('databricks', 'token') if False else ''

    results['host'] = host
    results['has_token'] = bool(token)

except Exception as e:
    results['env_error'] = str(e)

# Write results
output_path = '/Volumes/workspace/mlpabcbef07/csvdata/serving_explore.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Done:", json.dumps(results, indent=2)[:2000])
