# Try various approaches for online/synced table creation
from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()
results = {}

# Try different API paths for synced tables
paths_to_try = [
    ('POST', '/api/2.0/synced-tables', {'name': 'workspace.mlpabcbef07.transactions9dd1da_online'}),
    ('POST', '/api/preview/synced-tables', {'name': 'workspace.mlpabcbef07.transactions9dd1da_online'}),
    ('POST', '/api/2.0/tables/sync', {'source_table': 'workspace.mlpabcbef07.transactions9dd1da'}),
    ('POST', '/api/2.0/catalog/synced-tables', {'name': 'workspace.mlpabcbef07.transactions9dd1da_online'}),
    ('POST', '/api/2.0/feature-store/synced-tables', {'name': 'workspace.mlpabcbef07.transactions9dd1da_online'}),
    ('POST', '/api/2.0/serving-endpoints/feature-serving', {}),
    ('GET', '/api/2.0/feature-serving/feature-specs', {}),
    ('POST', '/api/2.0/feature-serving/feature-specs', {'name': 'test'}),
]

for method, path, body in paths_to_try:
    try:
        response = w.api_client.do(method, path, body=body)
        results[f'{method} {path}'] = {'success': True, 'response': str(response)[:200]}
    except Exception as e:
        results[f'{method} {path}'] = {'error': str(e)[:200]}

output_path = '/Volumes/workspace/mlpabcbef07/csvdata/synced_attempts.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Done:", json.dumps(results, indent=2)[:3000])
