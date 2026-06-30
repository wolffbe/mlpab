# Try serving endpoint with entity_name for feature table
from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()
results = {}

# Try different entity configurations
entity_configs = [
    {
        'name': 'mlpabcbef07_feat_serve_1',
        'config': {
            'served_entities': [{
                'entity_name': 'workspace.mlpabcbef07.transactions9dd1da',
                'entity_version': '1',
                'workload_size': 'Small',
                'scale_to_zero_enabled': True
            }]
        }
    },
    {
        'name': 'mlpabcbef07_feat_serve_2',
        'config': {
            'served_entities': [{
                'entity_name': 'workspace.mlpabcbef07.transactions9dd1da',
                'workload_size': 'Small',
                'scale_to_zero_enabled': True,
                'entity_type': 'FEATURE_SPEC'
            }]
        }
    },
]

for config in entity_configs:
    name = config['name']
    try:
        response = w.api_client.do('POST', '/api/2.0/serving-endpoints', body=config)
        results[name] = {'success': True, 'response': str(response)[:300]}
        # Try to delete if created
        try:
            w.api_client.do('DELETE', f'/api/2.0/serving-endpoints/{name}')
        except:
            pass
    except Exception as e:
        results[name] = {'error': str(e)[:300]}

# Also try GET on feature-serving paths
for path in [
    '/api/2.0/feature-serving/feature-specs/workspace.mlpabcbef07.transactions9dd1da',
    '/api/2.0/feature-serving/endpoints',
]:
    try:
        response = w.api_client.do('GET', path)
        results[f'GET {path}'] = {'success': True, 'response': str(response)[:300]}
    except Exception as e:
        results[f'GET {path}'] = {'error': str(e)[:300]}

output_path = '/Volumes/workspace/mlpabcbef07/csvdata/entity_name_result.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Done:", json.dumps(results, indent=2))
