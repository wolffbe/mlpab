# Try to create feature serving endpoint using different approaches
from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()
results = {}

# Try the feature-serving/feature-specs API with different bodies
specs_to_try = [
    {
        'name': 'workspace.mlpabcbef07.transactions9dd1da_spec',
        'features': [{'table_name': 'workspace.mlpabcbef07.transactions9dd1da', 'lookup_key': ['row_id']}]
    },
    {
        'name': 'workspace.mlpabcbef07.transactions_spec',
        'feature_table_name': 'workspace.mlpabcbef07.transactions9dd1da',
        'lookup_key': ['row_id']
    },
]

for i, body in enumerate(specs_to_try):
    try:
        response = w.api_client.do('POST', '/api/2.0/feature-serving/feature-specs', body=body)
        results[f'spec_{i}'] = {'success': True, 'response': str(response)[:500]}
    except Exception as e:
        results[f'spec_{i}'] = {'error': str(e)[:300]}

# Try PUT instead of POST for feature-serving specs
try:
    response = w.api_client.do('PUT', '/api/2.0/feature-serving/feature-specs', body={
        'name': 'workspace.mlpabcbef07.transactions9dd1da_spec',
        'features': [{'table_name': 'workspace.mlpabcbef07.transactions9dd1da', 'lookup_key': ['row_id']}]
    })
    results['put_spec'] = {'success': True, 'response': str(response)[:500]}
except Exception as e:
    results['put_spec'] = {'error': str(e)[:300]}

# Try creating a serving endpoint with feature_spec_name pointing to the feature table
try:
    response = w.api_client.do('POST', '/api/2.0/serving-endpoints', body={
        'name': 'mlpabcbef07_transactions_feat_serve',
        'config': {
            'served_entities': [
                {
                    'feature_spec_name': 'workspace.mlpabcbef07.transactions9dd1da',
                    'workload_size': 'Small',
                    'scale_to_zero_enabled': True
                }
            ]
        }
    })
    results['feature_serve_endpoint'] = {'success': True, 'response': str(response)[:500]}
except Exception as e:
    results['feature_serve_endpoint'] = {'error': str(e)[:300]}

output_path = '/Volumes/workspace/mlpabcbef07/csvdata/feature_serving_result.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Done:", json.dumps(results, indent=2)[:4000])
