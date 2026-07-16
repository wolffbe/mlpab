# Try feature store publish API with different formats
from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()
results = {}

# Try different publish API configurations
publish_configs = [
    {
        'cloud': 'AWS',
        'feature_table_name': 'workspace.mlpabcbef07.transactions9dd1da',
        'online_table': 'mlpabcbef07_transactions_online',
        'store_type': 'DATABRICKS'
    },
    {
        'cloud': 'AWS',
        'feature_table_name': 'workspace.mlpabcbef07.transactions9dd1da',
        'online_table_name': 'mlpabcbef07_transactions_online',
        'store_type': 'DATABRICKS'
    },
    {
        'cloud': 'AWS',
        'feature_table': 'workspace.mlpabcbef07.transactions9dd1da',
        'online_table': 'mlpabcbef07_transactions_online',
        'store_type': 'DATABRICKS',
        'overwrite': True
    },
    {
        'cloud': 'AWS',
        'feature_table': 'workspace.mlpabcbef07.transactions9dd1da',
        'online_table': 'mlpabcbef07_transactions_online',
        'store_type': 'AWS_DYNAMO_DB'
    },
    {
        'cloud': 'AWS',
        'feature_table': 'workspace.mlpabcbef07.transactions9dd1da',
        'online_table': 'mlpabcbef07_transactions_online',
        'store_type': 'MANAGED'
    },
]

for i, config in enumerate(publish_configs):
    try:
        response = w.api_client.do('POST', '/api/2.0/feature-store/feature-tables/publish', body=config)
        results[f'publish_{i}'] = {'success': True, 'response': str(response)[:500]}
    except Exception as e:
        results[f'publish_{i}'] = {'error': str(e)[:300], 'config_store_type': config.get('store_type')}

output_path = '/Volumes/workspace/mlpabcbef07/csvdata/publish_result.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Done:", json.dumps(results, indent=2))
