# Try feature store publish with specific store types
from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()
results = {}

store_types = [
    'DYNAMODB_ONLINE_STORE',
    'AZURE_COSMOS_DB_ONLINE_STORE',
    'BIGTABLE_ONLINE_STORE',
    'DYNAMO_DB_ONLINE_STORE',
    'COSMOS_DB_ONLINE_STORE',
    'AWS_DYNAMODB',
    'AMAZON_DYNAMODB',
    'DYNAMODB',
]

for store_type in store_types:
    try:
        response = w.api_client.do('POST', '/api/2.0/feature-store/feature-tables/publish', body={
            'cloud': 'AWS',
            'feature_table': 'workspace.mlpabcbef07.transactions9dd1da',
            'online_table': 'mlpabcbef07_transactions_online',
            'store_type': store_type
        })
        results[store_type] = {'success': True, 'response': str(response)[:500]}
    except Exception as e:
        results[store_type] = {'error': str(e)[:200]}

output_path = '/Volumes/workspace/mlpabcbef07/csvdata/publish2_result.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Done:", json.dumps(results, indent=2))
