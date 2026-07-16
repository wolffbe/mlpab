# Try to create a Feature Spec for online serving
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as cat
import json

w = WorkspaceClient()
results = {}

# Explore the catalog module for feature spec related items
results['catalog_all'] = [x for x in dir(cat) if not x.startswith('_')]

# Check if there's a feature_specs API
try:
    # Try to create a feature spec via UC tables API
    # Feature specs in UC are created as a special table type
    response = w.api_client.do(
        'POST',
        '/api/2.0/unity-catalog/feature-specs',
        body={
            'name': 'workspace.mlpabcbef07.transactions9dd1da_spec',
            'features': [
                {
                    'feature_table_name': 'workspace.mlpabcbef07.transactions9dd1da',
                    'lookup_key': ['row_id'],
                    'feature_names': ['account_id', 'event_time', 'amount', 'category']
                }
            ]
        }
    )
    results['feature_spec_result'] = str(response)
except Exception as e:
    results['feature_spec_error'] = str(e)

# Try alternative API path
try:
    response2 = w.api_client.do(
        'POST',
        '/api/2.0/feature-store/feature-specs/create',
        body={
            'name': 'workspace.mlpabcbef07.transactions9dd1da_spec',
            'features': [{'table_name': 'workspace.mlpabcbef07.transactions9dd1da', 'lookup_key': ['row_id']}]
        }
    )
    results['alt_spec_result'] = str(response2)
except Exception as e:
    results['alt_spec_error'] = str(e)

# Write results
output_path = '/Volumes/workspace/mlpabcbef07/csvdata/feature_spec_result.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Done:", json.dumps(results, indent=2)[:3000])
