# Check SDK and create online table
import json

results = {}

# Check SDK version
try:
    import databricks.sdk
    results['sdk_version'] = databricks.sdk.__version__
except Exception as e:
    results['sdk_error'] = str(e)

# Check feature engineering
try:
    import databricks.feature_engineering as fe_module
    results['fe_available'] = True
    results['fe_methods'] = [x for x in dir(fe_module) if not x.startswith('_')]
except Exception as e:
    results['fe_error'] = str(e)

# Check feature store
try:
    import databricks.feature_store as fs_module
    results['fs_available'] = True
    results['fs_methods'] = [x for x in dir(fs_module) if not x.startswith('_')]
except Exception as e:
    results['fs_error'] = str(e)

# Write results to volume
output_path = '/Volumes/workspace/mlpabcbef07/csvdata/sdk_check.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Done - results written to", output_path)
print(json.dumps(results, indent=2))
