# Try Feature Serving Endpoint as online access mechanism
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import serving

w = WorkspaceClient()
results = {}

# Check serving endpoint entity types
try:
    se_classes = [a for a in dir(serving) if 'Entity' in a or 'Spec' in a or 'Feature' in a or 'Synced' in a]
    results["serving_classes"] = se_classes
except Exception as e:
    results["serving_classes_error"] = str(e)[:100]

# Try creating a Feature Serving Endpoint
try:
    # Feature Serving endpoints use different entity configuration
    result = w.api_client.do("POST", "/api/2.0/serving-endpoints", body={
        "name": "mlpab0d6714-scaled7ecfaf-fs",
        "config": {
            "served_entities": [{
                "name": "scaled7ecfaf",
                "feature_spec_name": "workspace.mlpab0d6714.scaled7ecfaf",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }]
        }
    })
    results["feature_serving_endpoint"] = {"success": True, "data": str(result)[:300]}
except Exception as e:
    results["feature_serving_endpoint"] = {"success": False, "error": str(e)[:300]}

# Check if there are any synced tables in the system
try:
    result = w.api_client.do("GET", "/api/2.0/online-tables/workspace.mlpab0d6714.scaled7ecfaf")
    results["get_online_table"] = {"success": True, "data": str(result)[:300]}
except Exception as e:
    results["get_online_table"] = {"success": False, "error": str(e)[:200]}

dbutils.notebook.exit(json.dumps(results))
