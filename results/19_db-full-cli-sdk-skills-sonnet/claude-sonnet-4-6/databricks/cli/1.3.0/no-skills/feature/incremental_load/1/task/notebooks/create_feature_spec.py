# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

catalog_name = "workspace"
schema_name = "mlpab312fe6"
table_name = "incremental3526e9"
full_name = f"{catalog_name}.{schema_name}.{table_name}"
spec_name = f"{catalog_name}.{schema_name}.incremental3526e9_spec"
endpoint_name = "mlpab312fe6_feat_serving"

results = {}

# COMMAND ----------

# Use Feature Engineering client to create feature spec and deploy
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

# Create feature spec
try:
    feature_spec = fe.create_feature_spec(
        name=spec_name,
        features=[
            FeatureLookup(
                table_name=full_name,
                lookup_key="row_id",
                feature_names=["account_id", "event_time", "amount", "category"]
            )
        ]
    )
    results['feature_spec'] = "created"
    print(f"Feature spec created: {spec_name}")
except Exception as e:
    results['feature_spec_error'] = str(e)
    print(f"Feature spec creation error: {e}")

# COMMAND ----------

# Try deploying feature spec
try:
    endpoint = fe.deploy_feature_spec(
        name=endpoint_name,
        feature_spec_name=spec_name,
        workload_size="Small",
        scale_to_zero_enabled=True
    )
    results['endpoint'] = str(endpoint)
    print(f"Feature serving endpoint deployed: {endpoint_name}")
except Exception as e:
    results['endpoint_error'] = str(e)
    print(f"Endpoint deployment error: {e}")

# COMMAND ----------

# Check what fe methods are available
fe_methods = [m for m in dir(fe) if not m.startswith('_')]
results['fe_methods'] = fe_methods
print(f"Available FE methods: {fe_methods}")

dbutils.notebook.exit(json.dumps(results, indent=2)[:3000])
