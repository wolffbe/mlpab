# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.ml_features.entities.feature_serving_endpoint import EndpointCoreConfig, ServedEntity

fe = FeatureEngineeringClient()

catalog_name = "workspace"
schema_name = "mlpab312fe6"
table_name = "incremental3526e9"
full_name = f"{catalog_name}.{schema_name}.{table_name}"
spec_name = f"{catalog_name}.{schema_name}.incremental3526e9_spec"
# Endpoint name must follow naming rules (alphanumeric, hyphens, underscores)
endpoint_name = "mlpab312fe6-feat-svc"

results = {}

# COMMAND ----------

# Create feature serving endpoint using ServedEntity
try:
    config = EndpointCoreConfig(
        served_entities=ServedEntity(
            feature_spec_name=spec_name,
            name="incremental3526e9",
            workload_size="Small",
            scale_to_zero_enabled=True
        )
    )
    endpoint = fe.create_feature_serving_endpoint(
        name=endpoint_name,
        config=config
    )
    results['endpoint_created'] = str(endpoint)
    print(f"Feature serving endpoint created: {endpoint}")
except Exception as e:
    results['endpoint_error'] = str(e)
    print(f"Endpoint error: {e}")

# COMMAND ----------

# Also create online store and publish table (Databricks Online Store approach)
online_store_name = "mlpab312fe6-online"  # DNS compliant

try:
    online_store = fe.create_online_store(
        name=online_store_name,
        capacity="CU_1"  # Smallest capacity
    )
    results['online_store_created'] = str(online_store)
    print(f"Online store created: {online_store}")

    # Publish feature table to online store
    publish_result = fe.publish_table(
        online_store=online_store,
        source_table_name=full_name,
        publish_mode="TRIGGERED"
    )
    results['publish_result'] = str(publish_result)
    print(f"Table published: {publish_result}")
except Exception as e:
    results['online_store_error'] = str(e)
    print(f"Online store error: {e}")

dbutils.notebook.exit(json.dumps(results, indent=2)[:5000])
