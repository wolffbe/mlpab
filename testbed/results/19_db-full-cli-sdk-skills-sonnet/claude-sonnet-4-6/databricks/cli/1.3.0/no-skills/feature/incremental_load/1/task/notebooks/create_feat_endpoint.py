# Databricks notebook source
# MAGIC %pip install databricks-feature-engineering -q

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import inspect

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

fe = FeatureEngineeringClient()

catalog_name = "workspace"
schema_name = "mlpab312fe6"
table_name = "incremental3526e9"
full_name = f"{catalog_name}.{schema_name}.{table_name}"
spec_name = f"{catalog_name}.{schema_name}.incremental3526e9_spec"
endpoint_name = "mlpab312fe6_feat_serving"

results = {}

# Inspect create_feature_serving_endpoint signature
try:
    sig = str(inspect.signature(fe.create_feature_serving_endpoint))
    results['endpoint_sig'] = sig
    print(f"Signature: {sig}")
except Exception as e:
    results['sig_error'] = str(e)

# Try creating feature serving endpoint
try:
    endpoint = fe.create_feature_serving_endpoint(
        name=endpoint_name,
        feature_spec_name=spec_name,
        workload_size="Small",
        scale_to_zero_enabled=True
    )
    results['endpoint_created'] = str(endpoint)
    print(f"Feature serving endpoint created: {endpoint}")
except Exception as e:
    results['endpoint_error1'] = str(e)
    print(f"Error 1: {e}")

    # Try without scale_to_zero
    try:
        endpoint = fe.create_feature_serving_endpoint(
            name=endpoint_name,
            feature_spec_name=spec_name,
            workload_size="Small"
        )
        results['endpoint_created'] = str(endpoint)
        print(f"Feature serving endpoint created: {endpoint}")
    except Exception as e2:
        results['endpoint_error2'] = str(e2)
        print(f"Error 2: {e2}")

        # Try minimal args
        try:
            endpoint = fe.create_feature_serving_endpoint(
                name=endpoint_name,
                feature_spec_name=spec_name
            )
            results['endpoint_created'] = str(endpoint)
            print(f"Feature serving endpoint created: {endpoint}")
        except Exception as e3:
            results['endpoint_error3'] = str(e3)
            print(f"Error 3: {e3}")

# COMMAND ----------

# Also try publish_table and create_online_store
try:
    sig_publish = str(inspect.signature(fe.publish_table))
    results['publish_sig'] = sig_publish
    print(f"publish_table signature: {sig_publish}")
except Exception as e:
    results['publish_sig_error'] = str(e)

try:
    sig_online = str(inspect.signature(fe.create_online_store))
    results['online_store_sig'] = sig_online
    print(f"create_online_store signature: {sig_online}")
except Exception as e:
    results['online_store_sig_error'] = str(e)

dbutils.notebook.exit(json.dumps(results, indent=2)[:5000])
