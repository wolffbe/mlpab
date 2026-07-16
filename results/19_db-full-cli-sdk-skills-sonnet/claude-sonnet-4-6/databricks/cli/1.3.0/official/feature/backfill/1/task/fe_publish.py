# Databricks notebook source
# COMMAND ----------
import json

table_name = "workspace.mlpab0442b8.accountse81ff1"

# COMMAND ----------
# Register as feature table and publish online
results = {}

try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    results["fe_import"] = "success"

    # Try to get the table first
    try:
        ft = fe.get_table(name=table_name)
        results["get_table"] = str(ft)
    except Exception as e:
        results["get_table_error"] = str(e)
        # Register
        try:
            fe.register_table(
                name=table_name,
                primary_keys=["row_id"],
                timestamp_keys=["updated_at"]
            )
            results["register_table"] = "success"
        except Exception as e2:
            results["register_error"] = str(e2)

    # Try to publish to online store
    try:
        from databricks.feature_engineering.entities.online_store_spec import AzureCosmosDBSpec, PineconeSpec, AmazonDynamoDBSpec
        results["online_store_specs"] = "found cosmos/pinecone/dynamo"
    except Exception as e:
        results["online_store_specs_error"] = str(e)

    # Try databricks online store
    try:
        from databricks.feature_engineering import DatabricksOnlineStoreSpec
        online_spec = DatabricksOnlineStoreSpec()
        fe.publish_table(name=table_name, online_store=online_spec)
        results["publish_databricks"] = "success"
    except Exception as e:
        results["publish_databricks_error"] = str(e)

except ImportError as e:
    results["fe_import_error"] = str(e)

# COMMAND ----------
# Try old FeatureStoreClient
try:
    from databricks import feature_store as fs
    fsc = fs.FeatureStoreClient()
    results["fsc_import"] = "success"

    try:
        from databricks.feature_store.online_store_spec import AmazonDynamoDBSpec
        results["fsc_dynamo"] = "found"
    except:
        pass

    try:
        from databricks.feature_store import DatabricksOnlineStoreSpec
        online_spec = DatabricksOnlineStoreSpec()
        fsc.publish_table(name=table_name, online_store=online_spec)
        results["fsc_publish"] = "success"
    except Exception as e:
        results["fsc_publish_error"] = str(e)

except ImportError as e:
    results["fsc_import_error"] = str(e)

# COMMAND ----------
# Write results
results_json = json.dumps(results, indent=2)
spark.createDataFrame([(results_json,)], ["results"]).write.mode("overwrite").text("/Volumes/workspace/mlpab0442b8/mlpab0442b8_uploads/fe_results.txt")
print(results_json)
dbutils.notebook.exit(results_json)
