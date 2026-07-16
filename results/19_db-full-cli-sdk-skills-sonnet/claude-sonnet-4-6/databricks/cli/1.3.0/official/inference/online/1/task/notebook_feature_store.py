# Databricks notebook source
# COMMAND ----------
from databricks.feature_store import FeatureStoreClient
import json

# COMMAND ----------
schema = "workspace.mlpabae7d2f"
table_name = f"{schema}.profilesaa70e4"
online_store_name = "mlpabae7d2f-store"
online_table_name = f"{schema}.profilesaa70e4_online"

lookup_keys = [
    "A0003", "A0005", "A0012", "A0015", "A0023", "A0030", "A0031", "A0034",
    "A0048", "A0049", "A0055", "A0063", "A0066", "A0072", "A0085", "A0090",
    "A0103", "A0109", "A0112", "A0113"
]

# COMMAND ----------
# Register feature table with primary key
fs = FeatureStoreClient()

try:
    table_info = fs.get_table(table_name)
    print(f"Feature table already exists: {table_info}")
except Exception as e:
    print(f"Creating feature table: {e}")
    fs.create_table(
        name=table_name,
        primary_keys=["account_id"],
        schema=spark.table(table_name).schema,
        description="Account feature profiles v1"
    )

# COMMAND ----------
# Write data to the feature table (it's already there as a delta table)
# Just make sure the feature table metadata is registered
df = spark.table(table_name)
print(f"Row count: {df.count()}")
df.show(5)

# COMMAND ----------
# Publish to online feature store
from databricks.feature_store.online_store_spec import AzureMySqlSpec, AmazonDynamoDBSpec

# For Databricks Online Feature Store, use the publish API
from databricks.feature_store import FeatureStoreClient
from databricks.feature_store.entities.online_store_publish_api_response import OnlineStorePublishApiResponse

fs.publish_table(
    name=table_name,
    online_store=online_store_name,
    online_table_name=online_table_name,
    mode="overwrite"
)
print("Table published to online store")

# COMMAND ----------
# Read from online store
results = {}
for account_id in lookup_keys:
    features = fs.get_online_features(
        features=[f"{table_name}:f1", f"{table_name}:f2", f"{table_name}:f3", f"{table_name}:f4"],
        lookup_key={"account_id": account_id},
        online_store=online_store_name
    )
    results[account_id] = [features["f1"], features["f2"], features["f3"], features["f4"]]
    print(f"{account_id}: {results[account_id]}")

# COMMAND ----------
# Write results to volume
output = {"vectors": results}
output_path = "/Volumes/workspace/mlpabae7d2f/mlpabae7d2f_vol/answers.json"
with open(output_path, "w") as f:
    json.dump(output, f)
print(f"Results written to {output_path}")
print(json.dumps(output, indent=2))
