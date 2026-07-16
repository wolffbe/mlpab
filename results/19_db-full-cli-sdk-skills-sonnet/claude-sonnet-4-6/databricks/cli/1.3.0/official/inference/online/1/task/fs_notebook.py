# Databricks notebook source
# COMMAND ----------
import json

schema = "workspace.mlpabae7d2f"
table_name = f"{schema}.profilesaa70e4"
online_store_name = "mlpabae7d2f-store"
online_table_name = f"{schema}.profilesaa70e4_online"
output_path = "/Volumes/workspace/mlpabae7d2f/mlpabae7d2f_vol/answers.json"

lookup_keys = [
    "A0003", "A0005", "A0012", "A0015", "A0023", "A0030", "A0031", "A0034",
    "A0048", "A0049", "A0055", "A0063", "A0066", "A0072", "A0085", "A0090",
    "A0103", "A0109", "A0112", "A0113"
]

# COMMAND ----------
# Register feature table using Feature Store client
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# Drop and recreate feature table metadata with primary key
try:
    fs.drop_table(name=table_name)
    print(f"Dropped existing feature table metadata")
except Exception as e:
    print(f"Table not registered yet (OK): {e}")

# COMMAND ----------
# Create feature table with primary key registered
df = spark.table(table_name)
print(f"Data schema: {df.schema}")
print(f"Row count: {df.count()}")

fs.create_table(
    name=table_name,
    primary_keys=["account_id"],
    schema=df.schema,
    description="Account feature profiles v1"
)
print(f"Feature table registered with primary key")

# Write data to it (fs.write_table uses the delta table under the hood)
fs.write_table(
    name=table_name,
    df=df,
    mode="overwrite"
)
print("Data written to feature table")

# COMMAND ----------
# Publish to Databricks Online Feature Store
fs.publish_table(
    name=table_name,
    online_store=online_store_name,
    online_table_name=online_table_name,
    mode="overwrite"
)
print(f"Table published to online store {online_store_name}")

# COMMAND ----------
# Read features from online store
results = {}
for account_id in lookup_keys:
    feature_vector = fs.get_online_features(
        feature_view=table_name,
        lookup_keys=[{"account_id": account_id}],
        online_store_name=online_store_name
    )
    f = feature_vector[0]
    results[account_id] = [float(f["f1"]), float(f["f2"]), float(f["f3"]), float(f["f4"])]
    print(f"{account_id}: {results[account_id]}")

# COMMAND ----------
# Write results to volume
output = {"vectors": results}
with open(output_path, "w") as fh:
    json.dump(output, fh)
print(f"Results written to {output_path}")
print(json.dumps(output, indent=2))
