# Databricks notebook source
# MAGIC %md
# Batch Scoring Notebook

# COMMAND ----------

import pandas as pd
import numpy as np
import json

# COMMAND ----------

# Read the input files from DBFS
dbfs_prefix = "/FileStore/tmp/mlpab383e7b/"

# Upload files first (this will be done via CLI)
# Then read them

# For now, let's read from the local data directory
# But we need to upload them to DBFS first

# COMMAND ----------

# Read feature history
feature_history = spark.read.csv("dbfs:/FileStore/tmp/mlpab383e7b/feature_history.csv", header=True, inferSchema=True)

# Read model
with open("dbfs:/FileStore/tmp/mlpab383e7b/model.json", "r") as f:
    model = json.load(f)

# Read scoring request
with open("dbfs:/FileStore/tmp/mlpab383e7b/scoring_request.md", "r") as f:
    scoring_request = f.read()

# COMMAND ----------

# Parse T from scoring request
import re
match = re.search(r'T = (\d+)', scoring_request)
T = int(match.group(1))
print(f"Scoring timestamp T: {T}")

# Get weights and bias
weights = model["weights"]
bias = model["bias"]
print(f"Weights: {weights}")
print(f"Bias: {bias}")

# COMMAND ----------

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# COMMAND ----------

# Filter feature history to get most recent revision at or before T for each account
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Convert event_time to long type
feature_history = feature_history.withColumn("event_time", F.col("event_time").cast("long"))

# Filter to only include revisions at or before T
filtered_history = feature_history.filter(F.col("event_time") <= T)

# For each account, get the most recent revision
window_spec = Window.partitionBy("account_id").orderBy(F.col("event_time").desc())
ranked_history = filtered_history.withColumn("rank", F.row_number().over(window_spec))
latest_revisions = ranked_history.filter(F.col("rank") == 1).drop("rank", "event_time")

# COMMAND ----------

# Compute scores
weights_f1 = weights["f1"]
weights_f2 = weights["f2"]
weights_f3 = weights["f3"]

# Create the linear combination
latest_revisions = latest_revisions.withColumn(
    "linear_comb",
    F.col("f1") * weights_f1 + F.col("f2") * weights_f2 + F.col("f3") * weights_f3 + bias
)

# Apply sigmoid and round to 6 decimal places
from pyspark.sql.types import DoubleType

def sigmoid_udf(x):
    return float(1 / (1 + np.exp(-x)))

sigmoid_func = F.udf(sigmoid_udf, DoubleType())

scores_df = latest_revisions.withColumn("score", F.round(sigmoid_func(F.col("linear_comb")), 6))

# Select only account_id and score
scores_df = scores_df.select("account_id", "score")

# COMMAND ----------

# Write to Unity Catalog table
catalog_name = "workspace"
schema_name = "mlpab383e7b"
table_name = "scores076684"

# Create the table
scores_df.write.mode("overwrite").option("mergeSchema", "true").saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")

# COMMAND ----------

# Create an online table for low-latency lookup
# First, register the table as a feature table
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

w = WorkspaceClient()

# Create or update the feature table
feature_table_name = f"{catalog_name}.{schema_name}.{table_name}"

# Enable online feature lookup
# This requires using the Feature Store API
try:
    # Get the feature store client
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    
    # Create feature table
    feature_table = fs.create_feature_table(
        name=feature_table_name,
        primary_keys=["account_id"],
        df=scores_df,
        description="Batch scores for all accounts as of T=1773568800000"
    )
    
    # Publish the feature table for online access
    fs.publish_feature_table(feature_table)
    
    print(f"Feature table {feature_table_name} created and published for online access")
except Exception as e:
    print(f"Could not create feature table via FeatureStoreClient: {e}")
    # Try alternative approach using REST API
    print("Trying alternative approach...")

# COMMAND ----------

# Alternative: Use Unity Catalog to create the table and then enable online access
# We already created the table above, now let's try to enable online feature serving

# For Databricks, online feature tables are created via the Feature Store
# Let's use the REST API approach

import requests
import os

# Get workspace URL
workspace_url = os.environ.get("DATABRICKS_HOST", "https://adb-1234567890123456.7.azuredatabricks.net")
token = os.environ.get("DATABRICKS_TOKEN", "")

# Create feature table via REST API
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# First, check if the table exists
try:
    response = requests.get(
        f"{workspace_url}/api/2.1/unity-catalog/tables/{catalog_name}.{schema_name}.{table_name}",
        headers=headers
    )
    print(f"Table check response: {response.status_code}")
except Exception as e:
    print(f"Error checking table: {e}")

# COMMAND ----------

print("Scoring complete!")
