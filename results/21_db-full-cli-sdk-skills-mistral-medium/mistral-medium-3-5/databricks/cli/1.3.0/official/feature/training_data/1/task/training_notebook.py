# Databricks notebook source
# MAGIC %md
# Create Churn Training Dataset

# COMMAND ----------

# MAGIC %md
# Read data from workspace files

# COMMAND ----------

import os
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Get environment variables
schema_name = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpaba01c0d')
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpaba01c0d')

# Path to data files in workspace
workspace_path = f"/Workspace/Users/benedict@logicalclocks.com/{prefix}/data"

# COMMAND ----------

# Read all CSV files from workspace
labels_df = spark.read.csv(f"{workspace_path}/labels.csv", header=True, inferSchema=True)
transactions_df = spark.read.csv(f"{workspace_path}/transactions.csv", header=True, inferSchema=True)
transactions_late_df = spark.read.csv(f"{workspace_path}/transactions_late.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv(f"{workspace_path}/profiles.csv", header=True, inferSchema=True)
activity_df = spark.read.csv(f"{workspace_path}/activity.csv", header=True, inferSchema=True)
account_health_df = spark.read.csv(f"{workspace_path}/account_health.csv", header=True, inferSchema=True)

# COMMAND ----------

# Combine transactions
all_transactions = transactions_df.unionByName(transactions_late_df)

# COMMAND ----------

# MAGIC %md
# Function to get most recent features for each (account_id, label_time)

# COMMAND ----------

def get_most_recent_features(feature_df, feature_columns, label_df, join_col='account_id', time_col='event_time'):
    """
    For each (account_id, label_time) in label_df, get the most recent feature values 
    from feature_df where event_time <= label_time.
    """
    # Join feature_df with label_df on account_id
    joined = feature_df.join(label_df, on=join_col)
    
    # Filter to event_time <= label_time
    filtered = joined.filter(F.col(time_col) <= F.col('label_time'))
    
    # For each (account_id, label_time), get the most recent row based on event_time
    window_spec = Window.partitionBy(join_col, 'label_time').orderBy(F.col(time_col).desc())
    ranked = filtered.withColumn('row_num', F.row_number().over(window_spec))
    
    # Get only the most recent row
    most_recent = ranked.filter(F.col('row_num') == 1).drop('row_num', time_col)
    
    # Select the columns we need
    result_cols = [join_col, 'label_time'] + feature_columns
    result = most_recent.select(*result_cols)
    
    return result

# COMMAND ----------

# Get features from each table
transactions_features = get_most_recent_features(
    all_transactions, 
    ['amount', 'balance'], 
    labels_df,
    'account_id', 
    'event_time'
)

profiles_features = get_most_recent_features(
    profiles_df,
    ['credit_score', 'tier'],
    labels_df,
    'account_id',
    'event_time'
)

activity_features = get_most_recent_features(
    activity_df,
    ['sessions_7d'],
    labels_df,
    'account_id',
    'event_time'
)

health_features = get_most_recent_features(
    account_health_df,
    ['health_score'],
    labels_df,
    'account_id',
    'event_time'
)

# COMMAND ----------

# MAGIC %md
# Join all features with labels

# COMMAND ----------

# Start with labels
result_df = labels_df

# Join with each feature set
result_df = result_df.join(transactions_features, on=['account_id', 'label_time'], how='left')
result_df = result_df.join(profiles_features, on=['account_id', 'label_time'], how='left')
result_df = result_df.join(activity_features, on=['account_id', 'label_time'], how='left')
result_df = result_df.join(health_features, on=['account_id', 'label_time'], how='left')

# COMMAND ----------

# Select final columns in required order
final_columns = ['account_id', 'label_time', 'amount', 'balance', 'credit_score', 'tier', 'sessions_7d', 'health_score', 'churned']
final_df = result_df.select(*final_columns)

# COMMAND ----------

# MAGIC %md
# Save as Unity Catalog table

# COMMAND ----------

# Extract catalog and schema
if '.' in schema_name:
    catalog, schema = schema_name.split('.', 1)
else:
    catalog = 'workspace'
    schema = schema_name

table_name = 'churntraining580502'
full_table_name = f'{catalog}.{schema}.{table_name}'

# Save as managed table in Unity Catalog
final_df.writeTo(f'{catalog}.{schema}.{table_name}').create()

# COMMAND ----------

print(f"Training dataset created: {full_table_name}")
print(f"Row count: {final_df.count()}")
print(f"Expected row count: {labels_df.count()}")

# Verify we have the correct number of rows (should match labels.csv)
assert final_df.count() == labels_df.count(), f"Row count mismatch: {final_df.count()} != {labels_df.count()}"

# COMMAND ----------

# Add version comment
spark.sql(f"COMMENT ON TABLE {full_table_name} IS 'Version 1 - Churn training dataset'")

# COMMAND ----------

print("Sample data:")
final_df.show(5)
print("Done!")