# Databricks notebook source
# MAGIC %md
# Create Training Dataset for Churn Prediction

# COMMAND ----------

# MAGIC %md
# Load and prepare data

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, max as spark_max
from pyspark.sql.window import Window
import os

# Get environment variables
schema_name = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpaba01c0d')
catalog_name = 'workspace'

# COMMAND ----------

# MAGIC %md
# Read CSV files from DBFS

# COMMAND ----------

# First, let's upload the data files to DBFS so we can read them
# We'll use the workspace import to create a notebook that does the processing

# Actually, let's create a notebook that reads from the local data directory
# But since we're in a notebook, we need to first get the data to DBFS

# For now, let's create the processing logic

# COMMAND ----------

# Read all CSV files
labels_df = spark.read.csv("/dbfs/FileStore/data/labels.csv", header=True, inferSchema=True)
transactions_df = spark.read.csv("/dbfs/FileStore/data/transactions.csv", header=True, inferSchema=True)
transactions_late_df = spark.read.csv("/dbfs/FileStore/data/transactions_late.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv("/dbfs/FileStore/data/profiles.csv", header=True, inferSchema=True)
activity_df = spark.read.csv("/dbfs/FileStore/data/activity.csv", header=True, inferSchema=True)
account_health_df = spark.read.csv("/dbfs/FileStore/data/account_health.csv", header=True, inferSchema=True)

# COMMAND ----------

# Combine transactions
all_transactions = transactions_df.unionByName(transactions_late_df)

# COMMAND ----------

# MAGIC %md
# For each feature table, get the most recent value at or before each label_time

# COMMAND ----------

# Create a function to get the most recent value for each account at or before each label_time
def get_most_recent_features(feature_df, feature_columns, label_df, join_column='account_id', time_column='event_time'):
    """
    For each (account_id, label_time) in label_df, get the most recent feature values from feature_df
    where event_time <= label_time.
    """
    # Cross join to get all combinations
    cross_joined = feature_df.crossJoin(label_df)
    
    # Filter to only rows where event_time <= label_time and account_id matches
    filtered = cross_joined.filter(
        (feature_df[join_column] == label_df[join_column]) & 
        (feature_df[time_column] <= label_df['label_time'])
    )
    
    # For each (account_id, label_time), get the most recent row
    window_spec = Window.partitionBy(label_df[join_column], label_df['label_time']).orderBy(feature_df[time_column].desc())
    
    ranked = filtered.withColumn('rank', spark_max(feature_df[time_column]).over(window_spec))
    
    # Actually, let's use a simpler approach with row_number
    from pyspark.sql.functions import row_number
    
    window_spec = Window.partitionBy(label_df[join_column], label_df['label_time']).orderBy(feature_df[time_column].desc())
    ranked = filtered.withColumn('row_num', row_number().over(window_spec))
    
    # Get only the most recent row for each (account_id, label_time)
    most_recent = ranked.filter(col('row_num') == 1).drop('row_num')
    
    # Select only the columns we need
    result_cols = [label_df[join_column], label_df['label_time']] + feature_columns
    result = most_recent.select(*result_cols)
    
    return result

# COMMAND ----------

# Actually, the above approach might be inefficient. Let's use a better approach:
# For each account, get the most recent feature values at or before each label_time

# Better approach: For each feature table, create a window function to get the most recent value per account
# Then join with labels on account_id and ensure event_time <= label_time

def get_features_for_labels(feature_df, feature_columns, label_df, join_column='account_id', time_column='event_time'):
    """
    Get the most recent feature values for each (account_id, label_time) pair.
    """
    # Add label columns to feature df for joining
    # We need to join on account_id and find the most recent event_time <= label_time for each label
    
    # Create a cross join between feature_df and label_df filtered by account_id
    joined = feature_df.join(label_df, on=join_column)
    
    # Filter to event_time <= label_time
    filtered = joined.filter(col(time_column) <= col('label_time'))
    
    # For each (account_id, label_time), get the most recent row
    from pyspark.sql.functions import row_number
    window_spec = Window.partitionBy(join_column, 'label_time').orderBy(col(time_column).desc())
    ranked = filtered.withColumn('row_num', row_number().over(window_spec))
    
    # Get only the most recent row
    most_recent = ranked.filter(col('row_num') == 1).drop('row_num', time_column)
    
    # Select the columns we need
    result_cols = [join_column, 'label_time'] + feature_columns
    result = most_recent.select(*result_cols)
    
    return result

# COMMAND ----------

# Get features from each table
transactions_features = get_features_for_labels(
    all_transactions, 
    ['amount', 'balance'], 
    labels_df,
    'account_id', 
    'event_time'
)

profiles_features = get_features_for_labels(
    profiles_df,
    ['credit_score', 'tier'],
    labels_df,
    'account_id',
    'event_time'
)

activity_features = get_features_for_labels(
    activity_df,
    ['sessions_7d'],
    labels_df,
    'account_id',
    'event_time'
)

health_features = get_features_for_labels(
    account_health_df,
    ['health_score'],
    labels_df,
    'account_id',
    'event_time'
)

# COMMAND ----------

# Join all features together with labels
# Start with labels
result_df = labels_df

# Join with transactions features
result_df = result_df.join(transactions_features, on=['account_id', 'label_time'], how='left')

# Join with profiles features
result_df = result_df.join(profiles_features, on=['account_id', 'label_time'], how='left')

# Join with activity features
result_df = result_df.join(activity_features, on=['account_id', 'label_time'], how='left')

# Join with health features
result_df = result_df.join(health_features, on=['account_id', 'label_time'], how='left')

# COMMAND ----------

# Select the final columns in the required order
final_columns = ['account_id', 'label_time', 'amount', 'balance', 'credit_score', 'tier', 'sessions_7d', 'health_score', 'churned']
final_df = result_df.select(*final_columns)

# COMMAND ----------

# MAGIC %md
# Save as Unity Catalog table

# COMMAND ----------

# Extract catalog and schema from schema_name
if '.' in schema_name:
    catalog, schema = schema_name.split('.', 1)
else:
    catalog = 'workspace'
    schema = schema_name

# Create the table name
table_name = 'churntraining580502'
full_table_name = f'{catalog}.{schema}.{table_name}'

# Save as a managed table in Unity Catalog
final_df.writeTo(f'{catalog}.{schema}.{table_name}').create()

# COMMAND ----------

print(f"Training dataset created successfully: {full_table_name}")
print(f"Row count: {final_df.count()}")
print("Sample data:")
final_df.show(5)

# COMMAND ----------

# Also create a version 1 of this table using the versioning feature
# We can use the table properties to set the version

# Actually, for Unity Catalog, we can use the DESCRIBE HISTORY to see versions
# But for now, let's just create the table with version 1 in the name or as a property

# Let's check if we need to set version as a property
# For now, the table is created, and we can add version metadata if needed

# COMMAND ----------

# Add version comment to the table
spark.sql(f"COMMENT ON TABLE {full_table_name} IS 'Version 1 - Churn training dataset'")

# COMMAND ----------

print("Done!")