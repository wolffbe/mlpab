# Databricks notebook source
# MAGIC %md
# Create Training Dataset using SQL

# COMMAND ----------

# MAGIC %sql
# Create temporary views from the CSV files

# COMMAND ----------

# First, let's read the CSV files and create temporary views
# We'll use the workspace files

# COMMAND ----------

import os

# Get environment variables
schema_name = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpaba01c0d')
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpaba01c0d')
workspace_path = f"/Workspace/Users/benedict@logicalclocks.com/{prefix}/data"

# COMMAND ----------

# Read CSV files and create temp views
labels_df = spark.read.csv(f"{workspace_path}/labels.csv", header=True, inferSchema=True)
labels_df.createOrReplaceTempView("labels")

transactions_df = spark.read.csv(f"{workspace_path}/transactions.csv", header=True, inferSchema=True)
transactions_late_df = spark.read.csv(f"{workspace_path}/transactions_late.csv", header=True, inferSchema=True)
all_transactions = transactions_df.unionByName(transactions_late_df)
all_transactions.createOrReplaceTempView("all_transactions")

profiles_df = spark.read.csv(f"{workspace_path}/profiles.csv", header=True, inferSchema=True)
profiles_df.createOrReplaceTempView("profiles")

activity_df = spark.read.csv(f"{workspace_path}/activity.csv", header=True, inferSchema=True)
activity_df.createOrReplaceTempView("activity")

account_health_df = spark.read.csv(f"{workspace_path}/account_health.csv", header=True, inferSchema=True)
account_health_df.createOrReplaceTempView("account_health")

# COMMAND ----------

# MAGIC %sql
# Create the training dataset using SQL

# COMMAND ----------

# Extract catalog and schema
if '.' in schema_name:
    catalog, schema = schema_name.split('.', 1)
else:
    catalog = 'workspace'
    schema = schema_name

table_name = 'churntraining580502'
full_table_name = f'{catalog}.{schema}.{table_name}'

# COMMAND ----------

# Use SQL to create the final dataset
query = f"""
WITH 
-- Get most recent transaction for each (account_id, label_time)
recent_transactions AS (
  SELECT 
    l.account_id,
    l.label_time,
    t.amount,
    t.balance,
    ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY t.event_time DESC) as rn
  FROM labels l
  JOIN all_transactions t ON l.account_id = t.account_id AND t.event_time <= l.label_time
),

-- Get most recent profile for each (account_id, label_time)
recent_profiles AS (
  SELECT 
    l.account_id,
    l.label_time,
    p.credit_score,
    p.tier,
    ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY p.event_time DESC) as rn
  FROM labels l
  JOIN profiles p ON l.account_id = p.account_id AND p.event_time <= l.label_time
),

-- Get most recent activity for each (account_id, label_time)
recent_activity AS (
  SELECT 
    l.account_id,
    l.label_time,
    a.sessions_7d,
    ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY a.event_time DESC) as rn
  FROM labels l
  JOIN activity a ON l.account_id = a.account_id AND a.event_time <= l.label_time
),

-- Get most recent health for each (account_id, label_time)
recent_health AS (
  SELECT 
    l.account_id,
    l.label_time,
    h.health_score,
    ROW_NUMBER() OVER (PARTITION BY l.account_id, l.label_time ORDER BY h.event_time DESC) as rn
  FROM labels l
  JOIN account_health h ON l.account_id = h.account_id AND h.event_time <= l.label_time
)

SELECT 
  l.account_id,
  l.label_time,
  rt.amount,
  rt.balance,
  rp.credit_score,
  rp.tier,
  ra.sessions_7d,
  rh.health_score,
  l.churned
FROM labels l
LEFT JOIN recent_transactions rt ON l.account_id = rt.account_id AND l.label_time = rt.label_time AND rt.rn = 1
LEFT JOIN recent_profiles rp ON l.account_id = rp.account_id AND l.label_time = rp.label_time AND rp.rn = 1
LEFT JOIN recent_activity ra ON l.account_id = ra.account_id AND l.label_time = ra.label_time AND ra.rn = 1
LEFT JOIN recent_health rh ON l.account_id = rh.account_id AND l.label_time = rh.label_time AND rh.rn = 1
"""

# Execute the query
result_df = spark.sql(query)

# COMMAND ----------

# Save as Unity Catalog table
result_df.writeTo(f'{catalog}.{schema}.{table_name}').create()

# COMMAND ----------

print(f"Training dataset created: {full_table_name}")
print(f"Row count: {result_df.count()}")
print(f"Expected row count: {labels_df.count()}")

# Verify we have the correct number of rows
assert result_df.count() == labels_df.count(), f"Row count mismatch: {result_df.count()} != {labels_df.count()}"

# Add version comment
spark.sql(f"COMMENT ON TABLE {full_table_name} IS 'Version 1 - Churn training dataset'")

# COMMAND ----------

print("Sample data:")
result_df.show(5)
print("Done!")