# Databricks notebook source
# Create churn training dataset with point-in-time correct feature joins

schema = "workspace.mlpabecab87"
volume_path = "/Volumes/workspace/mlpabecab87/rawdata"

# COMMAND ----------
# Read raw data from volume
transactions = spark.read.csv(f"{volume_path}/transactions.csv", header=True, inferSchema=True)
transactions_late = spark.read.csv(f"{volume_path}/transactions_late.csv", header=True, inferSchema=True)
profiles = spark.read.csv(f"{volume_path}/profiles.csv", header=True, inferSchema=True)
activity = spark.read.csv(f"{volume_path}/activity.csv", header=True, inferSchema=True)
health = spark.read.csv(f"{volume_path}/account_health.csv", header=True, inferSchema=True)
labels = spark.read.csv(f"{volume_path}/labels.csv", header=True, inferSchema=True)

# Combine transactions and transactions_late
all_transactions = transactions.union(transactions_late)

# Register as temp views
all_transactions.createOrReplaceTempView("transactions_all")
profiles.createOrReplaceTempView("profiles_all")
activity.createOrReplaceTempView("activity_all")
health.createOrReplaceTempView("health_all")
labels.createOrReplaceTempView("labels_tbl")

# COMMAND ----------
# Point-in-time join: for each (account_id, label_time), get most recent feature values at or before label_time
result = spark.sql("""
WITH t_ranked AS (
  SELECT t.account_id, t.event_time, t.amount, t.balance, l.label_time,
         ROW_NUMBER() OVER (PARTITION BY t.account_id, l.label_time ORDER BY t.event_time DESC) AS rn
  FROM transactions_all t
  JOIN labels_tbl l ON t.account_id = l.account_id AND t.event_time <= l.label_time
),
p_ranked AS (
  SELECT p.account_id, p.event_time, p.credit_score, p.tier, l.label_time,
         ROW_NUMBER() OVER (PARTITION BY p.account_id, l.label_time ORDER BY p.event_time DESC) AS rn
  FROM profiles_all p
  JOIN labels_tbl l ON p.account_id = l.account_id AND p.event_time <= l.label_time
),
a_ranked AS (
  SELECT a.account_id, a.event_time, a.sessions_7d, l.label_time,
         ROW_NUMBER() OVER (PARTITION BY a.account_id, l.label_time ORDER BY a.event_time DESC) AS rn
  FROM activity_all a
  JOIN labels_tbl l ON a.account_id = l.account_id AND a.event_time <= l.label_time
),
h_ranked AS (
  SELECT h.account_id, h.event_time, h.health_score, l.label_time,
         ROW_NUMBER() OVER (PARTITION BY h.account_id, l.label_time ORDER BY h.event_time DESC) AS rn
  FROM health_all h
  JOIN labels_tbl l ON h.account_id = l.account_id AND h.event_time <= l.label_time
)
SELECT
  l.account_id,
  l.label_time,
  t.amount,
  t.balance,
  p.credit_score,
  p.tier,
  a.sessions_7d,
  h.health_score,
  l.churned
FROM labels_tbl l
LEFT JOIN (SELECT account_id, label_time, amount, balance FROM t_ranked WHERE rn = 1) t
  ON l.account_id = t.account_id AND l.label_time = t.label_time
LEFT JOIN (SELECT account_id, label_time, credit_score, tier FROM p_ranked WHERE rn = 1) p
  ON l.account_id = p.account_id AND l.label_time = p.label_time
LEFT JOIN (SELECT account_id, label_time, sessions_7d FROM a_ranked WHERE rn = 1) a
  ON l.account_id = a.account_id AND l.label_time = a.label_time
LEFT JOIN (SELECT account_id, label_time, health_score FROM h_ranked WHERE rn = 1) h
  ON l.account_id = h.account_id AND l.label_time = h.label_time
""")

# COMMAND ----------
# Save as Delta table
result.write.format("delta").mode("overwrite").saveAsTable(f"{schema}.churntraining807643")
print(f"Training dataset created: {schema}.churntraining807643")
print(f"Row count: {result.count()}")
