# Databricks notebook source
# COMMAND ----------
df = spark.read.csv("/Volumes/workspace/mlpab97d2fb/leakage_vol/training_data.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("training_data")
print(f"Row count: {df.count()}")

# COMMAND ----------
# Compute Pearson correlations with label using SQL
result = spark.sql("""
SELECT
  corr(f1, label) as corr_f1,
  corr(f2, label) as corr_f2,
  corr(f3, label) as corr_f3,
  corr(f4, label) as corr_f4,
  corr(f5, label) as corr_f5,
  corr(f6, label) as corr_f6
FROM training_data
""")
result.show()

row = result.collect()[0]
corrs = {
  "f1": abs(row["corr_f1"]),
  "f2": abs(row["corr_f2"]),
  "f3": abs(row["corr_f3"]),
  "f4": abs(row["corr_f4"]),
  "f5": abs(row["corr_f5"]),
  "f6": abs(row["corr_f6"])
}
print("Absolute correlations:", corrs)
leaking = max(corrs, key=corrs.get)
print(f"Leaking feature: {leaking} with |corr|={corrs[leaking]:.6f}")

# COMMAND ----------
# Also check mean per class
spark.sql("""
SELECT label,
  avg(f1) as avg_f1, avg(f2) as avg_f2, avg(f3) as avg_f3,
  avg(f4) as avg_f4, avg(f5) as avg_f5, avg(f6) as avg_f6
FROM training_data
GROUP BY label ORDER BY label
""").show()

# COMMAND ----------
# Save result
import json
evidence = {f: float(v) for f, v in corrs.items()}
result_obj = {"feature": leaking, "evidence": evidence}
result_str = json.dumps(result_obj)
print("Result:", result_str)

spark.sql(f"CREATE OR REPLACE TABLE workspace.mlpab97d2fb.leakage_result AS SELECT '{result_str}' as result")
print("Saved to table workspace.mlpab97d2fb.leakage_result")
