# Databricks notebook source

# COMMAND ----------
import json

# Investigate f4 skew more deeply
detail = spark.sql("""
WITH joined AS (
  SELECT
    t.entity_id,
    t.f4 AS t_f4,
    s.f4 AS s_f4,
    t.f4 - s.f4 AS diff_f4,
    CASE WHEN s.f4 != 0 THEN t.f4 / s.f4 END AS ratio_f4
  FROM workspace.mlpabb808b1.training t
  INNER JOIN workspace.mlpabb808b1.serving s ON t.entity_id = s.entity_id
)
SELECT
  AVG(t_f4) AS avg_train_f4,
  AVG(s_f4) AS avg_serve_f4,
  STDDEV(t_f4) AS std_train_f4,
  STDDEV(s_f4) AS std_serve_f4,
  MIN(t_f4) AS min_train_f4,
  MIN(s_f4) AS min_serve_f4,
  MAX(t_f4) AS max_train_f4,
  MAX(s_f4) AS max_serve_f4,
  AVG(ratio_f4) AS avg_ratio,
  STDDEV(ratio_f4) AS std_ratio,
  AVG(diff_f4) AS avg_diff,
  STDDEV(diff_f4) AS std_diff,
  CORR(t_f4, s_f4) AS correlation
FROM joined
""").collect()[0]

output = {
    "avg_train_f4": float(detail["avg_train_f4"]),
    "avg_serve_f4": float(detail["avg_serve_f4"]),
    "std_train_f4": float(detail["std_train_f4"]),
    "std_serve_f4": float(detail["std_serve_f4"]),
    "min_train_f4": float(detail["min_train_f4"]),
    "min_serve_f4": float(detail["min_serve_f4"]),
    "max_train_f4": float(detail["max_train_f4"]),
    "max_serve_f4": float(detail["max_serve_f4"]),
    "avg_ratio": float(detail["avg_ratio"]) if detail["avg_ratio"] is not None else None,
    "std_ratio": float(detail["std_ratio"]) if detail["std_ratio"] is not None else None,
    "avg_diff": float(detail["avg_diff"]) if detail["avg_diff"] is not None else None,
    "std_diff": float(detail["std_diff"]) if detail["std_diff"] is not None else None,
    "correlation": float(detail["correlation"]) if detail["correlation"] is not None else None
}

print(json.dumps(output))

# COMMAND ----------
# Sample values to see the relationship
sample = spark.sql("""
WITH joined AS (
  SELECT
    t.entity_id,
    t.f4 AS t_f4,
    s.f4 AS s_f4
  FROM workspace.mlpabb808b1.training t
  INNER JOIN workspace.mlpabb808b1.serving s ON t.entity_id = s.entity_id
)
SELECT * FROM joined
ORDER BY entity_id
LIMIT 20
""").collect()

sample_out = [{"entity_id": r["entity_id"], "t_f4": float(r["t_f4"]), "s_f4": float(r["s_f4"])} for r in sample]
print(json.dumps(sample_out))
dbutils.notebook.exit(json.dumps({"stats": output, "sample": sample_out}))
