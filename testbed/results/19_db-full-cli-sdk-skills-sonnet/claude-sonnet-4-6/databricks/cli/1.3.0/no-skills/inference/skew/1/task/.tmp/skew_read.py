# Databricks notebook source

# COMMAND ----------
import json

result = spark.sql("SELECT * FROM workspace.mlpabb808b1.skew_results").collect()
row = result[0]

output = {
    "avg_diff_f1": float(row["avg_diff_f1"]),
    "avg_diff_f2": float(row["avg_diff_f2"]),
    "avg_diff_f3": float(row["avg_diff_f3"]),
    "avg_diff_f4": float(row["avg_diff_f4"]),
    "avg_diff_f5": float(row["avg_diff_f5"]),
    "max_diff_f1": float(row["max_diff_f1"]),
    "max_diff_f2": float(row["max_diff_f2"]),
    "max_diff_f3": float(row["max_diff_f3"]),
    "max_diff_f4": float(row["max_diff_f4"]),
    "max_diff_f5": float(row["max_diff_f5"]),
    "n_common_entities": int(row["n_common_entities"])
}

print(json.dumps(output))
dbutils.notebook.exit(json.dumps(output))
