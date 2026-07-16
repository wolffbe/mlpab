# Databricks notebook source
# COMMAND ----------
import json

results_df = spark.table("workspace.mlpabe8ecdb.mlpabe8ecdb_drift_results")
results_pdf = results_df.toPandas().sort_values("score", ascending=False)

all_results = results_pdf.to_dict(orient="records")
print("All feature drift scores:")
for r in all_results:
    print(r)

with open("/Volumes/workspace/mlpabe8ecdb/mlpabe8ecdb_drift_data/all_results.json", "w") as f:
    json.dump(all_results, f, default=str, indent=2)

print("Written all results!")
