# Databricks notebook source
# COMMAND ----------
import json

# Read from the Delta table that was created by the drift analysis job
results_df = spark.table("workspace.mlpabe8ecdb.mlpabe8ecdb_drift_results")
results_pdf = results_df.toPandas()
print(results_pdf.to_string())

# Get the best feature (highest score)
best_row = results_pdf.loc[results_pdf["score"].idxmax()]
best_feature = str(best_row["feature"])
best_onset = str(best_row["onset_date"])[:10]

print("Best feature: " + best_feature)
print("Onset date: " + best_onset)

answer = {"feature": best_feature, "onset": best_onset}
answer_json = json.dumps(answer)
print("ANSWER JSON: " + answer_json)

# Write to volume as a JSON file
with open("/Volumes/workspace/mlpabe8ecdb/mlpabe8ecdb_drift_data/answer.json", "w") as f:
    f.write(answer_json)

print("Written to volume!")
