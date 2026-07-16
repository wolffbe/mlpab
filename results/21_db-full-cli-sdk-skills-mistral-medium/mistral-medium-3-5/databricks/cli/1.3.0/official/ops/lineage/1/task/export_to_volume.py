# Databricks notebook source
# MAGIC %python

# COMMAND ----------

# Set up the schema
catalog_name = "workspace"
schema_name = "mlpab81f308"

# COMMAND ----------

# Read the derived table
derived_df = spark.sql(f"SELECT * FROM {catalog_name}.{schema_name}.derived1b0772")

# Write to volume
derived_df.coalesce(1).write.mode("overwrite").csv(f"{catalog_name}.{schema_name}.submission_volume/derived1b0772.csv", header=True)
print("Wrote derived1b0772.csv to volume")

# COMMAND ----------

# Create answers.json in volume
import json
answers = {
    "derived_from": ["rawa1b0772", "rawb1b0772"]
}

with open(f"/Volumes/{catalog_name}/{schema_name}/submission_volume/answers.json", "w") as f:
    json.dump(answers, f)

print("Wrote answers.json to volume")

# COMMAND ----------

print("Done!")
