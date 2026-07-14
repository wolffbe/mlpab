# Databricks notebook source
# MAGIC %md
# Analyze Feature Leakage

# COMMAND ----------

# Read the CSV file from the Volume
volume_path = "/Volumes/workspace/mlpabb1ccad/training_data_volume/training_data.csv"
df = spark.read.csv(volume_path, header=True, inferSchema=True)

# Create or replace a Delta table
df.write.format("delta").mode("overwrite").saveAsTable("workspace.mlpabb1ccad.training_data")

# COMMAND ----------

# Compute correlation between each feature and the label
from pyspark.sql.functions import corr

features = ["f1", "f2", "f3", "f4", "f5", "f6"]
correlations = {}

for feature in features:
    correlation = df.select(corr(feature, "label")).collect()[0][0]
    correlations[feature] = correlation

# Identify the feature with the highest absolute correlation
leaking_feature = max(correlations.items(), key=lambda x: abs(x[1]))[0]

# Output the result
print(f"Correlations: {correlations}")
print(f"Leaking feature: {leaking_feature}")

# COMMAND ----------

# Write the result to submission/answers.json
dbutils.fs.put("file:/dbfs/submission/answers.json", f"{{\"feature\": \"{leaking_feature}\", \"evidence\": \"Highest absolute correlation with label: {correlations[leaking_feature]}\"}}", overwrite=True)