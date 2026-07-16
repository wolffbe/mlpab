# Databricks notebook source
# COMMAND ----------
# Read predictions from volume
volume_path = "/Volumes/workspace/mlpab5c18ba/mlpab5c18ba_data"
preds = spark.read.csv(f"{volume_path}/predictions.csv", header=True, inferSchema=True)
preds.show(5)
print(f"Count: {preds.count()}")

# COMMAND ----------
# Drop existing table and create feature table with primary key
spark.sql("DROP TABLE IF EXISTS workspace.mlpab5c18ba.predictions7b586d")

# Create feature table with primary key constraint
spark.sql("""
CREATE TABLE workspace.mlpab5c18ba.predictions7b586d (
  row_id STRING NOT NULL,
  score DOUBLE,
  CONSTRAINT predictions7b586d_pk PRIMARY KEY (row_id)
)
""")
print("Feature table created")

# COMMAND ----------
# Write data to the feature table
preds.write.mode("append").saveAsTable("workspace.mlpab5c18ba.predictions7b586d")
print("Data written to feature table")

# COMMAND ----------
# Verify
result = spark.sql("SELECT * FROM workspace.mlpab5c18ba.predictions7b586d LIMIT 5")
result.show()
count = spark.sql("SELECT COUNT(*) as cnt FROM workspace.mlpab5c18ba.predictions7b586d").collect()[0]['cnt']
print(f"Total rows: {count}")
print("Feature table setup complete!")
