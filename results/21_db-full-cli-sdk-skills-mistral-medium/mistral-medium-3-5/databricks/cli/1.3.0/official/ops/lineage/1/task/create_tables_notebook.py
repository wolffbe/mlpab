# Databricks notebook source
# MAGIC %md
# MAGIC ## Create Feature Tables and Derived Table

# COMMAND ----------

# Set up the schema
schema_name = "workspace.mlpab81f308"

# COMMAND ----------

# Upload raw_a.csv to DBFS
dbutils.fs.cp("file:/databricks/driver/data/raw_a.csv", "dbfs:/tmp/raw_a.csv")

# Upload raw_b.csv to DBFS  
dbutils.fs.cp("file:/databricks/driver/data/raw_b.csv", "dbfs:/tmp/raw_b.csv")

# COMMAND ----------

# Read the CSV files
raw_a_df = spark.read.csv("dbfs:/tmp/raw_a.csv", header=True, inferSchema=True)
raw_b_df = spark.read.csv("dbfs:/tmp/raw_b.csv", header=True, inferSchema=True)

# COMMAND ----------

# Create rawa1b0772 table
raw_a_df.write.saveAsTable(f"{schema_name}.rawa1b0772")

# Create rawb1b0772 table
raw_b_df.write.saveAsTable(f"{schema_name}.rawb1b0772")

# COMMAND ----------

# Create derived table with inner join and sum
from pyspark.sql.functions import round

derived_df = raw_a_df.join(raw_b_df, "row_id", "inner") \
    .select("row_id", round(raw_a_df["a_val"] + raw_b_df["b_val"], 6).alias("col_sum"))

# Write derived table
derived_df.write.saveAsTable(f"{schema_name}.derived1b0772")

# COMMAND ----------

# Create online tables for low-latency access
spark.sql(f"CREATE ONLINE TABLE {schema_name}.rawa1b0772_online AS SELECT * FROM {schema_name}.rawa1b0772")
spark.sql(f"CREATE ONLINE TABLE {schema_name}.rawb1b0772_online AS SELECT * FROM {schema_name}.rawb1b0772")
spark.sql(f"CREATE ONLINE TABLE {schema_name}.derived1b0772_online AS SELECT * FROM {schema_name}.derived1b0772")

# COMMAND ----------

# Write the derived table to CSV for submission
derived_df.coalesce(1).write.csv("dbfs:/tmp/derived1b0772.csv", header=True, mode="overwrite")

# COMMAND ----------

# Create answers.json
import json
answers = {
    "derived_from": ["rawa1b0772", "rawb1b0772"]
}

with open("/dbfs/tmp/answers.json", "w") as f:
    json.dump(answers, f)

# COMMAND ----------

print("Done!")
