# Databricks notebook source
# COMMAND ----------
import json

# Read existing data
df = spark.table("workspace.mlpabc8d80a.scaled7ecfaf")

# COMMAND ----------
# Drop and recreate with NOT NULL constraint on row_id
spark.sql("DROP TABLE IF EXISTS workspace.mlpabc8d80a.scaled7ecfaf")

# Create with NOT NULL and primary key in one step
spark.sql("""
    CREATE TABLE workspace.mlpabc8d80a.scaled7ecfaf (
        row_id STRING NOT NULL,
        split STRING,
        f1 DOUBLE,
        f2 DOUBLE,
        f3 DOUBLE,
        f4 DOUBLE,
        CONSTRAINT scaled7ecfaf_pk PRIMARY KEY (row_id)
    )
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")

# COMMAND ----------
# Insert data
df.write.format("delta").mode("append").saveAsTable("workspace.mlpabc8d80a.scaled7ecfaf")

# COMMAND ----------
# Verify
count = spark.sql("SELECT COUNT(*) as cnt FROM workspace.mlpabc8d80a.scaled7ecfaf").collect()[0][0]
sample = spark.sql("SELECT * FROM workspace.mlpabc8d80a.scaled7ecfaf LIMIT 3").collect()
sample_str = str([(r.row_id, r.split, r.f1, r.f2) for r in sample])

dbutils.notebook.exit(json.dumps({"count": count, "sample": sample_str}))
