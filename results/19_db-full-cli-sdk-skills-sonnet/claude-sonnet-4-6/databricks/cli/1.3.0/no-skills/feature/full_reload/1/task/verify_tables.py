# Databricks notebook source

TABLE_NAME = "workspace.mlpab2c4304.customersc31b07"
output_lines = []

# COMMAND ----------

# Verify feature table schema and data
schema_df = spark.sql(f"DESCRIBE {TABLE_NAME}")
output_lines.append("Schema:")
for row in schema_df.collect():
    output_lines.append(f"  {row['col_name']}: {row['data_type']}")

# COMMAND ----------

count = spark.sql(f"SELECT COUNT(*) FROM {TABLE_NAME}").collect()[0][0]
output_lines.append(f"\nRow count: {count}")

# COMMAND ----------

sample = spark.sql(f"SELECT * FROM {TABLE_NAME} LIMIT 3")
output_lines.append("\nSample rows:")
for row in sample.collect():
    output_lines.append(f"  {dict(row.asDict())}")

# COMMAND ----------

# Check table properties (feature table registration)
props_df = spark.sql(f"SHOW TBLPROPERTIES {TABLE_NAME}")
output_lines.append("\nTable properties (feature table):")
for row in props_df.collect():
    output_lines.append(f"  {row['key']}: {row['value']}")

# Verify no old columns exist
cols = [r['col_name'] for r in spark.sql(f"DESCRIBE {TABLE_NAME}").collect() if not r['col_name'].startswith('#')]
output_lines.append(f"\nColumns: {cols}")
has_old = 'name' in cols or 'balance_eur' in cols
has_new = 'full_name' in cols and 'balance' in cols and 'currency' in cols
output_lines.append(f"Has old v1 columns (should be False): {has_old}")
output_lines.append(f"Has new v2 columns (should be True): {has_new}")

dbutils.notebook.exit("\n".join(output_lines))
