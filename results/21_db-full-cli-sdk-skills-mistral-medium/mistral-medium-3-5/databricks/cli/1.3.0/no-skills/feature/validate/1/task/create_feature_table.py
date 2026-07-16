# Databricks notebook source
# MAGIC %python

# COMMAND ----------

# Read the valid events CSV from workspace
file_path = "/Workspace/Users/benedict@logicalclocks.com/mlpab47905b/valid_events.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Show schema and count
print(f"Total valid rows: {df.count()}")
df.printSchema()
df.show(5)

# COMMAND ----------

# Create the feature table in Unity Catalog
catalog_name = "workspace"
schema_name = "mlpab47905b"
table_name = "eventsa45e2a"

# Create the table with the valid data
(df.write 
   .format("delta")
   .mode("overwrite")
   .option("overwriteSchema", "true")
   .saveAsTable(f"{catalog_name}.{schema_name}.{table_name}"))

print(f"Table {catalog_name}.{schema_name}.{table_name} created successfully!")

# COMMAND ----------

# Verify the table was created
spark.sql(f"SELECT COUNT(*) as count FROM {catalog_name}.{schema_name}.{table_name}").show()

# COMMAND ----------

# Create an online table for low-latency access
# First, let's check if the table exists and get its full name
full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

# Create the online table
spark.sql(f"""
CREATE ONLINE TABLE IF NOT EXISTS {full_table_name}
USING DELTA 
LOCATION 'delta.`{full_table_name}`'
""")

print(f"Online table created for {full_table_name}")
