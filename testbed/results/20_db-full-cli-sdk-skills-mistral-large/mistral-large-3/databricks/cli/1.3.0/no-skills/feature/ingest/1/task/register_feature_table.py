# Databricks notebook source
# MAGIC %pip install databricks-sql-connector

# COMMAND ----------

from databricks import sql
import os

# Connect to Databricks SQL
connection = sql.connect(
    server_hostname=os.getenv("DATABRICKS_HOST"),
    http_path="",  # Use the HTTP path of your SQL warehouse
    access_token=os.getenv("DATABRICKS_TOKEN")
)

cursor = connection.cursor()

# Register the table as a feature table
cursor.execute("""
CREATE FEATURE TABLE IF NOT EXISTS workspace.mlpabd62957.transactions4adadd
AS SELECT * FROM workspace.mlpabd62957.transactions4adadd
""")

# Enable online access for low-latency lookup
cursor.execute("""
CREATE OR REFRESH ONLINE TABLE workspace.mlpabd62957.transactions4adadd
FROM FEATURES workspace.mlpabd62957.transactions4adadd
""")

cursor.close()
connection.close()