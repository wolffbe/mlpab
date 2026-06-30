# Databricks notebook source
# COMMAND ----------
import json

# Add primary key constraint
try:
    spark.sql("""
        ALTER TABLE workspace.mlpabc8d80a.scaled7ecfaf
        ADD CONSTRAINT scaled7ecfaf_pk PRIMARY KEY(row_id)
    """)
    result = "primary key added"
except Exception as e:
    result = f"error: {e}"

dbutils.notebook.exit(json.dumps({"result": result}))
