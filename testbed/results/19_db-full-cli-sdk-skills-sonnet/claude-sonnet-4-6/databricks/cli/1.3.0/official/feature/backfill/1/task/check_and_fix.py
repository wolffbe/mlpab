# Databricks notebook source
# COMMAND ----------
import requests
import json

table_name = "workspace.mlpab0442b8.accountse81ff1"
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# COMMAND ----------
# Check table count
count = spark.table(table_name).count()
dbutils.notebook.exit(f"Table row count: {count}")
