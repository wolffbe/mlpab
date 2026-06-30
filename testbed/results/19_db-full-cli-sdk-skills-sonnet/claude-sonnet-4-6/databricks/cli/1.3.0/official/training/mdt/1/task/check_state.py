# Databricks notebook source
# COMMAND ----------
import json

# Check available packages
import pkg_resources
packages = sorted([d.project_name for d in pkg_resources.working_set])
fs_packages = [p for p in packages if 'feature' in p.lower() or 'databrick' in p.lower()]

# Check table constraints
try:
    constraints = spark.sql("SHOW CONSTRAINTS ON workspace.mlpabc8d80a.scaled7ecfaf").collect()
    constraint_info = str(constraints)
except Exception as e:
    constraint_info = str(e)

# Check table description
desc = spark.sql("DESCRIBE EXTENDED workspace.mlpabc8d80a.scaled7ecfaf").collect()
desc_str = str([(r[0], r[1]) for r in desc if r[0] in ['Table Properties', 'Constraints', 'primary_key', 'row_filter']])

result = {
    "fs_packages": fs_packages,
    "constraint_info": constraint_info,
    "desc_relevant": desc_str
}

dbutils.notebook.exit(json.dumps(result))
