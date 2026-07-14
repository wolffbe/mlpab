# Databricks notebook source
# COMMAND ----------
import json

schema = "workspace.mlpabae7d2f"
online_table_full = f"{schema}.profilesaa70e4_online"
output_path = "/Volumes/workspace/mlpabae7d2f/mlpabae7d2f_vol/answers.json"

lookup_keys = [
    "A0003", "A0005", "A0012", "A0015", "A0023", "A0030", "A0031", "A0034",
    "A0048", "A0049", "A0055", "A0063", "A0066", "A0072", "A0085", "A0090",
    "A0103", "A0109", "A0112", "A0113"
]

# COMMAND ----------
# Query the online table for all lookup keys
keys_str = ", ".join(f"'{k}'" for k in lookup_keys)
rows = spark.sql(f"""
    SELECT account_id, f1, f2, f3, f4
    FROM {online_table_full}
    WHERE account_id IN ({keys_str})
""").collect()

print(f"Retrieved {len(rows)} rows from online table {online_table_full}")

# Build the results dict
vectors = {}
for row in rows:
    vectors[row.account_id] = [float(row.f1), float(row.f2), float(row.f3), float(row.f4)]

print(f"Found {len(vectors)} vectors for {len(lookup_keys)} lookup keys")

# Verify all keys are present
missing = [k for k in lookup_keys if k not in vectors]
if missing:
    print(f"WARNING: Missing keys: {missing}")

# COMMAND ----------
# Write results to volume
output = {"vectors": vectors}
with open(output_path, "w") as f:
    json.dump(output, f)

print(f"Results written to {output_path}")
print(json.dumps(output, indent=2))

dbutils.notebook.exit(json.dumps(output))
