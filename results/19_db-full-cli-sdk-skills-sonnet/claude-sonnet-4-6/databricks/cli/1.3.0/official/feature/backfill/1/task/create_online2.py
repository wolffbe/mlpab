# Databricks notebook source
# COMMAND ----------
import requests
import json

table_name = "workspace.mlpab0442b8.accountse81ff1"
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
base_url = f"https://{host}"

results = {}

# COMMAND ----------
# Try different scheduling_policy formats

payloads = {
    "enum_triggered": {
        "name": table_name,
        "database_instance_name": "mlpab0442b8-lakebase",
        "logical_database_name": "mlpab0442b8",
        "spec": {
            "source_table_full_name": table_name,
            "primary_key_columns": ["row_id"],
            "scheduling_policy_type": "TRIGGERED"
        }
    },
    "enum_at_top": {
        "name": table_name,
        "database_instance_name": "mlpab0442b8-lakebase",
        "logical_database_name": "mlpab0442b8",
        "scheduling_policy_type": "TRIGGERED",
        "spec": {
            "source_table_full_name": table_name,
            "primary_key_columns": ["row_id"]
        }
    },
    "triggered_true": {
        "name": table_name,
        "database_instance_name": "mlpab0442b8-lakebase",
        "logical_database_name": "mlpab0442b8",
        "spec": {
            "source_table_full_name": table_name,
            "primary_key_columns": ["row_id"],
            "run_triggered": {"user_triggered": True}
        }
    },
    "pipeline_triggered": {
        "name": table_name,
        "database_instance_name": "mlpab0442b8-lakebase",
        "logical_database_name": "mlpab0442b8",
        "spec": {
            "source_table_full_name": table_name,
            "primary_key_columns": ["row_id"],
            "run_triggered": {"pipeline_type": "TRIGGERED"}
        }
    }
}

for key, payload in payloads.items():
    r = requests.post(f"{base_url}/api/2.0/database/synced_tables", headers=headers, json=payload)
    results[key] = {"status": r.status_code, "response": r.text[:200]}

# Also try GET to check database instance
r_inst = requests.get(f"{base_url}/api/2.0/database/instances/mlpab0442b8-lakebase", headers=headers)
results["instance_check"] = {"status": r_inst.status_code, "response": r_inst.text[:200]}

# Try list database instances
r_list = requests.get(f"{base_url}/api/2.0/database/instances", headers=headers)
results["instances_list"] = {"status": r_list.status_code, "response": r_list.text[:300]}

output = json.dumps(results, indent=2)
print(output)
dbutils.notebook.exit(output)
