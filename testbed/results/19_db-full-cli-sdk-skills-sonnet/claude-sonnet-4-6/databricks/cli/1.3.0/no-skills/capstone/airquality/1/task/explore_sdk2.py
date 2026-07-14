# Databricks notebook source

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# Find online/synced table related attributes
attrs = [a for a in dir(w) if any(x in a.lower() for x in ['sync', 'online', 'feature'])]
print("Relevant SDK attributes:", attrs)

# COMMAND ----------

# Check what classes are in catalog service
import databricks.sdk.service.catalog as cat
catalog_attrs = [a for a in dir(cat) if any(x in a.lower() for x in ['sync', 'online', 'feature'])]
print("Catalog service classes:", catalog_attrs)

# COMMAND ----------

# Try to use online_tables if available
if hasattr(w, 'online_tables'):
    print("online_tables methods:", [m for m in dir(w.online_tables) if not m.startswith('_')])
    try:
        from databricks.sdk.service import catalog as cat_svc
        result = w.online_tables.create(
            name="workspace.mlpabd7768b.airqpredfdfb59_online",
            spec=cat_svc.OnlineTableSpec(
                source_table_full_name="workspace.mlpabd7768b.airqpredfdfb59",
                primary_key_columns=["date"],
                run_triggered=cat_svc.OnlineTableSpecTriggeredSchedulingPolicy()
            )
        )
        print(f"Created: {result}")
    except Exception as e:
        print(f"Failed to create online table: {e}")
else:
    print("No online_tables in SDK")

# COMMAND ----------

dbutils.notebook.exit("done")
