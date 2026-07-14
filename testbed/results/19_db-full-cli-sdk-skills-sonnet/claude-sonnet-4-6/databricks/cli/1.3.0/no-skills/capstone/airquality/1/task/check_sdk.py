# Databricks notebook source

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

results = []

# Find online/synced table related attributes
attrs = [a for a in dir(w) if any(x in a.lower() for x in ['sync', 'online', 'feature'])]
results.append(f"SDK attrs: {attrs}")

# Check catalog service
import databricks.sdk.service.catalog as cat
catalog_attrs = [a for a in dir(cat) if any(x in a.lower() for x in ['sync', 'online', 'feature'])]
results.append(f"Catalog service: {catalog_attrs}")

# Write results to volume
with open("/Volumes/workspace/mlpabd7768b/airqdata/sdk_check.txt", "w") as f:
    f.write("\n".join(results))
    f.write("\n\n")

    # Try online_tables
    if hasattr(w, 'online_tables'):
        f.write("has online_tables: YES\n")
        f.write(f"online_tables methods: {[m for m in dir(w.online_tables) if not m.startswith('_')]}\n")

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
            f.write(f"Created online table: {result}\n")
        except Exception as e:
            f.write(f"Failed to create online table: {e}\n")
    else:
        f.write("has online_tables: NO\n")

    # Try synced_tables
    if hasattr(w, 'synced_tables'):
        f.write("has synced_tables: YES\n")
        f.write(f"synced_tables methods: {[m for m in dir(w.synced_tables) if not m.startswith('_')]}\n")
    else:
        f.write("has synced_tables: NO\n")

print("Written to /Volumes/workspace/mlpabd7768b/airqdata/sdk_check.txt")
dbutils.notebook.exit("done")
