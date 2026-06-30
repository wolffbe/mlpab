# Databricks notebook source
# COMMAND ----------
import json, requests

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

results = {}

# Check if there's a catalog associated with the Lakebase project
# The existing synced table project was mlpabc1d5e2-scaled
# Let's check catalogs
for project_id in ["mlpabc1d5e2-scaled", "mlpabf1452c-feat", "mlpabefbb2e-feat"]:
    try:
        r = requests.get(
            f"{host}/api/2.0/postgres/projects/{project_id}/catalogs",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        results[f'catalogs_{project_id}'] = {"status": r.status_code, "body": r.text[:400]}
    except Exception as e:
        results[f'catalogs_{project_id}'] = {"error": str(e)}

# Also check if there's a lakebase catalog associated with any existing projects
try:
    r = requests.get(
        f"{host}/api/2.0/postgres/catalogs",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    results['list_all_catalogs'] = {"status": r.status_code, "body": r.text[:800]}
except Exception as e:
    results['list_all_catalogs'] = {"error": str(e)}

# Check what catalogs are registered via create-catalog
try:
    r = requests.get(
        f"{host}/api/2.1/unity-catalog/catalogs",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30
    )
    resp = r.json()
    catalogs = [{c.get("name"): c.get("catalog_type")} for c in resp.get("catalogs", [])]
    results['uc_catalogs'] = catalogs
except Exception as e:
    results['uc_catalogs'] = str(e)

# Try creating a catalog linked to our database
try:
    r2 = requests.post(
        f"{host}/api/2.0/postgres/catalogs?catalog_id=mlpabf1452c",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "database": "projects/mlpabf1452c-feat/branches/production/databases/databricks-postgres"
        },
        timeout=30
    )
    results['create_catalog'] = {"status": r2.status_code, "body": r2.text[:400]}
except Exception as e:
    results['create_catalog'] = {"error": str(e)}

dbutils.notebook.exit(json.dumps(results))
