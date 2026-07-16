"""Try multiple approaches to make predictions table available for online serving."""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

w = WorkspaceClient()
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG = SCHEMA.split(".")[0]
DB = SCHEMA.split(".")[1]

PRED_NAME = "airqpredf4aae3"
ONLINE_STORE = f"{PREFIX}-pred-store"
source_table = f"{CATALOG}.{DB}.{PRED_NAME}"

print(f"Source: {source_table}")
print(f"Online store: {ONLINE_STORE}")
print()

# Approach 1: Try publish_table with short online_table_name (no catalog.schema prefix)
print("--- Approach 1: publish_table with short name ---")
for table_name in [PRED_NAME, f"{DB}.{PRED_NAME}", source_table, None]:
    try:
        spec = PublishSpec(
            online_store=ONLINE_STORE,
            online_table_name=table_name,
            publish_mode=PublishSpecPublishMode.TRIGGERED,
        )
        result = w.feature_store.publish_table(
            source_table_name=source_table,
            publish_spec=spec,
        )
        print(f"SUCCESS with online_table_name={table_name!r}: {result}")
        break
    except Exception as e:
        print(f"FAIL online_table_name={table_name!r}: {e}")

print()

# Approach 2: Try w.online_tables.create() - see exact error
print("--- Approach 2: w.online_tables.create() ---")
try:
    ot = OnlineTable(
        name=source_table,
        spec=OnlineTableSpec(
            source_table_full_name=source_table,
            primary_key_columns=["date"],
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
        ),
    )
    result = w.online_tables.create(table=ot)
    print(f"SUCCESS: {result}")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")

print()

# Approach 3: Direct REST call to check what works
import requests
host = os.environ.get("DATABRICKS_HOST", "")
token = os.environ.get("DATABRICKS_TOKEN", "")
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

print("--- Approach 3: Direct REST publish ---")
r = requests.post(
    f"{host}/api/2.0/feature-store/tables/{source_table}/publish",
    headers=headers,
    json={"publish_spec": {"online_store": ONLINE_STORE, "publish_mode": "TRIGGERED"}},
)
print(f"REST publish (no online_table_name): {r.status_code} {r.text[:400]}")

print()

# Approach 4: Check if online table already exists
print("--- Approach 4: Check existing online tables ---")
r2 = requests.get(f"{host}/api/2.0/online-tables", headers=headers,
                  params={"catalog_name": CATALOG, "schema_name": DB})
print(f"List online tables: {r2.status_code} {r2.text[:500]}")

r3 = requests.get(f"{host}/api/2.0/feature-store/online-tables", headers=headers)
print(f"List FS online tables: {r3.status_code} {r3.text[:500]}")

r4 = requests.get(f"{host}/api/2.0/feature-store/tables/{source_table}", headers=headers)
print(f"Get FS table: {r4.status_code} {r4.text[:400]}")
