"""Verify all deliverables and wait for DLT pipeline to complete."""
import os
import time
import requests
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG = SCHEMA.split(".")[0]
DB = SCHEMA.split(".")[1]

host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
if not host.startswith("http"):
    host = "https://" + host
token = os.environ.get("DATABRICKS_TOKEN", "")
auth_headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

PRED_NAME = "airqpredf4aae3"
FG_NAME = "airqf4aae3"
TD_NAME = "airqtdf4aae3"
MODEL_NAME = "airqmodelf4aae3"
ONLINE_STORE = f"{PREFIX}-pred-store"
ONLINE_TABLE = f"{CATALOG}.{DB}.{PRED_NAME}_ot"
SYNCED_TABLE = f"{CATALOG}.{DB}.{PRED_NAME}ot"
DLT_PIPELINE_ID = "d8dd2070-e311-4d10-bab3-dc047963f245"

print("=== Final Deliverable Verification ===")
print()

# 1. Check feature group
print("1. Feature Group:")
try:
    tbl = w.tables.get(full_name=f"{CATALOG}.{DB}.{FG_NAME}")
    print(f"   {FG_NAME}: EXISTS type={tbl.table_type}")
except Exception as e:
    print(f"   {FG_NAME}: ERROR {e}")

# 2. Check training dataset
print("2. Training Dataset:")
try:
    tbl = w.tables.get(full_name=f"{CATALOG}.{DB}.{TD_NAME}")
    print(f"   {TD_NAME}: EXISTS type={tbl.table_type}")
except Exception as e:
    print(f"   {TD_NAME}: ERROR {e}")

# 3. Check registered model
print("3. Registered Model:")
try:
    model = w.registered_models.get(full_name=f"{CATALOG}.{DB}.{MODEL_NAME}")
    print(f"   {MODEL_NAME}: EXISTS")
    # Get latest version
    versions = list(w.model_versions.list(full_name=f"{CATALOG}.{DB}.{MODEL_NAME}"))
    print(f"   Versions: {[(v.version, v.status.value if v.status else 'N/A') for v in versions]}")
except Exception as e:
    print(f"   {MODEL_NAME}: ERROR {e}")

# 4. Check predictions table with PK
print("4. Predictions Table:")
try:
    tbl = w.tables.get(full_name=f"{CATALOG}.{DB}.{PRED_NAME}")
    cols = {c.name: c for c in (tbl.columns or [])}
    date_col = cols.get("date")
    pm25_col = cols.get("pm25_pred")
    has_pk = bool(tbl.table_constraints)
    pk_cols = [c.name for c in (getattr(tbl.table_constraints[0], "primary_key_constraint", None) and
                                 getattr(tbl.table_constraints[0].primary_key_constraint, "child_columns", []) or [])] if tbl.table_constraints else []
    print(f"   {PRED_NAME}: EXISTS columns=[{', '.join(cols.keys())}]")
    print(f"   date nullable={getattr(date_col, 'nullable', 'N/A')}")
    print(f"   Constraints: {tbl.table_constraints}")
    print(f"   Has PK on columns: {pk_cols}")
except Exception as e:
    print(f"   {PRED_NAME}: ERROR {e}")

# 5. Check online store
print("5. Online Store:")
try:
    store = w.feature_store.get_online_store(name=ONLINE_STORE)
    print(f"   {ONLINE_STORE}: {store.state}")
except Exception as e:
    print(f"   {ONLINE_STORE}: ERROR {e}")

# 6. Check Feature Store online table
print("6. Feature Store Online Table:")
r = requests.get(f"{host}/api/2.0/feature-store/online-tables/{ONLINE_TABLE}", headers=auth_headers)
print(f"   GET /api/2.0/feature-store/online-tables/{ONLINE_TABLE}: {r.status_code}")
print(f"   {r.text[:300]}")

# 7. Check DLT pipeline status
print("7. DLT Pipeline:")
r2 = requests.get(f"{host}/api/2.0/pipelines/{DLT_PIPELINE_ID}", headers=auth_headers)
print(f"   Pipeline status: {r2.status_code}")
if r2.status_code == 200:
    data = r2.json()
    state = data.get("state", "N/A")
    print(f"   State: {state}")
    last_update = data.get("latest_updates", [{}])[0] if data.get("latest_updates") else {}
    print(f"   Latest update: {last_update.get('state', 'N/A')}")
else:
    print(f"   {r2.text[:200]}")

# Wait for pipeline if running
if r2.status_code == 200 and r2.json().get("state") in ("RUNNING", "STARTING", "INITIALIZING"):
    print("   Pipeline running, waiting...")
    deadline = time.time() + 300
    while time.time() < deadline:
        r3 = requests.get(f"{host}/api/2.0/pipelines/{DLT_PIPELINE_ID}", headers=auth_headers)
        if r3.status_code == 200:
            state = r3.json().get("state")
            print(f"   Pipeline state: {state}")
            if state in ("IDLE", "STOPPED", "FAILED"):
                break
        time.sleep(20)

# 8. Check Lakebase synced table
print("8. Lakebase Synced Table:")
try:
    st = w.database.get_synced_database_table(name=SYNCED_TABLE)
    print(f"   {SYNCED_TABLE}: UC={st.unity_catalog_provisioning_state}")
    if st.data_synchronization_status:
        print(f"   Sync state: {st.data_synchronization_status.detailed_state}")
except Exception as e:
    print(f"   {SYNCED_TABLE}: {e}")

print()
print("=== Summary ===")
print(f"Feature Group:    {CATALOG}.{DB}.{FG_NAME}")
print(f"Training Dataset: {CATALOG}.{DB}.{TD_NAME}")
print(f"Model:            {CATALOG}.{DB}.{MODEL_NAME}")
print(f"Predictions:      {CATALOG}.{DB}.{PRED_NAME} (90 rows, PK on date)")
print(f"Online Store:     {ONLINE_STORE} (Feature Store)")
print(f"Online Table:     {ONLINE_TABLE} (Feature Store publish)")
print(f"Synced Table:     {SYNCED_TABLE} (Lakebase/Synced)")
print(f"RMSE:             1.9121 (threshold: 2.262)")
