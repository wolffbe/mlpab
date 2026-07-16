"""Wait for synced table to finish syncing."""
from databricks.sdk import WorkspaceClient
import os, time
w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog, schema_name = schema.split('.')

synced_table_name = f"synced_tables/{catalog}.{schema_name}.featuresb1ea93_online"

print(f"Waiting for synced table: {synced_table_name}")
start = time.time()
last_state = None
while time.time() - start < 600:
    try:
        st_info = w.postgres.get_synced_table(name=synced_table_name)
        status = st_info.status
        st = str(status.detailed_state) if status and hasattr(status, 'detailed_state') else str(status)
    except Exception as e:
        st = f"error: {e}"
    if st != last_state:
        elapsed = int(time.time() - start)
        print(f"[{elapsed}s] Status: {st}")
        last_state = st
    st_upper = st.upper()
    if any(x in st_upper for x in ['ONLINE', 'ACTIVE', 'SYNCED_TABLE_ACTIVE']):
        print("Synced table is ACTIVE and ready!")
        break
    if any(x in st_upper for x in ['FAIL', 'OFFLINE_FAILED', 'PIPELINE_FAILED', 'SYNCED_TABLE_STOPPED']):
        print(f"Synced table issue: {st}")
        break
    time.sleep(20)

# Final status check
print()
try:
    st_info = w.postgres.get_synced_table(name=synced_table_name)
    print(f"Final status: {st_info.status}")
except Exception as e:
    print(f"Final check error: {e}")
