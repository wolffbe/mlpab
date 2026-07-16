import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
SCHEMA = "workspace.mlpab6cf45f"
ONLINE = f"{SCHEMA}.profilesf45007_online"
PIPELINE = "79d976ea-eddb-40a3-8f55-5e8a520f7b9e"

deadline = time.time() + 1200
while True:
    st = w.database.get_synced_database_table(name=ONLINE)
    dss = st.data_synchronization_status
    ucp = st.unity_catalog_provisioning_state
    cont = dss.detailed_state if dss else None
    print("uc_state:", ucp, "| sync:", cont, flush=True)
    if cont is not None and str(cont).endswith("ONLINE") or str(cont).endswith("ONLINE_NO_PENDING_UPDATE"):
        print("SYNCED")
        break
    if time.time() > deadline:
        print("timeout; last:", dss)
        break
    time.sleep(15)

st = w.database.get_synced_database_table(name=ONLINE)
print("=== final synced table ===")
print("name:", st.name)
print("db_instance:", st.effective_database_instance_name)
print("logical_db:", st.effective_logical_database_name)
print("spec:", st.spec)
print("sync_status:", st.data_synchronization_status)
