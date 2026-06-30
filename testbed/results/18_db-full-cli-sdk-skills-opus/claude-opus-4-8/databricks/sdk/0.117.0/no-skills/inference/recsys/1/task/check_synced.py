import os, time
import databricks.sdk
w = databricks.sdk.WorkspaceClient()
CAT, SCH = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
synced_name = f"{CAT}.{SCH}.recs2ead15_synced"

for _ in range(60):
    st = w.database.get_synced_database_table(name=synced_name)
    prov = st.unity_catalog_provisioning_state
    dss = st.data_synchronization_status
    detail = dss.detailed_state if dss else None
    print("prov:", prov, "| sync:", detail)
    if str(prov) in ("ProvisioningInfoState.ACTIVE", "ACTIVE") and detail and "ONLINE" in str(detail):
        print("READY")
        break
    if detail and "FAILED" in str(detail):
        print("SYNC FAILED")
        break
    time.sleep(15)
