import os, time
import databricks.sdk
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy)

w = databricks.sdk.WorkspaceClient()
CAT, SCH = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
src = f"{CAT}.{SCH}.recs2ead15"
ot_name = f"{CAT}.{SCH}.recs2ead15_online"

spec = OnlineTableSpec(
    source_table_full_name=src,
    primary_key_columns=["rec_id"],
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
    perform_full_copy=True,
)
ot = OnlineTable(name=ot_name, spec=spec)
print("creating online table", ot_name)
res = w.online_tables.create_and_wait(table=ot)
print("created:", res.name)
print("state:", res.status.detailed_state if res.status else None)
