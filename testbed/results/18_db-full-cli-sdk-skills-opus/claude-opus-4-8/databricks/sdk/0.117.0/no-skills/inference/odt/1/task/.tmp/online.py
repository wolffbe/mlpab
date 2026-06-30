from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as c
import os

w = WorkspaceClient()
catalog, schema = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
src = f'{catalog}.{schema}.scoredbfc4ef'
online_name = f'{catalog}.{schema}.scoredbfc4ef_online'

spec = c.OnlineTableSpec(
    source_table_full_name=src,
    primary_key_columns=['request_id'],
    perform_full_copy=True,
    run_triggered=c.OnlineTableSpecTriggeredSchedulingPolicy(),
)
ot = c.OnlineTable(name=online_name, spec=spec)

print('creating online table', online_name)
res = w.online_tables.create_and_wait(table=ot)
print('state:', res.unity_catalog_provisioning_state)
print('serving_url:', res.table_serving_url)
print('status:', res.status)
