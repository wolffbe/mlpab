from databricks.sdk import WorkspaceClient
from databricks.sdk.service import ml
import os, time

w = WorkspaceClient()
catalog, schema = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
src = f'{catalog}.{schema}.scoredbfc4ef'
store_name = f'{prefix}-scorestore'
online_tbl = f'{catalog}.{schema}.scoredbfc4ef_online'

# 1. Create online store (Lakebase instance) if not present
existing = {s.name: s for s in w.feature_store.list_online_stores()}
if store_name in existing:
    store = existing[store_name]
    print('reuse store', store_name, store.state)
else:
    store = w.feature_store.create_online_store(
        online_store=ml.OnlineStore(name=store_name, capacity='CU_1'))
    print('created store', store.name, store.state)

# 2. Wait for store to be AVAILABLE
for _ in range(120):
    s = w.feature_store.get_online_store(name=store_name)
    if s.state == ml.OnlineStoreState.AVAILABLE:
        print('store AVAILABLE')
        break
    print('store state:', s.state)
    time.sleep(10)
else:
    raise RuntimeError('online store did not become available')

# 3. Publish the feature table to the online store
resp = w.feature_store.publish_table(
    source_table_name=src,
    publish_spec=ml.PublishSpec(
        online_store=store_name,
        online_table_name=online_tbl,
        publish_mode=ml.PublishSpecPublishMode.TRIGGERED,
    ),
)
print('publish response:', resp)
