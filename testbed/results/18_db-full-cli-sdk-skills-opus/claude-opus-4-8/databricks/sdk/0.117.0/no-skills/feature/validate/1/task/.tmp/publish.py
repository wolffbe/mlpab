import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import ml

w = WorkspaceClient()
cat, sch = 'workspace', 'mlpabbaf12e'
src = f'{cat}.{sch}.events88b330'
store_name = 'mlpabbaf12e-events88b330-store'
online_tbl = f'{cat}.{sch}.events88b330_online'

# Create online store (Lakebase-backed). Try a capacity value.
store = None
for cap in ['CU_1', 'CU_2', 'CU_4']:
    try:
        store = w.feature_store.create_online_store(ml.OnlineStore(name=store_name, capacity=cap))
        print('created store with capacity', cap, '->', store.state)
        break
    except Exception as e:
        print('cap', cap, 'failed:', str(e)[:200])

if store is None:
    # maybe it already exists
    store = w.feature_store.get_online_store(store_name)
    print('using existing store:', store.state)

# wait for available
for _ in range(60):
    s = w.feature_store.get_online_store(store_name)
    print('store state:', s.state)
    if str(s.state) in ('OnlineStoreState.AVAILABLE', 'AVAILABLE'):
        break
    time.sleep(15)

# publish the feature table to the online store
resp = w.feature_store.publish_table(
    source_table_name=src,
    publish_spec=ml.PublishSpec(
        online_store=store_name,
        online_table_name=online_tbl,
        publish_mode=ml.PublishSpecPublishMode.TRIGGERED,
    ),
)
print('publish resp:', resp)
