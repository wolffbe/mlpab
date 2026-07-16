import sys, time
sys.path.insert(0, 'scripts')
from common import *
from databricks.sdk.service import ml

STORE = f'{PREFIX}-online-store'

try:
    s = w.feature_store.get_online_store(name=STORE)
    print('exists:', s.name, s.state)
except Exception:
    s = w.feature_store.create_online_store(online_store=ml.OnlineStore(name=STORE, capacity='CU_1'))
    print('create initiated:', s.name, s.state)

# wait until AVAILABLE
for _ in range(120):
    s = w.feature_store.get_online_store(name=STORE)
    if s.state == ml.OnlineStoreState.AVAILABLE:
        print('AVAILABLE:', s.name)
        break
    print('state:', s.state)
    time.sleep(15)
else:
    raise RuntimeError('online store did not become available')
