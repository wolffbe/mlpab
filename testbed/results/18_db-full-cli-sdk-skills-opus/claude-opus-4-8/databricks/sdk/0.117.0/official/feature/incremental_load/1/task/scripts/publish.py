import sys
sys.path.insert(0, 'scripts')
from common import *
from databricks.sdk.service import ml

STORE = f'{PREFIX}-online-store'
SRC = f'{CAT}.{SCH}.incrementalb48074'
ONLINE = f'{CAT}.{SCH}.incrementalb48074_online'

spec = ml.PublishSpec(
    online_store=STORE,
    online_table_name=ONLINE,
    publish_mode=ml.PublishSpecPublishMode.TRIGGERED,
)
resp = w.feature_store.publish_table(source_table_name=SRC, publish_spec=spec)
print('publish response:', resp.as_dict())
