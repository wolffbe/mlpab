import datetime
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import serving

w = WorkspaceClient()

config = serving.EndpointCoreConfigInput(
    name="scorer83d9cf",
    served_entities=[
        serving.ServedEntityInput(
            entity_name="workspace.mlpabb2baa7.scorer83d9cf",
            entity_version="1",
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
    ],
)

waiter = None
for attempt in range(12):
    try:
        waiter = w.serving_endpoints.create(name="scorer83d9cf", config=config)
        print("create accepted on attempt", attempt + 1)
        break
    except Exception as e:
        print(f"attempt {attempt + 1} failed: {e}")
        time.sleep(60)

if waiter is None:
    raise SystemExit("endpoint creation never accepted")

ep = waiter.result(timeout=datetime.timedelta(minutes=40))
print("endpoint:", ep.name)
print("state:", ep.state)
