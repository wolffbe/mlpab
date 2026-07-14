import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import serving

w = WorkspaceClient()

ep = w.serving_endpoints.create_and_wait(
    name="scorer83d9cf",
    config=serving.EndpointCoreConfigInput(
        name="scorer83d9cf",
        served_entities=[
            serving.ServedEntityInput(
                entity_name="workspace.mlpabb2baa7.scorer83d9cf",
                entity_version="1",
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ]
    ),
    timeout=datetime.timedelta(minutes=45),
)
print("endpoint:", ep.name)
print("state:", ep.state)
