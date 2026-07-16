from databricks.sdk import WorkspaceClient
from databricks.sdk.service import serving

w = WorkspaceClient()

config = serving.EndpointCoreConfigInput(
    name="mlpabb2baa7_probe",
    served_entities=[
        serving.ServedEntityInput(
            entity_name="workspace.mlpabb2baa7.scorer83d9cf",
            entity_version="1",
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
    ],
)
try:
    w.serving_endpoints.create(name="mlpabb2baa7_probe", config=config)
    print("probe create ACCEPTED — name-specific gate; deleting probe")
    w.serving_endpoints.delete("mlpabb2baa7_probe")
except Exception as e:
    print("probe failed:", e)
