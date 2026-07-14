import os, json
import google.cloud.aiplatform as aiplatform
from google.cloud.aiplatform_v1 import ModelServiceClient
from google.cloud.aiplatform_v1.types import Model as ModelProto
from google.protobuf import struct_pb2, field_mask_pb2

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
BASE_NAME = "churnmodelfd281c"
MODEL_RESOURCE = "projects/1014453977696/locations/***REDACTED***/models/2860474057659252736"

with open("data/metrics.json") as f:
    metrics = json.load(f)

aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")
msc = ModelServiceClient(client_options={"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}, transport="rest")

# 1) Attach metrics as a Vertex ML Metadata system.Metrics artifact tied to the model.
metrics_artifact = aiplatform.Artifact.create(
    schema_title="system.Metrics",
    display_name=f"{PREFIX}_{BASE_NAME}_metrics",
    metadata=dict(metrics),
    description=f"Evaluation metrics for model {BASE_NAME} version 1 ({MODEL_RESOURCE}).",
)
print("METRICS_ARTIFACT", metrics_artifact.resource_name)
print("ARTIFACT_METADATA", dict(metrics_artifact.metadata))

# 2) Also embed the metrics verbatim on the model-registry entry itself (description),
#    so the metrics are readable directly from the Model resource.
m = aiplatform.Model(model_name=MODEL_RESOURCE)
desc_obj = {
    "description": "Churn logistic regression model registered from artifact.",
    "metrics": metrics,
}
mp = ModelProto(name=m.versioned_resource_name, description=json.dumps(desc_obj),
                labels={"mlpab_prefix": PREFIX})
msc.update_model(model=mp, update_mask=field_mask_pb2.FieldMask(paths=["description", "labels"]))

# Read back
m2 = aiplatform.Model(model_name=MODEL_RESOURCE)
print("READBACK DISPLAY", m2.display_name, "VERSION", m2.version_id)
print("READBACK ARTIFACT_URI", m2.gca_resource.artifact_uri)
print("READBACK DESCRIPTION", m2.description)
arts = aiplatform.Artifact.list(filter=f'display_name="{PREFIX}_{BASE_NAME}_metrics"')
for a in arts:
    print("READBACK METRICS_ARTIFACT", a.display_name, dict(a.metadata))

result = {"model_name": BASE_NAME, "version": int(m2.version_id), "metrics": metrics}
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump(result, f, indent=2)
print("WROTE", json.dumps(result))
