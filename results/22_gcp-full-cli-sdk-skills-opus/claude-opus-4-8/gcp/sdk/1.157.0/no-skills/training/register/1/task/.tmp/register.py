import os, json
import google.cloud.aiplatform as aiplatform
from google.cloud.aiplatform_v1 import ModelServiceClient
from google.cloud.aiplatform_v1.types import Model as ModelProto, ModelEvaluation as MEProto
from google.protobuf import struct_pb2

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

BASE_NAME = "churnmodelfd281c"
DISPLAY = f"{PREFIX}_{BASE_NAME}"
BUCKET = "cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"
ARTIFACT_URI = f"gs://{BUCKET}/{PREFIX}_{BASE_NAME}"

with open("data/metrics.json") as f:
    metrics = json.load(f)

aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
msc = ModelServiceClient(client_options=client_options, transport="rest")

parent = f"projects/{PROJECT}/locations/{LOCATION}"

# Upload the model with just the artifact (no serving container) via the gapic API,
# which — unlike the high-level SDK — does not require a serving_container_image_uri.
model_proto = ModelProto(
    display_name=DISPLAY,
    description="Churn logistic regression model registered from artifact.",
    artifact_uri=ARTIFACT_URI,
    container_spec={"image_uri": "gcr.io/google-samples/hello-app:1.0"},
    labels={"mlpab_prefix": PREFIX},
)
op = msc.upload_model(parent=parent, model=model_proto)
print("waiting for upload LRO ...")
resp = op.result(timeout=600)
model_resource_name = resp.model  # e.g. projects/../locations/../models/1234
version_id = resp.model_version_id
print("MODEL_RESOURCE", model_resource_name)
print("VERSION_ID", version_id)

# Attach metrics via a model evaluation import.
metrics_val = struct_pb2.Value()
metrics_val.struct_value.update(metrics)
eval_proto = MEProto(
    display_name=f"{PREFIX}_{BASE_NAME}_eval",
    metrics_schema_uri="gs://google-cloud-aiplatform/schema/modelevaluation/general_metrics_1.0.0.yaml",
    metrics=metrics_val,
)
imported = msc.import_model_evaluation(parent=model_resource_name, model_evaluation=eval_proto)
print("EVAL_NAME", imported.name)

# Read back through the high-level SDK to confirm.
m = aiplatform.Model(model_name=model_resource_name)
print("READBACK DISPLAY", m.display_name, "VERSION", m.version_id)
for ev in m.list_model_evaluations():
    print("READBACK_METRICS", dict(ev.metrics))

result = {"model_name": BASE_NAME, "version": int(version_id), "metrics": metrics}
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump(result, f, indent=2)
print("WROTE submission/answers.json")
print(json.dumps(result))
