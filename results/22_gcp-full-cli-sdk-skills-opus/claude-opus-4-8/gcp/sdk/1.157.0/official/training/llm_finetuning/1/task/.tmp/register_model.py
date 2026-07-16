import os, json
import google.cloud.aiplatform as aiplatform
from google.cloud import storage

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
BUCKET = "cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"

OUT_PREFIX = f"{PREFIX}_ftjob2e5343/outputs"
MODEL_DIR_PREFIX = f"{PREFIX}_ftmodel2e5343/model"

# Assemble a clean model artifact directory containing the fine-tuned weights.
sc = storage.Client(project=PROJECT)
bkt = sc.bucket(BUCKET)
src = bkt.blob(f"{OUT_PREFIX}/finetuned_model.npz")
dst = bkt.blob(f"{MODEL_DIR_PREFIX}/finetuned_model.npz")
bkt.copy_blob(src, bkt, dst.name)
print("model artifact:", f"gs://{BUCKET}/{dst.name}")

metrics = json.loads(bkt.blob(f"{OUT_PREFIX}/metrics.json").download_as_text())
print("metrics:", metrics)

aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

# Vertex label values disallow '.', so encode for labels; keep exact JSON in description.
labels = {k: str(v).replace(".", "_") for k, v in metrics.items()}

model = aiplatform.Model.upload(
    display_name=f"{PREFIX}_ftmodel2e5343",
    artifact_uri=f"gs://{BUCKET}/{MODEL_DIR_PREFIX}",
    serving_container_image_uri="gcr.io/google-containers/pause:3.2",
    description=json.dumps(metrics),
    labels=labels,
    version_aliases=["v1"],
    is_default_version=True,
    sync=True,
)
print("MODEL_RESOURCE", model.resource_name)
print("MODEL_DISPLAY", model.display_name)
print("MODEL_VERSION", model.version_id)
print("MODEL_LABELS", model.labels)
print("MODEL_DESCRIPTION", model.description)
