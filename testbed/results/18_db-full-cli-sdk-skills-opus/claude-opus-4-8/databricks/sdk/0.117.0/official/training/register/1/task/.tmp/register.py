import json, io
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as cat
from databricks.sdk.service.ml import ModelVersionTag

w = WorkspaceClient()
CATALOG, SCHEMA = "workspace", "mlpabac627f"
MODEL = "churnmodel7948d0"
VOL = "churn_artifacts"

with open("data/metrics.json") as f:
    metrics = json.load(f)
with open("data/model.json", "rb") as f:
    artifact = f.read()

# 1. managed volume in the run schema for the artifact
try:
    w.volumes.create(catalog_name=CATALOG, schema_name=SCHEMA, name=VOL,
                     volume_type=cat.VolumeType.MANAGED)
    print("volume created")
except Exception as e:
    print("volume:", str(e)[:140])

base = f"/Volumes/{CATALOG}/{SCHEMA}/{VOL}/{MODEL}"
try:
    w.files.create_directory(base)
except Exception as e:
    print("mkdir:", str(e)[:140])
artifact_path = f"{base}/model.json"
w.files.upload(artifact_path, io.BytesIO(artifact), overwrite=True)
print("uploaded:", artifact_path)

# 2. registered model in the model registry
try:
    w.model_registry.create_model(name=MODEL)
    print("model created")
except Exception as e:
    print("model:", str(e)[:160])

# 3. model version, metrics attached as tags, artifact as content
tags = [ModelVersionTag(key=k, value=str(v)) for k, v in metrics.items()]
resp = w.model_registry.create_model_version(name=MODEL, source=base, tags=tags)
mv = resp.model_version
print("version:", mv.version)
print("source:", mv.source)
print("tags:", mv.tags)
