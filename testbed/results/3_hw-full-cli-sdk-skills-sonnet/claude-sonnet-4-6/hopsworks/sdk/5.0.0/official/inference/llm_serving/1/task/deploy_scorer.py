import hopsworks
import os
import json

DEPLOYMENT_NAME = "scorer359507"
MODEL_NAME = "scorer359507"

project = hopsworks.login()
mr = project.get_model_registry()
ms = project.get_model_serving()

# Check if deployment already exists and stop/delete it
try:
    existing = ms.get_deployment(DEPLOYMENT_NAME)
    print(f"Found existing deployment {DEPLOYMENT_NAME}, deleting...")
    try:
        existing.stop(await_stopped=120)
    except Exception as e:
        print(f"Stop error (ignored): {e}")
    existing.delete()
    print("Deleted existing deployment.")
except Exception as e:
    print(f"No existing deployment found: {e}")

# Register model
model = mr.python.create_model(
    name=MODEL_NAME,
    description="Trigram language model scorer",
    input_example={"instances": ["hello world"]},
)
model.save("./model_dir")
print(f"Model registered: {model.name} v{model.version}")

# Upload predictor.py to the model's Files directory in Hopsworks FS
script_dir = f"/Projects/{project.name}/Models/{model.name}/{model.version}/Files"
print(f"Uploading predictor.py to {script_dir}")
project.get_dataset_api().upload("./predictor.py", script_dir, overwrite=True)
script_file = f"{script_dir}/predictor.py"
print(f"Uploaded predictor to: {script_file}")

# Deploy
from hsml.resources import PredictorResources, Resources
from hsml.scaling_config import PredictorScalingConfig, ScaleMetric

deployment = model.deploy(
    name=DEPLOYMENT_NAME,
    script_file=script_file,
    resources=PredictorResources(
        requests=Resources(cores=1, memory=1024, gpus=0),
        limits=Resources(cores=2, memory=2048, gpus=0),
    ),
    scaling_configuration=PredictorScalingConfig(
        min_instances=1,
        max_instances=1,
        scale_metric=ScaleMetric.CONCURRENCY,
        target=10,
    ),
)
print(f"Deployment created: {deployment.name}")

deployment.start(await_running=600)
print(f"Deployment running: {deployment.is_running()}")

# Load payloads
with open("./data/payloads.json") as f:
    payloads = json.load(f)

print(f"Loaded {len(payloads)} payloads")

# Get predictions for each payload
responses = []
for i, payload in enumerate(payloads):
    print(f"Predicting payload {i+1}: {payload[:50]}...")
    result = deployment.predict(inputs=[payload])
    print(f"  Raw result: {result}")
    preds = result.get("predictions", result)
    if isinstance(preds, list):
        responses.append(preds[0])
    else:
        responses.append(preds)

print(f"All responses: {responses}")

# Write submission
os.makedirs("./submission", exist_ok=True)
answer = {
    "endpoint_name": DEPLOYMENT_NAME,
    "responses": responses,
}
with open("./submission/answers.json", "w") as f:
    json.dump(answer, f, indent=2)

print(f"Written to submission/answers.json")
print(json.dumps(answer, indent=2))
