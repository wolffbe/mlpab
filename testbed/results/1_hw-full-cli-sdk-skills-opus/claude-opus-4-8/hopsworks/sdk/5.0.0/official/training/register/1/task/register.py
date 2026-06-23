import os, json, shutil
import hopsworks

MODEL_NAME = "churnmodelec96f5"
VERSION = 1

with open("data/metrics.json") as f:
    metrics = json.load(f)

project = hopsworks.login()
mr = project.get_model_registry()

# Stage the artifact in a clean dir to upload as the model content.
model_dir = ".tmp/model_artifact"
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.makedirs(model_dir, exist_ok=True)
shutil.copy("data/model.json", os.path.join(model_dir, "model.json"))

hw_model = mr.python.create_model(
    name=MODEL_NAME,
    version=VERSION,
    metrics=metrics,
    description="Churn logistic regression model (pre-trained artifact).",
)
hw_model.save(model_dir)

print("REGISTERED:", hw_model.name, "v", hw_model.version)
print("METRICS:", hw_model.training_metrics)

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"model_name": MODEL_NAME, "version": VERSION, "metrics": metrics}, f, indent=2)
print("WROTE submission/answers.json")
