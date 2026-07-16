import hopsworks
import json
import os
import shutil

# Connect to Hopsworks
project = hopsworks.login()

# Read metrics
with open("data/metrics.json") as f:
    metrics = json.load(f)

print(f"Metrics: {metrics}")

# Get model registry
mr = project.get_model_registry()

# Create the model
model = mr.sklearn.create_model(
    name="churnmodel674467",
    version=1,
    metrics=metrics,
    description="Churn prediction model"
)

# Create a temp dir to put the model artifact
model_dir = "./tmp_model_dir"
os.makedirs(model_dir, exist_ok=True)
shutil.copy("data/model.json", model_dir)

# Save the model (uploads artifacts)
model.save(model_dir)

print(f"Model registered: {model.name}, version: {model.version}")
print(f"Metrics: {model.training_metrics}")

# Write submission
result = {
    "model_name": model.name,
    "version": model.version,
    "metrics": metrics
}

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Written submission/answers.json: {result}")
