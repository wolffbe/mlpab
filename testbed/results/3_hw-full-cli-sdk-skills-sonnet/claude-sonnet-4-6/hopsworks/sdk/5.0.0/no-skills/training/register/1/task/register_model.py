import hopsworks
import os
import json

# Connect to Hopsworks
project = hopsworks.login()

# Get model registry
mr = project.get_model_registry()

# Read metrics
with open("data/metrics.json") as f:
    metrics = json.load(f)

print(f"Metrics: {metrics}")

# Create model
model = mr.python.create_model(
    name="churnmodel674467",
    version=1,
    metrics=metrics,
    description="Churn prediction model"
)

# Save the model artifact
model.save("data/model.json")

print(f"Model registered: {model.name}, version: {model.version}")
print(f"Model metrics: {model.training_metrics}")

# Write answers
os.makedirs("submission", exist_ok=True)
answer = {
    "model_name": "churnmodel674467",
    "version": 1,
    "metrics": metrics
}
with open("submission/answers.json", "w") as f:
    json.dump(answer, f)

print("Done. submission/answers.json written.")
