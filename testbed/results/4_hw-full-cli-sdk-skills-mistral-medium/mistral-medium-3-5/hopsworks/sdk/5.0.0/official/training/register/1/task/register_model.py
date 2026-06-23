#!/usr/bin/env python3
import json
import os
import hopsworks

# Read the metrics
with open('data/metrics.json', 'r') as f:
    metrics = json.load(f)

# Connect to Hopsworks
hopsworks.login()

# Get the model registry
reg_api = hopsworks.hsml.core.model_registry_api.ModelRegistryApi()
reg = reg_api.get()

# Create the model with metrics
model_name = "churnmodel5bbb83"
model = reg.python.create_model(
    name=model_name,
    version=1,
    metrics=metrics,
    description="Churn prediction model"
)

# Save the model artifact
model.save(model_path="data/model.json")

# Write the submission file
submission = {
    "model_name": model_name,
    "version": 1,
    "metrics": metrics
}

os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as f:
    json.dump(submission, f, indent=2)

print("Model registered and submission file created successfully.")
print(f"Submission: {json.dumps(submission, indent=2)}")
