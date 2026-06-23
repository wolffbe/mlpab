#!/usr/bin/env python3

import hopsworks
import json
import os

# Load metrics and model
with open("data/metrics.json", "r") as f:
    metrics = json.load(f)

with open("data/model.json", "r") as f:
    model = json.load(f)

# Log in to Hopsworks
project = hopsworks.login()

# Access the model registry
mr = project.get_model_registry()

# Register the model
model_dir = "/tmp/churnmodel0a639a_v1"
os.makedirs(model_dir, exist_ok=True)

# Write model artifact to a temporary directory
model_path = os.path.join(model_dir, "model.json")
with open(model_path, "w") as f:
    json.dump(model, f)

# Register the model
churn_model = mr.python.create_model(
    name="churnmodel0a639a",
    version=1,
    metrics=metrics,
    description="Logistic regression model for churn prediction",
    input_example=model["features"],
    model_schema=None
)

# Upload the model artifact
churn_model.save(model_dir)

# Write submission/answers.json
answers = {
    "model_name": "churnmodel0a639a",
    "version": 1,
    "metrics": metrics
}

with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)