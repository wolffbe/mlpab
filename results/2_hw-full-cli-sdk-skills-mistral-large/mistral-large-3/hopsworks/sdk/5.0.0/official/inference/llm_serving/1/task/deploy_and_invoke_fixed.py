#!/usr/bin/env python3
"""
Deploy the scorer as a real-time endpoint named `scorer06901d` and invoke it on the payloads.
"""
import hopsworks
import json
import os
import shutil

# Load payloads
with open("./data/payloads.json", "r") as f:
    payloads = json.load(f)

# Login to Hopsworks
project = hopsworks.login()
mr = project.get_model_registry()
ms = project.get_model_serving()

# Create a clean directory for the model artifact
model_dir = "./model_artifact"
os.makedirs(model_dir, exist_ok=True)
shutil.copy("./data/scorer.py", model_dir)

# Create a Python model
model = mr.python.create_model(
    name="scorer_model_06901d",
    version=1,
    description="A deterministic pure-python language-model scorer."
)

# Save the model (only scorer.py is the artifact)
model.save(model_dir)

# Create a predictor
predictor = ms.create_predictor(
    model=model,
    name="scorer06901d",
    script_file="scorer.py",  # Use the scorer.py as the predictor script
    environment="python",
)

# Deploy the predictor
deployment = predictor.deploy()

# Wait for the deployment to be ready
deployment.wait_for()

# Invoke the endpoint on each payload
responses = []
for payload in payloads:
    response = deployment.predict(payload)
    responses.append(response["score"])

# Write the results to submission/answers.json
with open("./submission/answers.json", "w") as f:
    json.dump({
        "endpoint_name": "scorer06901d",
        "responses": responses
    }, f)

print("Deployment and invocation completed successfully.")