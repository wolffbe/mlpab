import hopsworks
import json
import os

# Connect to Hopsworks
project = hopsworks.login()

# Get the model registry from the project
mr = project.get_model_registry()

# Set the model registry for the python signature
import hsml.python.signature
hsml.python.signature._mr = mr

# Read the metrics
with open("data/metrics.json", "r") as f:
    metrics = json.load(f)

# Create the model using the python signature
model = hsml.python.signature.create_model(
    name="churnmodel5bbb83",
    version=1,
    metrics=metrics,
    description="Churn prediction model"
)

# Save the model artifact
model.save("data/model.json")

# Create submission directory
os.makedirs("submission", exist_ok=True)

# Write the answers file
answers = {
    "model_name": "churnmodel5bbb83",
    "version": 1,
    "metrics": metrics
}

with open("submission/answers.json", "w") as f:
    json.dump(answers, f)

print("Model registered and submission file created.")
