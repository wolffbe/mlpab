import hopsworks
import json

# Load metrics and model
with open("./data/metrics.json", "r") as f:
    metrics = json.load(f)

# Connect to Hopsworks
project = hopsworks.login()
mr = project.get_model_registry()

# Register the model
model = mr.python.create_model(
    name="churnmodel0a639a",
    version=1,
    metrics=metrics,
    description="Churn prediction model trained on logistic regression."
)

# Upload the model artifact
model.save("./data/model.json")

# Write the deliverable
with open("./submission/answers.json", "w") as f:
    json.dump({
        "model_name": "churnmodel0a639a",
        "version": 1,
        "metrics": metrics
    }, f)