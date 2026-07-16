import json

import hopsworks

with open("data/metrics.json") as f:
    metrics = json.load(f)

project = hopsworks.login()
mr = project.get_model_registry()

model = mr.python.create_model(
    name="churnmodel7ed7ab",
    version=1,
    metrics=metrics,
    description="Logistic regression churn model",
)
model.save("data/model.json")

registered = mr.get_model("churnmodel7ed7ab", version=1)
print("registered:", registered.name, registered.version, registered.training_metrics)

with open("submission/answers.json", "w") as f:
    json.dump(
        {"model_name": "churnmodel7ed7ab", "version": 1, "metrics": metrics}, f, indent=2
    )
print("answers.json written")
