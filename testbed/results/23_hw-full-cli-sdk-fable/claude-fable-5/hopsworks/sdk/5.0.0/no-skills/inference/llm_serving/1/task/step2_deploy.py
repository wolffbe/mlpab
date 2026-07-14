import os
import shutil

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import hopsworks

project = hopsworks.login()
mr = project.get_model_registry()
ms = project.get_model_serving()

# 1. Register the scorer as a python model (uploads scorer.py to the platform)
model_dir = ".tmp/scorer_model"
os.makedirs(model_dir, exist_ok=True)
shutil.copy("data/scorer.py", os.path.join(model_dir, "scorer.py"))

model = mr.python.create_model(
    name="scorer2840e8",
    description="Deterministic pure-python character-trigram log-likelihood scorer",
)
model.save(model_dir)
print("model registered:", model.name, "v", model.version)

# 2. Upload the predictor script to the project's Resources dataset
dataset_api = project.get_dataset_api()
uploaded = dataset_api.upload("predictor_scorer.py", "Resources", overwrite=True)
script_file = "/Projects/{}/{}".format(project.name, uploaded)
print("predictor script:", script_file)

# 3. Deploy as a real-time endpoint named scorer2840e8
deployment = model.deploy(name="scorer2840e8", script_file=script_file)
print("deployment created:", deployment.name)

deployment.start(await_running=900)
print("deployment state:", deployment.get_state().status)
