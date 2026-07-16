import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import hopsworks

project = hopsworks.login()
ms = project.get_model_serving()

dataset_api = project.get_dataset_api()
uploaded = dataset_api.upload("predictor_scorer.py", "Resources", overwrite=True)
print("re-uploaded predictor:", uploaded)

deployment = ms.get_deployment("scorer2840e8")
deployment.stop(await_stopped=600)
print("stopped")
deployment.start(await_running=900)
print("state:", deployment.get_state().status)

# quick smoke test
print("smoke:", deployment.predict(inputs=["hello world abc def"]))
