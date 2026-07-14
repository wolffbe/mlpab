import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import hopsworks

project = hopsworks.login()
ms = project.get_model_serving()

dataset_api = project.get_dataset_api()
uploaded = dataset_api.upload("predictor_scorer.py", "Resources", overwrite=True)
script_file = "/Projects/{}/{}".format(project.name, uploaded)
print("re-uploaded predictor:", script_file)

deployment = ms.get_deployment("scorer2840e8")
print("current script_file:", deployment.script_file)
print("current artifact_version:", deployment.artifact_version)

deployment.stop(await_stopped=600)
deployment.script_file = script_file
deployment.save()
print("saved; new artifact_version:", deployment.artifact_version)

deployment.start(await_running=900)
print("state:", deployment.get_state().status)

print("smoke:", deployment.predict(inputs=["hello world abc def"]))

try:
    logs = deployment.get_logs(component="predictor", tail=60)
    if logs:
        for entry in logs if isinstance(logs, list) else [logs]:
            print(getattr(entry, "content", entry))
except Exception as e:
    print("log fetch failed:", e)
