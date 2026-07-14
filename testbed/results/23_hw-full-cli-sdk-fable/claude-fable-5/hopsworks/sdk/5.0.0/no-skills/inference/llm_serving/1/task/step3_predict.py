import json
import os

os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import hopsworks

project = hopsworks.login()
ms = project.get_model_serving()

deployment = ms.get_deployment("scorer2840e8")
print("deployment:", deployment.name, "state:", deployment.get_state().status)

with open("data/payloads.json") as f:
    payloads = json.load(f)

responses = []
for text in payloads:
    result = deployment.predict(inputs=[text])
    print("raw:", result)
    responses.append(result["predictions"][0])

os.makedirs("submission", exist_ok=True)
answers = {"endpoint_name": "scorer2840e8", "responses": responses}
with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)
print(json.dumps(answers, indent=2))
