import json
import os

import hopsworks

BASE = "Resources/ftjob6f5e78"

project = hopsworks.login()
ds = project.get_dataset_api()
print("output listing:")
for p in ["/output/finetuned_model.npz", "/output/metrics.json"]:
    print(" ", BASE + p, "exists:", ds.exists(BASE + p))

resp = ds.read_content("/" + BASE + "/output/metrics.json")
metrics_raw = resp.content if hasattr(resp, "content") else resp
if isinstance(metrics_raw, bytes):
    metrics_raw = metrics_raw.decode()
print("metrics raw:", metrics_raw)
metrics = json.loads(metrics_raw)

os.makedirs(".tmp/model_dir", exist_ok=True)
ds.download(BASE + "/output/finetuned_model.npz", ".tmp/model_dir", overwrite=True)
print("downloaded:", os.listdir(".tmp/model_dir"))

mr = project.get_model_registry()
model = mr.python.create_model(
    name="ftmodel6f5e78",
    version=1,
    metrics=metrics,
    description="Fine-tuned bigram LM (rank-4 adapter) from job ftjob6f5e78",
)
model.save(".tmp/model_dir")
print("registered:", model.name, "v", model.version, "metrics:", model.training_metrics)

os.makedirs("submission", exist_ok=True)
answers = {
    "job_name": "ftjob6f5e78",
    "model_name": "ftmodel6f5e78",
    "eval_loss": metrics["eval_loss"],
    "base_eval_loss": metrics["base_eval_loss"],
}
with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)
print("answers:", answers)
