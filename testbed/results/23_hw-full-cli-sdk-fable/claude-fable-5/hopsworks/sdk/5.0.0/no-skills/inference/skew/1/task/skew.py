import os, json, math
import urllib3; urllib3.disable_warnings()
import hopsworks
import pandas as pd

proj = hopsworks.login()
fs = proj.get_feature_store()
print("project:", proj.name)

train_df = pd.read_csv("data/training_sample.csv")
serve_df = pd.read_csv("data/serving_log.csv")

fg_tr = fs.get_or_create_feature_group(name="skew_train", version=1,
        primary_key=["entity_id"], description="training-path features")
fg_tr.insert(train_df, write_options={"wait_for_job": True})
fg_sv = fs.get_or_create_feature_group(name="skew_serve", version=1,
        primary_key=["entity_id"], description="serving-path features")
fg_sv.insert(serve_df, write_options={"wait_for_job": True})

# platform-side join on entity_id
q = fg_tr.select_all().join(fg_sv.select_all(), on=["entity_id"], prefix="srv_")
joined = q.read()
print("joined rows:", len(joined), "cols:", list(joined.columns))

feats = ["f1", "f2", "f3", "f4", "f5"]
report = {}
for f in feats:
    d = (joined[f] - joined[f"srv_{f}"]).abs()
    report[f] = {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())}
    print(f, report[f])

skewed = max(feats, key=lambda f: report[f]["mean_abs_diff"])
print("skewed feature:", skewed)

# check log1p hypothesis on the skewed feature
chk = (joined[skewed] - joined[f"srv_{skewed}"].apply(math.log1p)).abs().mean()
print("mean |train - log1p(serve)| for", skewed, "=", chk)

answer = {"feature": skewed,
          "cause": ("The serving path skips the log1p transformation applied in the training "
                    "pipeline: training values equal log1p(served values) for the same entities, "
                    "so the online service serves the raw feature instead of the log-transformed one.")}
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as fh:
    json.dump(answer, fh)
print("local answer written:", answer["feature"])

# upload deliverable to the platform's dataset storage
ds = proj.get_dataset_api()
try:
    ds.mkdir("Resources/submission")
except Exception as e:
    print("mkdir note:", e)
path = ds.upload("submission/answers.json", "Resources/submission", overwrite=True)
print("uploaded to:", path)
local_copy = ds.download(path, ".tmp", overwrite=True)
print("readback:", open(local_copy).read())
