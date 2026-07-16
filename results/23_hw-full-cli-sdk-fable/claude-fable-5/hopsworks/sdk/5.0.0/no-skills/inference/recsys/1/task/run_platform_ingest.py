import csv
import os
import time

# The sandbox only allows outbound traffic via the localhost proxy; the
# platform host sits in 10.0.0.0/8 which NO_PROXY would bypass.
for var in ("NO_PROXY", "no_proxy"):
    os.environ.pop(var, None)

import hopsworks

# ---- compute top-5 recommendations per user ----
def load_emb(path, key):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            d[row[key]] = [float(row[f"e{i}"]) for i in range(1, 9)]
    return d

users = load_emb("data/user_embeddings.csv", "user_id")
items = load_emb("data/item_embeddings.csv", "item_id")

seen = {}
with open("data/interactions.csv") as f:
    for row in csv.DictReader(f):
        seen.setdefault(row["user_id"], set()).add(row["item_id"])

lines = ["rec_id,user_id,rank,item_id"]
for user_id in sorted(users):
    uvec = users[user_id]
    scored = []
    for item_id, ivec in items.items():
        if item_id in seen.get(user_id, set()):
            continue
        score = sum(a * b for a, b in zip(uvec, ivec))
        scored.append((-score, item_id))
    scored.sort()  # highest score first, exact ties by ascending item_id
    for rank, (_, item_id) in enumerate(scored[:5], start=1):
        lines.append(f"{user_id}#{rank},{user_id},{rank},{item_id}")

csv_blob = "\n".join(lines)
print("rows (excl header):", len(lines) - 1)

# ---- job script that runs ON the cluster and writes the feature group ----
job_script = '''import io
import hopsworks
import pandas as pd

CSV = """__CSV__"""

project = hopsworks.login()
fs = project.get_feature_store()
df = pd.read_csv(io.StringIO(CSV))
df["rank"] = df["rank"].astype("int64")
fg = fs.get_or_create_feature_group(
    name="recs48963e",
    version=1,
    primary_key=["rec_id"],
    online_enabled=True,
    description="Top-5 recommended items per user (two-tower dot product)",
)
fg.insert(df, wait=True)
print("insert complete:", len(df), "rows")
'''.replace("__CSV__", csv_blob)

with open("ingest_recs_job.py", "w") as f:
    f.write(job_script)

# ---- upload + run as a platform job ----
project = hopsworks.login()
dataset_api = project.get_dataset_api()
path = dataset_api.upload("ingest_recs_job.py", "Resources", overwrite=True)
print("uploaded:", path)

jobs_api = project.get_job_api()
cfg = jobs_api.get_configuration("PYTHON")
cfg["appPath"] = "/Projects/" + project.name + "/Resources/ingest_recs_job.py"
job = jobs_api.create_job("recs48963e_ingest", cfg)
execution = job.run(await_termination=True)
print("final state:", execution.state, execution.final_status)
out, err = execution.download_logs()
for p in (out, err):
    try:
        with open(p) as fh:
            print("----", p, "----")
            print(fh.read()[-4000:])
    except Exception as e:
        print("log read failed:", e)
time.sleep(1)
