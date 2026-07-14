import os, json
import google.cloud.aiplatform as aiplatform
from google.cloud.aiplatform import Featurestore

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

FS_ID = f"{PREFIX}_profilesaf22bf_fs"
ET_ID = "profilesaf22bf"
FEATURES = ["f1", "f2", "f3", "f4"]

fs = Featurestore(FS_ID)
et = fs.get_entity_type(ET_ID)

with open("data/lookup_keys.txt") as f:
    keys = [ln.strip() for ln in f if ln.strip()]

# Online / low-latency read through the featurestore online serving path
df = et.read(entity_ids=keys, feature_ids=FEATURES)
df = df.set_index("entity_id")

vectors = {}
for k in keys:
    row = df.loc[k]
    vectors[k] = [float(row[c]) for c in FEATURES]

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"vectors": vectors}, f, indent=2)

print("wrote", len(vectors), "vectors")
print(json.dumps({"vectors": {k: vectors[k] for k in keys[:3]}}, indent=2))
