import hopsworks
import json
import os

proj = hopsworks.login()
fs = proj.get_feature_store()

fg = fs.get_feature_group("profiles27ba29", version=1)

# Build a feature view over the feature group selecting features in order
query = fg.select(["f1", "f2", "f3", "f4"])
fv = fs.get_or_create_feature_view(
    name="profiles27ba29_fv",
    version=1,
    query=query,
)
print("fv ready", fv.name, fv.version)

fv.init_serving(training_dataset_version=None)

keys = [k.strip() for k in open("data/lookup_keys.txt") if k.strip()]
print("num keys", len(keys))

vectors = {}
for k in keys:
    vec = fv.get_feature_vector(entry={"account_id": k})
    # vec is a list in feature view schema order [f1,f2,f3,f4]
    vectors[k] = [float(x) for x in vec]

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"vectors": vectors}, f, indent=2)

print("wrote", len(vectors), "vectors")
print(json.dumps({"sample": dict(list(vectors.items())[:2])}, indent=2))
