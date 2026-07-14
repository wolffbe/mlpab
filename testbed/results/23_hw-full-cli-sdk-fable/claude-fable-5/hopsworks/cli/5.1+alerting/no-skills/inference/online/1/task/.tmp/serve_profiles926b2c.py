import json

import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()
key_path = dataset_api.download("Resources/lookup926b2c.txt/lookup_keys.txt", overwrite=True)
with open(key_path) as f:
    keys = [line.strip() for line in f if line.strip()]

fs = project.get_feature_store()
fv = fs.get_feature_view("profiles926b2c_view", 1)
fv.init_serving()
rows = fv.get_feature_vectors(
    entry=[{"account_id": k} for k in keys],
    return_type="pandas",
)
print("columns:", list(rows.columns))

vectors = {}
if "account_id" in rows.columns:
    for _, row in rows.iterrows():
        vectors[str(row["account_id"])] = [
            float(row["f1"]), float(row["f2"]), float(row["f3"]), float(row["f4"])
        ]
else:
    for k, (_, row) in zip(keys, rows.iterrows()):
        vectors[k] = [
            float(row["f1"]), float(row["f2"]), float(row["f3"]), float(row["f4"])
        ]

missing = [k for k in keys if k not in vectors]
with open("answers.json", "w") as f:
    json.dump({"vectors": vectors}, f)
dataset_api.upload("answers.json", "Resources", overwrite=True)
print("SERVE_OK count=%d missing=%s" % (len(vectors), missing))
