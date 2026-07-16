import json

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("profiles926b2c", version=1)

fv = fs.get_or_create_feature_view(
    name="profiles926b2c_fv",
    version=1,
    query=fg.select(["f1", "f2", "f3", "f4"]),
)

fv.init_serving(
    init_sql_client=False,
    init_rest_client=True,
    config_rest_client={"verify_certs": False},
    default_client="rest",
)

keys = [line.strip() for line in open("data/lookup_keys.txt") if line.strip()]
vectors = {}
for k in keys:
    vec = fv.get_feature_vector({"account_id": k}, force_rest_client=True)
    vectors[k] = [float(x) for x in vec]
    print(k, vectors[k])

with open("submission/answers.json", "w") as f:
    json.dump({"vectors": vectors}, f, indent=2)
print("Wrote submission/answers.json with", len(vectors), "entries")
