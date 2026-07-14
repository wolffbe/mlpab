import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()

for p in [
    "Resources/trainjob646af0",
    "Resources/trainjob646af0/train.csv",
    "Resources/trainjob646af0/score.csv",
    "Resources/trainjob646af0/train_model.py",
    "Resources/trainjob646af0/job_wrapper.py",
]:
    print(p, "->", ds.exists(p))

res = ds.list("Resources/trainjob646af0")
try:
    items = res["items"]
    for it in items:
        print("item:", it["attributes"]["path"], it["attributes"]["size"])
except Exception as e:
    print("raw list result:", res)
