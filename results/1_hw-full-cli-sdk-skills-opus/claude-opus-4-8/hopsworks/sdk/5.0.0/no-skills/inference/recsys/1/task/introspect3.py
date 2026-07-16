import hopsworks, json, inspect
proj = hopsworks.login()
ja = proj.get_jobs_api()
for t in ["PYTHON", "PYSPARK", "SPARK"]:
    try:
        print(f"=== {t} ===")
        print(json.dumps(ja.get_configuration(t), indent=2))
    except Exception as e:
        print(t, "ERR", e)
ds = proj.get_dataset_api()
print("upload sig:", inspect.signature(ds.upload))
print("env api:", [m for m in dir(proj.get_environment_api()) if not m.startswith('_')])
