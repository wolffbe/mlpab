import hw_env  # noqa: F401
import hopsworks

proj = hopsworks.login()
ds = proj.get_dataset_api()
for path in ("Resources", "Resources/drift_task"):
    print("==", path)
    try:
        res = ds.list(path)
        items = res["items"] if isinstance(res, dict) else res
        for it in items:
            try:
                attr = it["attributes"]
                print("  ", attr["path"], attr.get("size"))
            except Exception:
                print("  ", it)
    except Exception as e:
        print("  list failed:", e)
print("exists csv:", ds.exists("Resources/drift_task/features.csv"))
print("exists script:", ds.exists("Resources/drift_task/drift_job.py"))
