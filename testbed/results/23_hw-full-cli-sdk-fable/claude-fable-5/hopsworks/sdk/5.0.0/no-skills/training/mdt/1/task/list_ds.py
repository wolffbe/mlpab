import os
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)
import hopsworks

project = hopsworks.login()
ds = project.get_dataset_api()
for path in ["Resources", "Resources/scaledaff2b3"]:
    print("==", path, "exists:", ds.exists(path))
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
