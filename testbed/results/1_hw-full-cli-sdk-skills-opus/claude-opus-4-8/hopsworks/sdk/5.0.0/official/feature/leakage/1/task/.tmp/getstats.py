import json
import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("leakage_probe", version=1)

s = None
try:
    s = fg.get_statistics()
    print("got statistics object:", type(s))
except Exception as e:
    print("get_statistics err:", repr(e)[:400])

if s is not None:
    content = getattr(s, "content", None)
    print("content type:", type(content))
    with open(".tmp/stats_content.json", "w") as f:
        json.dump(content, f, default=str, indent=2)
    print("wrote .tmp/stats_content.json")
    txt = json.dumps(content, default=str)
    print("len", len(txt))
    print(txt[:2000])
