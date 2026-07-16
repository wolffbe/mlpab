import warnings, urllib3
warnings.filterwarnings("ignore")
urllib3.disable_warnings()
import hopsworks
proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_feature_group("events5b591e", version=1)
print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("event_time:", fg.event_time)
# offline count
try:
    n = fg.read().shape[0]
    print("offline rows:", n)
except Exception as e:
    print("offline read err:", e)
# online lookup test
try:
    row = fg.read(online=True) if False else None
    v = fg.select_all().show(1) if False else None
except Exception:
    pass
