import warnings, urllib3
warnings.filterwarnings("ignore")
urllib3.disable_warnings()
import hopsworks
proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_feature_group("events5b591e", version=1)
try:
    df = fg.read(online=True)
    print("online rows:", df.shape[0])
except Exception as e:
    print("online read err:", repr(e)[:300])
