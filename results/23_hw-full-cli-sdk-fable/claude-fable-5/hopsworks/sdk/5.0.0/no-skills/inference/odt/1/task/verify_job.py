import hopsworks
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("scored72af4e", 1)
print("features:", [f.name for f in fg.features], flush=True)
print("online_enabled:", fg.online_enabled, "pk:", fg.primary_key, flush=True)

on = fg.read(online=True)
print("online rows:", on.shape, list(on.columns), flush=True)

req = pd.read_csv("/hopsfs/Resources/requests.csv")
prof = pd.read_csv("/hopsfs/Resources/profiles.csv")
exp = req.merge(prof, on="account_id", how="left")
exp["distance_deg"] = (
    ((exp.request_lat - exp.home_lat) ** 2 + (exp.request_lon - exp.home_lon) ** 2)
    ** 0.5
).round(6)
exp["score"] = (exp.base_score - 0.1 * exp.distance_deg).round(6)
m = on.merge(exp[["request_id", "distance_deg", "score"]], on="request_id", suffixes=("", "_exp"))
print("rows compared:", len(m), flush=True)
print("max dist err:", (m.distance_deg - m.distance_deg_exp).abs().max(), flush=True)
print("max score err:", (m.score - m.score_exp).abs().max(), flush=True)
print("nulls:", on.isna().sum().to_dict(), flush=True)
print(on.head().to_string(), flush=True)

try:
    off = fg.read()
    print("offline rows:", off.shape, list(off.columns), flush=True)
    mo = off.merge(exp[["request_id", "distance_deg", "score"]], on="request_id", suffixes=("", "_exp"))
    print("offline compared:", len(mo),
          "max dist err:", (mo.distance_deg - mo.distance_deg_exp).abs().max(),
          "max score err:", (mo.score - mo.score_exp).abs().max(), flush=True)
except Exception as e:
    print("offline read failed:", e, flush=True)
