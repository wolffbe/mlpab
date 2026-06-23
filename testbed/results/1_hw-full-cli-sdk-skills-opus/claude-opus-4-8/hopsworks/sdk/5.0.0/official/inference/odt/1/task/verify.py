import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("scored26cb88", version=1)

print("== schema ==")
print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("features:", [(f.name, f.type, f.primary) for f in fg.features])

print("== OFFLINE read ==")
off = fg.read()
print("cols:", list(off.columns), "rows:", off.shape[0], "unique req:", off["request_id"].is_unique)

print("== ONLINE read ==")
on = fg.read(online=True)
print("cols:", list(on.columns), "rows:", on.shape[0])
print("online sample:", on.head(3).to_dict("records"))

# cross-check a few values against the formula using raw source rows
import pandas as pd
req = pd.read_csv("data/requests.csv")
prof = pd.read_csv("data/profiles.csv").set_index("account_id")
off_idx = off.set_index("request_id")
import math
bad = 0
for rid in ["Q00000", "Q00001", "Q00002", "Q00100", "Q00399"]:
    rrow = req[req.request_id == rid].iloc[0]
    p = prof.loc[rrow.account_id]
    d = round(math.sqrt((rrow.request_lat - p.home_lat) ** 2 + (rrow.request_lon - p.home_lon) ** 2), 6)
    s = round(p.base_score - 0.1 * d, 6)
    got = off_idx.loc[rid]
    okd = abs(got.distance_deg - d) < 1e-6
    oks = abs(got.score - s) < 1e-6
    if not (okd and oks):
        bad += 1
    print(rid, "expected", d, s, "got", round(got.distance_deg, 6), round(got.score, 6), "OK" if okd and oks else "MISMATCH")
print("MISMATCHES:", bad)
