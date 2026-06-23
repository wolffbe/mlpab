import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()

fg2 = fs.get_feature_group("customersa8deb9", version=2)
print("online_enabled:", fg2.online_enabled)
print("cols:", [f.name for f in fg2.features])

off = fg2.read()
print("OFFLINE rows:", len(off), "cols:", list(off.columns))

# online lookup test
import pandas as pd
sample = off["row_id"].head(3).tolist()
fv = None
for rid in sample:
    try:
        row = fg2.read(online=True) if False else None
    except Exception as e:
        print("err", e)
# online single-vector via feature view-less: use fg.find or online read
try:
    on = fg2.read(online=True)
    print("ONLINE rows:", len(on))
except Exception as e:
    print("online read err:", repr(e))
