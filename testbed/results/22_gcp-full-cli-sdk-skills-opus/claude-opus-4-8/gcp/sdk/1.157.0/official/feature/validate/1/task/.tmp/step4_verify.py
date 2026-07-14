import os, time
import vertexai
from vertexai.resources import preview as fs

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
vertexai.init(project=PROJECT, location=LOCATION, api_transport="rest")

STORE = "mlpaba45c1a_txn85a07a_store"
FV = "eventsd3c188"
store = fs.FeatureOnlineStore(STORE)
fv = None
for e in store.list_feature_views():
    if e.name == FV:
        fv = e
if fv is None:
    raise SystemExit("feature view missing")

# Wait for a completed sync
done = False
for attempt in range(30):
    syncs = fv.list_syncs()
    states = []
    for s in syncs:
        end = s.gca_resource.run_time.end_time
        states.append(("done" if end and end.seconds else "running"))
    print(f"attempt {attempt}: syncs={states}")
    if any(st == "done" for st in states):
        done = True
        break
    time.sleep(30)

# Try an online read of a valid record key
try:
    resp = fv.read(key=["R00001"])
    print("ONLINE READ R00001:", resp.to_dict())
except Exception as e:
    print("online read error (may still be warming up):", type(e).__name__, str(e)[:200])
print("sync_completed:", done)
