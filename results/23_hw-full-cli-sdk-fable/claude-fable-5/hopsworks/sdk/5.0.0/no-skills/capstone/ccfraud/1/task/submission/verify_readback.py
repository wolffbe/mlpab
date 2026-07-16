"""Verify ccpred4b8521 / cctxn4b8521 read back via platform paths."""

import time
import traceback

import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()

pfg = fs.get_feature_group("ccpred4b8521", 1)
print("online_enabled:", pfg.online_enabled, flush=True)

# online path first
try:
    on = pfg.select_all().read(online=True)
    print("ONLINE rows:", on.shape, flush=True)
    print(on.head(3), flush=True)
except Exception:
    print("ONLINE READ FAILED", flush=True)
    traceback.print_exc()

# offline read of the training feature group
try:
    fg = fs.get_feature_group("cctxn4b8521", 1)
    tx = fg.select(["transaction_id", "is_fraud"]).read()
    print("cctxn OFFLINE rows:", tx.shape, flush=True)
except Exception:
    print("cctxn OFFLINE READ FAILED", flush=True)
    traceback.print_exc()

# offline read of predictions, with retries
off = None
for attempt in range(5):
    try:
        off = pfg.select_all().read()
        break
    except Exception as e:
        print(f"ccpred offline attempt {attempt} failed: {e}", flush=True)
        time.sleep(20)
if off is not None:
    print("ccpred OFFLINE rows:", off.shape, flush=True)
    print("prob min/max:", float(off["fraud_probability"].min()),
          float(off["fraud_probability"].max()), flush=True)
    print(off.head(3), flush=True)
else:
    print("ccpred OFFLINE READ FAILED AFTER RETRIES", flush=True)

print("VERIFY_DONE", flush=True)
