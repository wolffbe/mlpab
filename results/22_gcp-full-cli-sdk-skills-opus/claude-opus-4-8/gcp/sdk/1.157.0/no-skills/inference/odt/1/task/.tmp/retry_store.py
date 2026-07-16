import os, time
import google.cloud.aiplatform as aiplatform
from google.api_core.exceptions import ResourceExhausted
from vertexai.resources.preview import FeatureOnlineStore

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
aiplatform.init(project=proj, location=loc, api_transport="rest")
name = os.environ['MLPAB_GCP_PREFIX'] + '_online_store'

store = None
for attempt in range(16):
    try:
        store = FeatureOnlineStore.create_optimized_store(name)
        print("CREATED", store.name, "on attempt", attempt)
        break
    except ResourceExhausted as e:
        print(f"attempt {attempt}: quota exhausted, waiting", flush=True)
        # a partial store may exist now; check
        try:
            store = FeatureOnlineStore(name)
            print("FOUND after attempt", attempt, store.name)
            break
        except Exception:
            pass
        time.sleep(40)
    except Exception as e:
        print("attempt", attempt, "other error:", str(e)[:150], flush=True)
        time.sleep(40)

if store is None:
    print("RESULT: FAILED_QUOTA")
else:
    print("RESULT: OK", store.name)
