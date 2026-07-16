import os, time
from google.cloud import aiplatform_v1 as v1
from google.api_core.exceptions import ResourceExhausted, AlreadyExists

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
PARENT = f"projects/{PROJECT}/locations/{LOCATION}"
STORE_ID = f"{PREFIX}_features347afc_store"

admin = v1.FeatureOnlineStoreAdminServiceClient(
    transport="rest", client_options={"api_endpoint": ENDPOINT})
store_res = f"{PARENT}/featureOnlineStores/{STORE_ID}"

def exists():
    return STORE_ID in {s.name.split("/")[-1]
                        for s in admin.list_feature_online_stores(parent=PARENT)}

if exists():
    print("already exists:", store_res); raise SystemExit(0)

for attempt in range(20):
    try:
        fos = v1.FeatureOnlineStore(
            bigtable=v1.FeatureOnlineStore.Bigtable(
                auto_scaling=v1.FeatureOnlineStore.Bigtable.AutoScaling(
                    min_node_count=1, max_node_count=1, cpu_utilization_target=50)))
        op = admin.create_feature_online_store(
            parent=PARENT, feature_online_store_id=STORE_ID, feature_online_store=fos)
        op.result(timeout=1200)
        print("CREATED:", store_res); break
    except AlreadyExists:
        print("AlreadyExists ->", store_res); break
    except ResourceExhausted as e:
        if exists():
            print("appeared despite quota:", store_res); break
        print(f"attempt {attempt}: quota exhausted, waiting..."); time.sleep(60)
else:
    print("GAVE UP: quota never freed")
