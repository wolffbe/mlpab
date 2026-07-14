from google.cloud import aiplatform_v1 as v1
import os
proj=os.environ['GCP_PROJECT']; loc=os.environ['GCP_LOCATION']; pref=os.environ['MLPAB_GCP_PREFIX']
ep=f"{loc}-aiplatform.googleapis.com"
c=v1.FeaturestoreServiceClient(client_options={"api_endpoint":ep}, transport="rest")
parent=f"projects/{proj}/locations/{loc}"
FS_ID=f"{pref}_ccfs"
fs=v1.Featurestore(
    online_serving_config=v1.Featurestore.OnlineServingConfig(fixed_node_count=1),
    labels={"mlpab_prefix":pref,"task":"fraud"})
try:
    op=c.create_featurestore(parent=parent, featurestore=fs, featurestore_id=FS_ID)
    print("creating featurestore (online nodes)...")
    r=op.result(timeout=1800)
    print("featurestore:", r.name, r.state)
except Exception as e:
    print("FS create note:", type(e).__name__, str(e)[:250])
    try:
        r=c.get_featurestore(name=c.featurestore_path(proj,loc,FS_ID))
        print("existing featurestore:", r.name, r.state)
    except Exception as e2:
        print("get err:", str(e2)[:200])
print("done")
