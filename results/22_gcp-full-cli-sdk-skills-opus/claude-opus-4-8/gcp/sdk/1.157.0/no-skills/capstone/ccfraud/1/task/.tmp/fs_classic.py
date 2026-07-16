from google.cloud import aiplatform_v1 as v1
import os
proj=os.environ['GCP_PROJECT']; loc=os.environ['GCP_LOCATION']; pref=os.environ['MLPAB_GCP_PREFIX']
ep=f"{loc}-aiplatform.googleapis.com"
c=v1.FeaturestoreServiceClient(client_options={"api_endpoint":ep}, transport="rest")
parent=f"projects/{proj}/locations/{loc}"
print("existing featurestores:")
try:
    for fs in c.list_featurestores(parent=parent):
        print("  ", fs.name)
except Exception as e:
    print("list err", type(e).__name__, str(e)[:150])
# proto fields
print("FS fields:", [f.name for f in v1.Featurestore.pb(v1.Featurestore()).DESCRIPTOR.fields])
print("OnlineServingConfig:", [f.name for f in v1.Featurestore.OnlineServingConfig.pb(v1.Featurestore.OnlineServingConfig()).DESCRIPTOR.fields])
print("EntityType fields:", [f.name for f in v1.EntityType.pb(v1.EntityType()).DESCRIPTOR.fields])
print("Feature fields:", [f.name for f in v1.Feature.pb(v1.Feature()).DESCRIPTOR.fields])
