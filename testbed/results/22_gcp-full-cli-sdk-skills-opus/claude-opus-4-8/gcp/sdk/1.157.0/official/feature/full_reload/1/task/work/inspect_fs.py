import os
for v in ("GRPC_PROXY", "grpc_proxy"):
    os.environ.pop(v, None)
import vertexai
from vertexai.resources.preview import feature_store as fs

vertexai.init(project=os.environ['GCP_PROJECT'],
              location=os.environ['GCP_LOCATION'], api_transport="rest")

print("=== FeatureGroups ===")
for fg in fs.FeatureGroup.list():
    print(fg.name, "|", getattr(fg.gca_resource.big_query, "big_query_source", None))
    try:
        feats = [f.name for f in fg.list_features()]
        print("   features:", feats)
    except Exception as e:
        print("   features err:", e)

print("=== FeatureOnlineStores ===")
for s in fs.FeatureOnlineStore.list():
    print(s.name, s.feature_online_store_type)
    for fv in s.list_feature_views():
        print("   view:", fv.name)
