import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs

aiplatform.init(project=os.environ["GCP_PROJECT"], location=os.environ["GCP_LOCATION"], api_transport="rest")
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

stores = fs.FeatureOnlineStore.list()
print(f"total online stores: {len(stores)}")
for s in stores:
    rn = s.name
    mine = PREFIX in rn
    print(("MINE " if mine else "     "), rn)
