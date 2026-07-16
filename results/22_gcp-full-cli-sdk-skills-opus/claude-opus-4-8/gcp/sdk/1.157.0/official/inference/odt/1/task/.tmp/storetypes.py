import os
import google.cloud.aiplatform as aiplatform
from vertexai.resources.preview import feature_store as fs

aiplatform.init(project=os.environ["GCP_PROJECT"], location=os.environ["GCP_LOCATION"], api_transport="rest")
for s in fs.FeatureOnlineStore.list():
    r = s.gca_resource
    kind = "bigtable" if r._pb.HasField("bigtable") else ("optimized" if r._pb.HasField("optimized") else "?")
    print(s.name, "->", kind)
