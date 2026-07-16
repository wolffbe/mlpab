import os
from google.cloud.aiplatform_v1.services.featurestore_online_serving_service import (
    FeaturestoreOnlineServingServiceClient,
)
from google.cloud.aiplatform_v1 import types as t
from google.api_core.client_options import ClientOptions

LOCATION = os.environ["GCP_LOCATION"]
PROJECT = os.environ["GCP_PROJECT"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

co = ClientOptions(api_endpoint=f"{LOCATION}-aiplatform.googleapis.com")
c = FeaturestoreOnlineServingServiceClient(client_options=co, transport="rest")
et_name = (
    f"projects/{PROJECT}/locations/{LOCATION}/featurestores/"
    f"{PREFIX}_scaled7b36f6/entityTypes/scaled7b36f6"
)
sel = t.FeatureSelector(id_matcher=t.IdMatcher(ids=["split", "f1", "f2", "f3", "f4"]))
for eid in ["R00000", "R00400"]:
    req = t.ReadFeatureValuesRequest(
        entity_type=et_name, entity_id=eid, feature_selector=sel
    )
    resp = c.read_feature_values(request=req)
    hdr = [d.id for d in resp.header.feature_descriptors]
    vals = []
    for fv in resp.entity_view.data:
        v = fv.value
        which = v._pb.WhichOneof("value")
        vals.append(getattr(v, which) if which else None)
    print(eid, dict(zip(hdr, vals)))
