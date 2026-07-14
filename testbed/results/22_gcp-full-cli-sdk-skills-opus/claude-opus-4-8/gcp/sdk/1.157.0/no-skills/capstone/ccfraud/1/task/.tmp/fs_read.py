from google.cloud import aiplatform_v1 as v1
import os
proj=os.environ['GCP_PROJECT']; loc=os.environ['GCP_LOCATION']; pref=os.environ['MLPAB_GCP_PREFIX']
ep=f"{loc}-aiplatform.googleapis.com"
oc=v1.FeaturestoreOnlineServingServiceClient(client_options={"api_endpoint":ep}, transport="rest")
et_path=f"projects/{proj}/locations/{loc}/featurestores/{pref}_ccfs/entityTypes/transaction"
for tid in ["T000032740","T000038691","T000037301"]:
    req=v1.ReadFeatureValuesRequest(entity_type=et_path, entity_id=tid,
        feature_selector=v1.FeatureSelector(id_matcher=v1.IdMatcher(ids=["fraud_probability"])))
    resp=oc.read_feature_values(req)
    val=resp.entity_view.data[0].value.double_value
    print(tid, "-> fraud_probability =", val)
print("ONLINE LOOKUP OK")
