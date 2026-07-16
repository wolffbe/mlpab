import os
import google.cloud.aiplatform_v1 as v1

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]
API = f"{LOCATION}-aiplatform.googleapis.com"
parent = f"projects/{PROJECT}/locations/{LOCATION}"
BQ_URI = f"bq://{PROJECT}.{DATASET}.features347afc"

reg = v1.FeatureRegistryServiceClient(client_options={"api_endpoint": API}, transport="rest")
admin = v1.FeatureOnlineStoreAdminServiceClient(client_options={"api_endpoint": API}, transport="rest")

# full FG error
fg = v1.FeatureGroup(
    big_query=v1.FeatureGroup.BigQuery(
        big_query_source=v1.BigQuerySource(input_uri=BQ_URI),
        entity_id_columns=["row_id"],
    ),
    description="test",
)
try:
    reg.create_feature_group(request=v1.CreateFeatureGroupRequest(parent=parent, feature_group=fg, feature_group_id="features347afc")).result(timeout=600)
    print("FG OK")
except Exception as e:
    print("FG ERR FULL:", repr(str(e)))

print("--- existing feature groups ---")
for fgx in reg.list_feature_groups(parent=parent):
    print(" ", fgx.name.split("/")[-1])

print("--- existing online stores ---")
for s in admin.list_feature_online_stores(parent=parent):
    print(" ", s.name.split("/")[-1], "state=", s.state.name, "type=", "bigtable" if s.bigtable else "optimized")
