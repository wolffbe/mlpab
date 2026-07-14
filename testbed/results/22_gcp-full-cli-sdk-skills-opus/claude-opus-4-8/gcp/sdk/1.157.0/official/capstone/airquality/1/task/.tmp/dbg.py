import os
from google.cloud import aiplatform_v1 as a
proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
ep = f"{loc}-aiplatform.googleapis.com"; copts = {"api_endpoint": ep}
parent = f"projects/{proj}/locations/{loc}"
reg = a.FeatureRegistryServiceClient(transport="rest", client_options=copts)

# full error for predictions feature group
try:
    op = reg.create_feature_group(
        parent=parent, feature_group_id="airqpredf3f1d8",
        feature_group=a.FeatureGroup(
            big_query=a.FeatureGroup.BigQuery(
                big_query_source=a.BigQuerySource(input_uri=f"bq://{proj}.{ds}.airqpredf3f1d8"),
                entity_id_columns=["date"])))
    print("created:", op.result(timeout=300).name)
except Exception as e:
    print("FULL ERR:", repr(e)[:600])

# full error for a feature under existing airqf3f1d8
fg_path = f"{parent}/featureGroups/airqf3f1d8"
try:
    op = reg.create_feature(parent=fg_path, feature_id="pm25", feature=a.Feature())
    print("feature created:", op.result(timeout=300).name)
except Exception as e:
    print("FEAT FULL ERR:", repr(e)[:600])

# list existing feature groups matching airq
print("== existing feature groups ==")
for fg in reg.list_feature_groups(parent=parent):
    if "airq" in fg.name:
        print(" ", fg.name.split('/')[-1])
