import os
from google.cloud import aiplatform_v1 as a
proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
prefix = os.environ['MLPAB_GCP_PREFIX']
ep = f"{loc}-aiplatform.googleapis.com"; copts = {"api_endpoint": ep}
parent = f"projects/{proj}/locations/{loc}"
reg = a.FeatureRegistryServiceClient(transport="rest", client_options=copts)

def mk_group(fg_id, table, entity_cols, feat_ids):
    try:
        op = reg.create_feature_group(
            parent=parent, feature_group_id=fg_id,
            feature_group=a.FeatureGroup(
                big_query=a.FeatureGroup.BigQuery(
                    big_query_source=a.BigQuerySource(input_uri=f"bq://{proj}.{ds}.{table}"),
                    entity_id_columns=entity_cols)))
        print("FeatureGroup:", op.result(timeout=300).name)
    except Exception as e:
        print("FeatureGroup", fg_id, ":", type(e).__name__, str(e)[:160])
    fg_path = f"{parent}/featureGroups/{fg_id}"
    fac = a.FeatureRegistryServiceClient(transport="rest", client_options=copts)
    for fid in feat_ids:
        try:
            op = fac.create_feature(parent=fg_path, feature_id=fid, feature=a.Feature())
            print("  Feature:", op.result(timeout=300).name.split('/')[-1])
        except Exception as e:
            print("  Feature", fid, ":", type(e).__name__, str(e)[:120])

# Offline feature-store registry entries (no online-node quota needed)
mk_group("airqf3f1d8", "airqf3f1d8", ["date"],
         ["pm25_lag1","temperature","humidity","wind_speed","pressure","precipitation",
          "month","doy","pm25_lag1_roll3","pm25_lag1_roll7","temp_roll3","precip_roll3","pm25"])
mk_group("airqpredf3f1d8", "airqpredf3f1d8", ["date"], ["pm25_pred"])
