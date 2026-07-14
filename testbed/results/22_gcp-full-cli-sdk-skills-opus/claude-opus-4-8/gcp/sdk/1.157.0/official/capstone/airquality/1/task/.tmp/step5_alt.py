import os
from google.cloud import aiplatform_v1 as a
proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; prefix = os.environ['MLPAB_GCP_PREFIX']
ep = f"{loc}-aiplatform.googleapis.com"; copts = {"api_endpoint": ep}
parent = f"projects/{proj}/locations/{loc}"

# A) Bigtable-type new online store
admin = a.FeatureOnlineStoreAdminServiceClient(transport="rest", client_options=copts)
try:
    op = admin.create_feature_online_store(
        parent=parent, feature_online_store_id=f"{prefix}_airq_bt",
        feature_online_store=a.FeatureOnlineStore(
            bigtable=a.FeatureOnlineStore.Bigtable(
                auto_scaling=a.FeatureOnlineStore.Bigtable.AutoScaling(
                    min_node_count=1, max_node_count=1, cpu_utilization_target=50))))
    print("bigtable store:", op.result(timeout=900).name)
except Exception as e:
    print("bigtable store:", type(e).__name__, str(e)[:160])

# B) legacy featurestore with autoscaling online config
svc = a.FeaturestoreServiceClient(transport="rest", client_options=copts)
try:
    op = svc.create_featurestore(
        parent=parent, featurestore_id=f"{prefix}_airqpred_fs2",
        featurestore=a.Featurestore(
            online_serving_config=a.Featurestore.OnlineServingConfig(
                scaling=a.Featurestore.OnlineServingConfig.Scaling(
                    min_node_count=1, max_node_count=1))))
    print("legacy fs autoscale:", op.result(timeout=900).name)
except Exception as e:
    print("legacy fs autoscale:", type(e).__name__, str(e)[:160])
