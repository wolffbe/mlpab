from google.cloud import bigquery, aiplatform_v1 as v1
import os
proj=os.environ['GCP_PROJECT']; ds=os.environ['GCP_BQ_DATASET']; loc=os.environ['GCP_LOCATION']; pref=os.environ['MLPAB_GCP_PREFIX']
bq=bigquery.Client(project=proj)
print("== BigQuery dataset tables ==")
for t in bq.list_tables(f"{proj}.{ds}"):
    tb=bq.get_table(f"{proj}.{ds}.{t.table_id}")
    print(f"  {t.table_id:16s} type={t.table_type:12s} rows={tb.num_rows}")
print("== BQML models ==")
for m in bq.list_models(f"{proj}.{ds}"):
    print("  ", m.model_id, m.model_type)
print("== ccpred76ccb2 schema ==")
tb=bq.get_table(f"{proj}.{ds}.ccpred76ccb2")
print("  ", [(f.name,f.field_type) for f in tb.schema])
print("== Vertex model ==")
ep=f"{loc}-aiplatform.googleapis.com"
mc=v1.ModelServiceClient(client_options={"api_endpoint":ep}, transport="rest")
for m in mc.list_models(parent=f"projects/{proj}/locations/{loc}"):
    if m.display_name.startswith(pref):
        print("  display_name:", m.display_name)
        print("  labels:", dict(m.labels))
        print("  desc:", m.description[:100])
print("== Featurestore online ==")
fc=v1.FeaturestoreServiceClient(client_options={"api_endpoint":ep}, transport="rest")
fsr=fc.get_featurestore(name=fc.featurestore_path(proj,loc,f"{pref}_ccfs"))
print("  ", fsr.name, "state", fsr.state, "nodes", fsr.online_serving_config.fixed_node_count)
print("ALL VERIFIED")
