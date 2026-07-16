import os
from google.cloud import bigquery, aiplatform_v1 as a
proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']; ds = os.environ['GCP_BQ_DATASET']
prefix = os.environ['MLPAB_GCP_PREFIX']
bq = bigquery.Client(project=proj)
print("=== BigQuery deliverables in", ds, "===")
for t in ["airqf3f1d8","airqtdf3f1d8","airqpredf3f1d8","airqmodelf3f1d8"]:
    try:
        obj = bq.get_table(f"{proj}.{ds}.{t}")
        print(f"  {t}: {obj.table_type}  rows={obj.num_rows}  cols={[f.name for f in obj.schema]}")
    except Exception as e:
        try:
            m = bq.get_model(f"{proj}.{ds}.{t}")
            print(f"  {t}: MODEL {m.model_type}")
        except Exception as e2:
            print(f"  {t}: MISSING {e2}")

ep = f"{loc}-aiplatform.googleapis.com"
mc = a.ModelServiceClient(transport="rest", client_options={"api_endpoint": ep})
parent = f"projects/{proj}/locations/{loc}"
vid = f"{prefix}_airqmodelf3f1d8"
m = next(x for x in mc.list_models(parent=parent) if x.display_name == vid)
print("=== Vertex Model ===")
print("  display_name:", m.display_name, " versions:", m.version_id)
print("  evaluations:", len(list(mc.list_model_evaluations(parent=m.name))))
