import os
import google.cloud.aiplatform as aiplatform
from google.cloud.aiplatform import Featurestore
from google.cloud import bigquery

PROJECT=os.environ["GCP_PROJECT"]; LOCATION=os.environ["GCP_LOCATION"]; DATASET=os.environ["GCP_BQ_DATASET"]
aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")

bq = bigquery.Client(project=PROJECT)
final = f"{PROJECT}.{DATASET}.accountsed4daa"
sample_ids = [r.row_id for r in bq.query(
    f"SELECT row_id FROM `{final}` ORDER BY row_id LIMIT 5", location=LOCATION).result()]
expected = {r.row_id: (r.status, round(r.balance,2), r.updated_at) for r in bq.query(
    f"SELECT * FROM `{final}` WHERE row_id IN UNNEST(@ids)",
    job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("ids","STRING",sample_ids)]),
    location=LOCATION).result()}

store = Featurestore("accountsed4daa_fs")
et = store.get_entity_type("accountsed4daa")
df = et.read(entity_ids=sample_ids, feature_ids=["status","balance","updated_at"])
print(df.to_string())
print()
for _, row in df.iterrows():
    rid = row["entity_id"]
    got = (row["status"], round(float(row["balance"]),2), int(row["updated_at"]))
    exp = expected[rid]
    print(rid, "online", got, "offline", exp, "MATCH" if got==exp else "MISMATCH")
