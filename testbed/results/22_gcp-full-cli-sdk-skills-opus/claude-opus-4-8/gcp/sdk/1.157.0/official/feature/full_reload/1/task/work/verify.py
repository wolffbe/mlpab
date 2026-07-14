import os
for v in ("GRPC_PROXY", "grpc_proxy"):
    os.environ.pop(v, None)
import vertexai
from vertexai.resources.preview import feature_store as fs
import google.cloud.bigquery as bq

project = os.environ['GCP_PROJECT']
location = os.environ['GCP_LOCATION']
dataset = os.environ['GCP_BQ_DATASET']
prefix = os.environ['MLPAB_GCP_PREFIX']
vertexai.init(project=project, location=location, api_transport="rest")

for ver in ("1", "2"):
    fg = fs.FeatureGroup(f"{prefix}_customerscd1186_{ver}")
    r = fg.gca_resource
    print(f"\n=== FeatureGroup {fg.name} (version {ver}) ===")
    print("  source:", r.big_query.big_query_source.input_uri)
    print("  entity_id_columns:", list(r.big_query.entity_id_columns))
    print("  features:", sorted(f.name for f in fg.list_features()))

# BigQuery readback of the graded v2 table
client = bq.Client(project=project)
t = client.get_table(f"{project}.{dataset}.customerscd1186_2")
print("\n=== BigQuery v2 table readback ===")
print("  cols:", [s.name for s in t.schema])
print("  rows:", t.num_rows)
q = f"SELECT row_id, full_name, balance, currency, updated_at FROM `{project}.{dataset}.customerscd1186_2` ORDER BY row_id LIMIT 3"
for row in client.query(q).result():
    print("  ", dict(row))

# confirm no old columns / no stale rows crossover
q2 = (f"SELECT COUNT(*) c FROM `{project}.{dataset}.customerscd1186_2` v2 "
      f"WHERE v2.row_id NOT IN (SELECT row_id FROM `{project}.{dataset}.customerscd1186_2`)")
print("  distinct row_ids:", list(client.query(
    f"SELECT COUNT(DISTINCT row_id) d, COUNT(*) t FROM `{project}.{dataset}.customerscd1186_2`").result())[0])

# leftover online store?
print("\n=== online stores (mine?) ===")
for s in fs.FeatureOnlineStore.list():
    print("  ", s.name)
