import os
import google.cloud.aiplatform as aiplatform
from google.cloud import bigquery
from vertexai.resources.preview import feature_store as fs

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
loc = os.environ["GCP_LOCATION"]
aiplatform.init(project=project, location=loc, api_transport="rest")
bq = bigquery.Client(project=project, location=loc)
ds = f"{project}.{dataset}"

# FeatureGroup (offline feature table) + its features and record key
fg = fs.FeatureGroup("scored3ace95")
print("FeatureGroup:", fg.resource_name)
print("  source uri:", fg.gca_resource.big_query.big_query_source.input_uri)
print("  entity_id_columns:", list(fg.gca_resource.big_query.entity_id_columns))
print("  labels:", dict(fg.gca_resource.labels))
print("  features:", sorted(f.name for f in fg.list_features()))

# BigQuery read-back of the feature table
t = bq.get_table(f"{ds}.scored3ace95")
print("BQ table columns:", [f.name for f in t.schema], "rows:", t.num_rows)
row = list(bq.query(
    f"SELECT request_id, account_id, distance_deg, score "
    f"FROM `{ds}.scored3ace95` ORDER BY request_id LIMIT 3", location=loc).result())
for r in row:
    print("  ", dict(r))
