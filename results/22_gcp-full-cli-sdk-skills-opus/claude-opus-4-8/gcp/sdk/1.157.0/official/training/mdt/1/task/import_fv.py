import os
from google.cloud.aiplatform_v1.services.featurestore_service import (
    FeaturestoreServiceClient,
)
from google.cloud.aiplatform_v1 import types as t
from google.api_core.client_options import ClientOptions

LOCATION = os.environ["GCP_LOCATION"]
PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

co = ClientOptions(api_endpoint=f"{LOCATION}-aiplatform.googleapis.com")
c = FeaturestoreServiceClient(client_options=co, transport="rest")
et_name = (
    f"projects/{PROJECT}/locations/{LOCATION}/featurestores/"
    f"{PREFIX}_scaled7b36f6/entityTypes/scaled7b36f6"
)
src = t.BigQuerySource(input_uri=f"bq://{PROJECT}.{DATASET}.scaled_source")
specs = [
    t.ImportFeatureValuesRequest.FeatureSpec(id=x)
    for x in ["split", "f1", "f2", "f3", "f4"]
]
req = t.ImportFeatureValuesRequest(
    entity_type=et_name,
    bigquery_source=src,
    entity_id_field="row_id",
    feature_time_field="feature_timestamp",
    feature_specs=specs,
    worker_count=1,
)
print("importing...")
op = c.import_feature_values(request=req)
res = op.result(timeout=1800)
print("imported_entity_count:", res.imported_entity_count)
print("imported_feature_value_count:", res.imported_feature_value_count)
print("invalid_row_count:", res.invalid_row_count)
