import os
from google.cloud import aiplatform_v1
from google.protobuf import struct_pb2

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
NAME = "churntrainingcdae59"
BQ_URI = f"bq://{PROJECT}.{DATASET}.{NAME}"
ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"

client = aiplatform_v1.DatasetServiceClient(
    transport="rest", client_options={"api_endpoint": ENDPOINT}
)
parent = f"projects/{PROJECT}/locations/{LOCATION}"

# idempotency for THIS run: reuse only if it points at our per-run BQ table.
for d in client.list_datasets(request={"parent": parent}):
    if d.display_name == NAME:
        uri = d.metadata.get("inputConfig", {}).get("bigquerySource", {}).get("uri", "")
        if uri == BQ_URI:
            print("reusing existing dataset for this run:", d.name)
            raise SystemExit(0)

metadata = struct_pb2.Value()
metadata.struct_value.update({"inputConfig": {"bigquerySource": {"uri": BQ_URI}}})

dataset = aiplatform_v1.Dataset(
    display_name=NAME,
    metadata_schema_uri="gs://google-cloud-aiplatform/schema/dataset/metadata/tabular_1.0.0.yaml",
    metadata=metadata,
)
op = client.create_dataset(request={"parent": parent, "dataset": dataset})
result = op.result(timeout=600)
print("created dataset:", result.name)
print("display_name:", result.display_name)
