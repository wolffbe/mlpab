import os
from google.cloud import aiplatform_v1
from google.protobuf.json_format import MessageToDict

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
NAME = "churntrainingcdae59"
EXPECT_URI = f"bq://{PROJECT}.{DATASET}.{NAME}"
ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"

client = aiplatform_v1.DatasetServiceClient(
    transport="rest", client_options={"api_endpoint": ENDPOINT}
)
parent = f"projects/{PROJECT}/locations/{LOCATION}"
print("EXPECT URI:", EXPECT_URI)
for d in client.list_datasets(request={"parent": parent}):
    if d.display_name == NAME:
        md = MessageToDict(d._pb.metadata)
        print("name:", d.name)
        print("metadata:", md)
