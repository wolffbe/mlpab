import os
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
NAME = "churntrainingcdae59"
BQ_URI = f"bq://{PROJECT}.{DATASET}.{NAME}"

aiplatform.init(project=PROJECT, location=LOCATION)

# Reuse if it already exists to keep display_name unique/idempotent.
existing = aiplatform.TabularDataset.list(filter=f'display_name="{NAME}"')
if existing:
    ds = existing[0]
    print("reusing existing dataset:", ds.resource_name)
else:
    ds = aiplatform.TabularDataset.create(display_name=NAME, bq_source=BQ_URI)
    print("created dataset:", ds.resource_name)

print("display_name:", ds.display_name)
print("metadata bq source:", ds.to_dict().get("metadata"))
