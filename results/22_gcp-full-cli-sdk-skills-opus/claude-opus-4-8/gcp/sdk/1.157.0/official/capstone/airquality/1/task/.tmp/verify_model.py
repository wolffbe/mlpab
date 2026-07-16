import os
from google.cloud import aiplatform_v1

proj = os.environ['GCP_PROJECT']; loc = os.environ['GCP_LOCATION']
prefix = os.environ['MLPAB_GCP_PREFIX']
ep = f"{loc}-aiplatform.googleapis.com"
client = aiplatform_v1.ModelServiceClient(
    transport="rest", client_options={"api_endpoint": ep})
parent = f"projects/{proj}/locations/{loc}"
print("listing models via REST...")
for m in client.list_models(parent=parent):
    if "airqmodelf3f1d8" in m.display_name:
        print("MODEL:", m.display_name, "| name:", m.name.split('/')[-1],
              "| versions? id:", m.version_id)
