from google.cloud import aiplatform_v1
import os
proj=os.environ['GCP_PROJECT']; loc=os.environ['GCP_LOCATION']
ep=f"{loc}-aiplatform.googleapis.com"
client=aiplatform_v1.ModelServiceClient(client_options={"api_endpoint":ep}, transport="rest")
parent=f"projects/{proj}/locations/{loc}"
for m in client.list_models(parent=parent):
    print("MODEL", m.name, "|", m.display_name)
print("OK REST works")
