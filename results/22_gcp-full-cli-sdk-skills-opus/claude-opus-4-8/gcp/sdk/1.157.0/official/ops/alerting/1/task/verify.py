import os, google.auth
from google.auth.transport.requests import AuthorizedSession
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ['GCP_PROJECT']
LOCATION = os.environ['GCP_LOCATION']
PREFIX = os.environ['MLPAB_GCP_PREFIX']

# verify alert policy
creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/monitoring"])
s = AuthorizedSession(creds)
r = s.get(f"https://monitoring.googleapis.com/v3/projects/{PROJECT}/alertPolicies")
pols = r.json().get("alertPolicies", [])
for p in pols:
    if "flakycdcd16" in p.get("displayName", ""):
        print("ALERT:", p["displayName"], "| enabled=", p.get("enabled"),
              "| filter=", p["conditions"][0]["conditionMatchedLog"]["filter"])

# verify job
aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")
for j in aiplatform.CustomJob.list():
    if "flakycdcd16" in j.display_name:
        print("JOB:", j.display_name, "| state=", j.state, "| resource=", j.resource_name)
