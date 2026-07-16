import os
import json
import google.auth
from google.auth.transport.requests import AuthorizedSession

PROJECT = os.environ['GCP_PROJECT']
PREFIX = os.environ['MLPAB_GCP_PREFIX']
JOB_ID = "4478837457281875968"

creds, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/monitoring"]
)
session = AuthorizedSession(creds)

alert_name = f"{PREFIX}_flakycdcd16_failure_alert"

log_filter = (
    'resource.type="ml_job" '
    f'AND resource.labels.job_id="{JOB_ID}" '
    'AND severity>=ERROR'
)

policy = {
    "displayName": alert_name,
    "documentation": {
        "content": (
            "Failure alert for Vertex AI CustomJob 'flakycdcd16' "
            f"(display_name {PREFIX}_flakycdcd16, job id {JOB_ID}). "
            "Fires when the job emits an error log / fails."
        ),
        "mimeType": "text/markdown",
    },
    "combiner": "OR",
    "conditions": [
        {
            "displayName": "flakycdcd16 custom job failure (error log)",
            "conditionMatchedLog": {"filter": log_filter},
        }
    ],
    "alertStrategy": {
        "notificationRateLimit": {"period": "300s"}
    },
    "enabled": True,
}

url = f"https://monitoring.googleapis.com/v3/projects/{PROJECT}/alertPolicies"
resp = session.post(url, data=json.dumps(policy),
                    headers={"Content-Type": "application/json"})
print("HTTP", resp.status_code)
body = resp.json()
if resp.status_code >= 300:
    print(json.dumps(body, indent=2))
    raise SystemExit(1)
print("ALERT_NAME:", body.get("name"))
print("ALERT_DISPLAY:", body.get("displayName"))
