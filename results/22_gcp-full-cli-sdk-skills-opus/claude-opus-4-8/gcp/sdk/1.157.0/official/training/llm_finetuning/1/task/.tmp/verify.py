import os
import google.cloud.aiplatform as aiplatform

PREFIX = os.environ["MLPAB_GCP_PREFIX"]
aiplatform.init(project=os.environ["GCP_PROJECT"], location=os.environ["GCP_LOCATION"],
                api_transport="rest")

models = aiplatform.Model.list(filter=f'display_name="{PREFIX}_ftmodel2e5343"')
for m in models:
    print("MODEL", m.display_name, "version", m.version_id, "desc", m.description,
          "labels", m.labels, "artifact", m.gca_resource.artifact_uri)

jobs = aiplatform.CustomJob.list(filter=f'display_name="{PREFIX}_ftjob2e5343"')
for j in jobs:
    print("JOB", j.display_name, "state", j.state)
