import google.cloud.aiplatform as aiplatform
import os
proj=os.environ['GCP_PROJECT']; loc=os.environ['GCP_LOCATION']
aiplatform.init(project=proj, location=loc)

# find our model
for m in aiplatform.Model.list():
    print("MODEL", m.resource_name, "|display:", m.display_name)

print("--- feature store classes ---")
names=[n for n in dir(aiplatform) if 'eature' in n]
print(names)
try:
    from google.cloud.aiplatform_v1 import types as T
    print("has aiplatform_v1")
except Exception as e:
    print("v1 err", e)
