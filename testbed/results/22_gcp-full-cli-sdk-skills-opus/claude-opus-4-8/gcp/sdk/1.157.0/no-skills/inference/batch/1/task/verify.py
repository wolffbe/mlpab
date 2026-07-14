import os
import google.cloud.aiplatform as aiplatform

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

aiplatform.init(project=PROJECT, location=LOCATION, api_transport="rest")
fs = aiplatform.Featurestore(f"{PREFIX}_fs")
et = fs.get_entity_type("scores36e30a")

# Online (low-latency) read for a few record keys
df = et.read(entity_ids=["A0000", "A0001", "A0002", "A0003", "A0004"], feature_ids=["score"])
print("ONLINE READ:")
print(df.to_string(index=False))
print("features:", [f.name for f in et.list_features()])
