import subprocess
import sys

try:
    from google.cloud import storage
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-cloud-storage"], check=True)
    from google.cloud import storage

BUCKET = "cloud-ai-platform-5dcfee9a-d8bf-457a-8b19-6d8f5db58035"
PREFIX = "mlpab011355_trainjob6329cd"

client = storage.Client()
bucket = client.bucket(BUCKET)

for name in ["train.csv", "score.csv", "train_model.py"]:
    bucket.blob(PREFIX + "/input/" + name).download_to_filename(name)
    print("downloaded " + name, flush=True)

subprocess.run([sys.executable, "train_model.py"], check=True)
print("train_model.py finished", flush=True)

bucket.blob(PREFIX + "/output/predictions.csv").upload_from_filename("predictions.csv")
print("uploaded predictions.csv", flush=True)
