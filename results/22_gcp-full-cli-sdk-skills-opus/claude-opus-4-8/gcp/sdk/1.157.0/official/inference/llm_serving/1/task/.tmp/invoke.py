import os, json, time
import google.cloud.aiplatform as aiplatform

aiplatform.init(project=os.environ["GCP_PROJECT"], location=os.environ["GCP_LOCATION"], api_transport="rest")
endpoint = aiplatform.Endpoint("projects/1014453977696/locations/***REDACTED***/endpoints/7590018926491205632")

with open("data/payloads.json") as f:
    payloads = json.load(f)

def try_predict():
    resp = endpoint.predict(instances=[payloads[0]])
    return resp.predictions

# retry predict a few times (cold start)
for attempt in range(6):
    try:
        p = try_predict()
        print("predict OK:", p, flush=True)
        break
    except Exception as e:
        print(f"attempt {attempt} predict failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(20)

# try raw_predict to observe direct container response
try:
    body = json.dumps({"instances": [payloads[0]]}).encode("utf-8")
    r = endpoint.raw_predict(body=body, headers={"Content-Type": "application/json"})
    print("raw_predict status:", getattr(r, "status_code", "?"), flush=True)
    print("raw_predict text:", getattr(r, "text", r), flush=True)
except Exception as e:
    print("raw_predict failed:", type(e).__name__, e, flush=True)
