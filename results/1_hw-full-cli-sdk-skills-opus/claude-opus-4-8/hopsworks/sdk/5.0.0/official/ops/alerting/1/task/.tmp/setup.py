import warnings, urllib3, os, json, traceback
warnings.filterwarnings('ignore')
urllib3.disable_warnings()
import hopsworks

JOB = "flakya9c625"
proj = hopsworks.login()
ds = proj.get_dataset_api()
ja = proj.get_job_api()

# 1) Upload script to HopsFS Resources/
upload_dir = "Resources"
remote = ds.upload("data/failing_job.py", upload_dir, overwrite=True)
print("UPLOADED ->", remote)
app_path = f"/Projects/{proj.name}/Resources/failing_job.py"

# 2) Create the job
if ja.exists(JOB):
    job = ja.get_job(JOB)
    print("Job already exists")
else:
    cfg = ja.get_configuration("PYTHON")
    cfg["appPath"] = app_path
    job = ja.create_job(name=JOB, config=cfg)
    print("CREATED job", job.name)

# 3) Create alert receiver (try email, fallback others)
aa = proj.get_alerts_api()
recv_name = None
existing = aa.get_alert_receivers() or []
for r in existing:
    recv_name = getattr(r, "name", None)
    if recv_name:
        print("Using existing receiver:", recv_name)
        break
if not recv_name:
    for kind, kwargs in [
        ("email", {"email_configs": [{"to": "flakya9c625-alerts@example.com"}]}),
        ("slack", {"slack_configs": [{"channel": "#flakya9c625-alerts"}]}),
        ("webhook", {"webhook_configs": [{"url": "https://example.com/flakya9c625"}]}),
    ]:
        try:
            rcv = aa.create_alert_receiver(name=f"flakya9c625-{kind}", **kwargs)
            recv_name = getattr(rcv, "name", f"flakya9c625-{kind}")
            print("CREATED receiver:", recv_name)
            break
        except Exception as e:
            print(f"receiver {kind} failed:", repr(e)[:200])

print("RECEIVER FINAL:", recv_name)
