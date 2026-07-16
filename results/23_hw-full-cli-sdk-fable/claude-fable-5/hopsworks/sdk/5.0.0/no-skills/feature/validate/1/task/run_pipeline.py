import hopsworks
import csv
import json
import os

VALID_CATEGORIES = {"grocery", "travel", "salary", "rent", "other"}

valid_rows = []
rejected = []
with open("data/events.csv", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for r in reader:
        amount_raw = (r["amount"] or "").strip()
        category = r["category"] if r["category"] is not None else ""
        ok = True
        if amount_raw == "":
            ok = False
        else:
            try:
                amount = float(amount_raw)
                if not (0.0 <= amount <= 10000.0):
                    ok = False
            except ValueError:
                ok = False
        if category not in VALID_CATEGORIES:
            ok = False
        if ok:
            valid_rows.append(r)
        else:
            rejected.append(r["row_id"])

print(f"total={len(valid_rows) + len(rejected)} valid={len(valid_rows)} rejected={len(rejected)}")

os.makedirs(".tmp", exist_ok=True)
with open(".tmp/valid_events.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(valid_rows)

project = hopsworks.login()
ds = project.get_dataset_api()

up_csv = ds.upload(".tmp/valid_events.csv", "Resources", overwrite=True)
up_py = ds.upload("platform_ingest.py", "Resources", overwrite=True)
print("uploaded:", up_csv, up_py)

jobs = project.get_job_api()
cfg = jobs.get_configuration("PYTHON")
cfg["appPath"] = f"/Projects/{project.name}/Resources/platform_ingest.py"
cfg["environmentName"] = "python-feature-pipeline"
cfg["resourceConfig"]["memory"] = 4096
job = jobs.create_job("ingest_eventsee881b", cfg)
print("job created:", job.name)

execution = job.run(await_termination=True)
print("execution state:", execution.state, "final status:", execution.final_status)

try:
    out, err = execution.download_logs()
    for p in (out, err):
        if p and os.path.exists(p):
            print(f"--- {p} ---")
            print(open(p).read()[-8000:])
except Exception as e:
    print("log fetch failed:", e)

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"rejected": rejected}, f, indent=2)
print("wrote submission/answers.json with", len(rejected), "rejected ids")

if execution.final_status.lower() != "succeeded":
    raise SystemExit("platform job did not succeed")
