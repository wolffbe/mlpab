"""
Orchestrator — runs locally using only the Hopsworks SDK (pandas for data prep, no ML libs).
Uploads job scripts to the platform, runs training + inference jobs there.
"""
import hopsworks
import pandas as pd

# ── 1. Connect ────────────────────────────────────────────────────────────────
project = hopsworks.login()
fs = project.get_feature_store()

# ── 2. Feature pipeline ───────────────────────────────────────────────────────
history_df = pd.read_csv("data/airquality_history.csv", parse_dates=["date"])

history_df["month"]       = history_df["date"].dt.month
history_df["day_of_year"] = history_df["date"].dt.dayofyear
history_df["date"]        = history_df["date"].dt.strftime("%Y-%m-%d")

print("History shape:", history_df.shape)

fg = fs.get_or_create_feature_group(
    name="airq3c8c0c",
    version=1,
    primary_key=["date"],
    description="Air quality + weather features",
    online_enabled=True,
)
fg.insert(history_df, write_options={"wait_for_job": True})
print("Feature group insert done.")

fv = fs.get_or_create_feature_view(
    name="airqtd3c8c0c",
    version=1,
    query=fg.select_all(),
    labels=["pm25"],
)
print("Feature view ready.")

# ── 3. Upload resources to HopsFS ─────────────────────────────────────────────
dataset_api = project.get_dataset_api()

try:
    dataset_api.mkdir("Resources")
except Exception:
    pass

dataset_api.upload("training_job.py",        "Resources", overwrite=True)
dataset_api.upload("inference_job.py",       "Resources", overwrite=True)
dataset_api.upload("data/forecast_days.csv", "Resources", overwrite=True)
print("Files uploaded to HopsFS Resources/.")

# ── 4. Create + run training job ──────────────────────────────────────────────
job_api = project.get_jobs_api()

train_config = job_api.get_configuration("PYTHON")
train_config["appPath"] = "Resources/training_job.py"
train_job = job_api.create_job("train_airq3c8c0c", train_config)
print("Training job ready, starting …")

train_exec = train_job.run(await_termination=True)
print("Training execution state:", train_exec.state)
print("Training execution final state:", train_exec.final_status)

# Print logs regardless of status
try:
    logs = train_exec.get_logs()
    print("=== Training logs ===")
    print(logs[:3000] if logs else "(no logs)")
except Exception as e:
    print("Could not fetch logs:", e)

if train_exec.final_status not in ("SUCCEEDED", "Undefined"):
    raise RuntimeError(f"Training job failed: {train_exec.final_status}")

# ── 5. Create + run inference job ─────────────────────────────────────────────
infer_config = job_api.get_configuration("PYTHON")
infer_config["appPath"] = "Resources/inference_job.py"
infer_job = job_api.create_job("infer_airq3c8c0c", infer_config)
print("Inference job ready, starting …")

infer_exec = infer_job.run(await_termination=True)
print("Inference execution state:", infer_exec.state)
print("Inference execution final state:", infer_exec.final_status)

try:
    logs = infer_exec.get_logs()
    print("=== Inference logs ===")
    print(logs[:3000] if logs else "(no logs)")
except Exception as e:
    print("Could not fetch logs:", e)

if infer_exec.final_status not in ("SUCCEEDED", "Undefined"):
    raise RuntimeError(f"Inference job failed: {infer_exec.final_status}")

print("\nPipeline complete — predictions stored in airqpred3c8c0c.")
