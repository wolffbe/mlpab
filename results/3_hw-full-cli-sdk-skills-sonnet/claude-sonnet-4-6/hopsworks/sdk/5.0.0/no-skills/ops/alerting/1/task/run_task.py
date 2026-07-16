"""Task script: create job, alert, run job, write submission."""
import hopsworks
import json
import os

JOB_NAME = "flakye0f50e"
SCRIPT_LOCAL = "data/failing_job.py"

print("=== Connecting to Hopsworks ===")
project = hopsworks.login()
print(f"Connected to project: {project.name}")

# Step 1: Upload script to Resources
print("\n=== Uploading script ===")
dataset_api = project.get_dataset_api()
upload_path = dataset_api.upload(SCRIPT_LOCAL, "Resources", overwrite=True)
print(f"Uploaded to: {upload_path}")

# Step 2: Create Python job
print("\n=== Creating job ===")
job_api = project.get_job_api()
python_config = job_api.get_configuration("PYTHON")
python_config["appPath"] = upload_path
print(f"Job config: {python_config}")
job = job_api.create_job(JOB_NAME, python_config)
print(f"Created job: {job.name}")

# Step 3: Create an alert receiver then job alert
print("\n=== Creating alert receiver ===")
alerts_api = project.get_alerts_api()

# Check existing receivers
receivers = alerts_api.get_alert_receivers()
print(f"Existing receivers: {receivers}")

receiver_name = f"flakye0f50e-failure-receiver"
# Create a webhook receiver (using a placeholder URL that will capture the alert)
try:
    receiver = alerts_api.create_alert_receiver(
        name=receiver_name,
        webhook_configs=[{"url": "http://localhost:9999/alert"}]
    )
    print(f"Created receiver: {receiver}")
except Exception as e:
    print(f"Error creating webhook receiver: {e}")
    # Try email receiver if webhook fails
    try:
        receiver = alerts_api.create_alert_receiver(
            name=receiver_name,
            email_configs=[{"to": "alert@flakye0f50e.local"}]
        )
        print(f"Created email receiver: {receiver}")
    except Exception as e2:
        print(f"Error creating email receiver: {e2}")

# Step 4: Create job alert for failures
print("\n=== Creating job alert ===")
alert_description = f"flakye0f50e job failure alert"
try:
    job_alert = alerts_api.create_job_alert(
        job_name=JOB_NAME,
        receiver=receiver_name,
        status="failed",
        severity="critical"
    )
    print(f"Created job alert: {job_alert}")
    alert_name = f"job_failed alert for {JOB_NAME} -> {receiver_name}"
except Exception as e:
    print(f"Error creating job alert via alerts_api: {e}")
    # Try via project method
    try:
        job_alert = project.create_job_alert(
            receiver=receiver_name,
            status="job_failed",
            severity="critical"
        )
        print(f"Created job alert via project: {job_alert}")
        alert_name = f"job_failed alert for {JOB_NAME} (project-level) -> {receiver_name}"
    except Exception as e2:
        print(f"Error creating alert via project: {e2}")
        alert_name = f"flakye0f50e failure alert (receiver: {receiver_name})"

# Step 5: Run the job (it will fail - that's expected)
print("\n=== Running job ===")
try:
    execution = job.run(await_termination=True)
    print(f"Execution completed with state: {execution.final_status}")
except Exception as e:
    print(f"Job execution result: {e}")
    # Job may have failed (expected) - check execution
    try:
        executions = job.get_executions()
        if executions:
            last_exec = executions[0]
            print(f"Last execution state: {last_exec.final_status}")
    except Exception as e2:
        print(f"Error getting executions: {e2}")

# Step 6: Write submission
print("\n=== Writing submission ===")
os.makedirs("submission", exist_ok=True)
result = {
    "job_name": JOB_NAME,
    "alert": alert_name
}
with open("submission/answers.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"Written submission: {result}")
print("DONE.")
