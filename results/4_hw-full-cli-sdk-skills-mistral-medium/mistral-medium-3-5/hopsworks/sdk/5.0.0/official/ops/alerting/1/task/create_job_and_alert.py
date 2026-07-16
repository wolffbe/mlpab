#!/usr/bin/env python3
"""Script to create job flaky57faa7 and configure failure alert."""
import hopsworks
import json
import os

# Login to Hopsworks
hopsworks.login()

# Get the current project
project = hopsworks.get_current_project()

# Upload the failing_job.py to the project's Resources directory
dataset_api = project.get_dataset_api()
# Check if file exists first
if not dataset_api.exists("Resources/failing_job.py"):
    dataset_api.upload("data/failing_job.py", "Resources/failing_job.py")
    print("Uploaded failing_job.py to Resources/")
else:
    print("failing_job.py already exists in Resources/")

# Get the job API
job_api = project.get_job_api()

# Get configuration for a Python job
config = job_api.get_configuration("PYTHON")
print(f"Default Python config: {config}")

# Configure the job to run the failing script
config["appPath"] = "Resources/failing_job.py"

# Create the job
job = job_api.create_job("flaky57faa7", config)
print(f"Created job: {job.name} with ID: {job.id}")

# Get alerts API and create a receiver first
alerts_api = project.get_alerts_api()

# Try to create a simple alert receiver
# Let's try with webhook since it might not require external configuration
try:
    receiver = alerts_api.create_alert_receiver(
        name="flaky57faa7_alert",
        webhook_configs=[{"url": "http://example.com/webhook"}]
    )
    print(f"Created alert receiver: {receiver.name}")
except Exception as e:
    print(f"Could not create webhook receiver: {e}")
    # Try with email
    try:
        receiver = alerts_api.create_alert_receiver(
            name="flaky57faa7_alert",
            email_configs=[{"to": "test@example.com"}]
        )
        print(f"Created alert receiver: {receiver.name}")
    except Exception as e2:
        print(f"Could not create email receiver: {e2}")
        # Try without creating a receiver - maybe we can use a built-in one
        pass

# List existing receivers
receivers = alerts_api.get_alert_receivers()
print(f"Available receivers: {[r.name for r in receivers]}")

# Create an alert for job failures using the receiver we created or a default one
# If we couldn't create a receiver, try using a simple name
try:
    alert = job.create_alert(
        receiver="flaky57faa7_alert",
        status="failed",
        severity="warning"
    )
    print(f"Created alert: {alert}")
except Exception as e:
    print(f"Failed to create alert with custom receiver: {e}")
    # Try with a simpler receiver name
    try:
        alert = job.create_alert(
            receiver="email",
            status="failed",
            severity="warning"
        )
        print(f"Created alert with 'email' receiver: {alert}")
    except Exception as e2:
        print(f"Failed to create alert with 'email' receiver: {e2}")
        # Try with 'webhook'
        try:
            alert = job.create_alert(
                receiver="webhook",
                status="failed",
                severity="warning"
            )
            print(f"Created alert with 'webhook' receiver: {alert}")
        except Exception as e3:
            print(f"Failed to create alert with 'webhook' receiver: {e3}")
            raise

# Run the job once (it will fail)
print("Running job...")
try:
    execution = job.run(await_termination=True)
    print(f"Job execution state: {execution.state}")
except Exception as e:
    print(f"Job failed as expected: {e}")

# Write the answers.json
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({
        "job_name": "flaky57faa7",
        "alert": "flaky57faa7_alert"
    }, f)

print("Done! answers.json written.")
