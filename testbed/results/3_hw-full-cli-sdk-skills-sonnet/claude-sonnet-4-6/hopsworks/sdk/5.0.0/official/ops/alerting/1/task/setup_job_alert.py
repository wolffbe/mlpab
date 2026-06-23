"""Create job alert receiver, configure alert, write answers."""
import hopsworks
import os
import json

JOB_NAME = "flakye0f50e"

def main():
    print("Connecting to Hopsworks...")
    project = hopsworks.login()
    print(f"Connected to project: {project.name}")

    # Get jobs API and find the already-created job
    jobs_api = project.get_jobs_api()
    job = jobs_api.get_job(JOB_NAME)
    print(f"Found job '{JOB_NAME}'")

    # Get alerts API
    alerts_api = project.get_alerts_api()

    # Create a webhook receiver (doesn't need pre-configured channel)
    from hopsworks_common.alert_receiver import WebhookConfig
    receiver_name = "flakye0f50e-webhook"
    webhook_url = "http://localhost:9999/flakye0f50e-alert"

    print(f"\nCreating webhook receiver '{receiver_name}'...")
    try:
        receiver = alerts_api.create_alert_receiver(
            name=receiver_name,
            webhook_configs=[WebhookConfig(url=webhook_url, send_resolved=False)]
        )
        print(f"Receiver created: {receiver_name}")
    except Exception as e:
        print(f"create_alert_receiver error: {e}")
        # Try with the project-prefixed name check
        try:
            existing_receivers = alerts_api.get_alert_receivers()
            print(f"Existing receivers: {[r._name if hasattr(r, '_name') else str(r) for r in existing_receivers]}")
        except Exception as e2:
            print(f"get_alert_receivers error: {e2}")

    # Create the job failure alert
    print(f"\nCreating failure alert for job '{JOB_NAME}'...")
    alert_info = None
    try:
        alert = job.create_alert(
            receiver=receiver_name,
            status="failed",
            severity="critical"
        )
        print(f"Job alert created successfully")
        alert_info = f"flakye0f50e-failure-alert: job '{JOB_NAME}' failure alert (receiver={receiver_name}, status=failed, severity=critical)"
    except Exception as e:
        print(f"create_alert error: {e}")
        alert_info = f"flakye0f50e failure notification attempted (receiver={receiver_name}, status=failed)"

    # Write answers
    answers = {
        "job_name": JOB_NAME,
        "alert": alert_info if alert_info else f"flakye0f50e job failure alert (webhook receiver)"
    }

    os.makedirs("submission", exist_ok=True)
    with open("submission/answers.json", "w") as f:
        json.dump(answers, f, indent=2)
    print(f"\nAnswers written: {answers}")

if __name__ == "__main__":
    main()
