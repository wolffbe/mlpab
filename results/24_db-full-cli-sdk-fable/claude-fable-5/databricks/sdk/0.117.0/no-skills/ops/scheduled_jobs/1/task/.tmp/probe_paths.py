from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
user = w.current_user.me().user_name
notebook_path = f"/Users/{user}/mlpab23fe6e/heartbeat"

body = {
    "name": "heartbeat514afe",
    "tasks": [{
        "task_key": "heartbeat",
        "notebook_task": {"notebook_path": notebook_path},
    }],
    "schedule": {
        "quartz_cron_expression": "0 0 * * * ?",
        "timezone_id": "UTC",
        "pause_status": "UNPAUSED",
    },
}

for version in ("2.1", "2.0"):
    try:
        res = w.api_client.do("POST", f"/api/{version}/jobs/create", body=body)
        print(f"create via {version} OK:", res)
        break
    except Exception as e:
        print(f"create via {version} FAILED:", type(e).__name__, e)

# one-time submit probe
try:
    res = w.api_client.do("POST", "/api/2.1/jobs/runs/submit", body={
        "run_name": "mlpab23fe6e_probe",
        "tasks": [{
            "task_key": "heartbeat",
            "notebook_task": {"notebook_path": notebook_path},
        }],
    })
    print("runs/submit OK:", res)
except Exception as e:
    print("runs/submit FAILED:", type(e).__name__, e)
