import json

HEARTBEAT_SRC = open("data/heartbeat.py").read()
# Run the provided script verbatim inside the container.
cmd = ["python3", "-c", HEARTBEAT_SRC]

spec = {
    "components": {"comp-heartbeat": {"executorLabel": "exec-heartbeat"}},
    "deploymentSpec": {
        "executors": {
            "exec-heartbeat": {
                "container": {"image": "python:3.11-slim", "command": cmd}
            }
        }
    },
    "pipelineInfo": {"name": "heartbeat-pipeline"},
    "root": {
        "dag": {
            "tasks": {
                "heartbeat": {
                    "cachingOptions": {"enableCache": False},
                    "componentRef": {"name": "comp-heartbeat"},
                    "taskInfo": {"name": "heartbeat"},
                }
            }
        }
    },
    "schemaVersion": "2.1.0",
    "sdkVersion": "kfp-2.7.0",
}
with open(".tmp/heartbeat_pipeline.json", "w") as f:
    json.dump(spec, f, indent=2)
print("wrote pipeline spec")
