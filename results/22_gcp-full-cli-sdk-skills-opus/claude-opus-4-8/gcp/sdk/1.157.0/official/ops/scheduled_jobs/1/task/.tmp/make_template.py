"""Hand-author a minimal KFP v2 IR pipeline spec that runs the heartbeat script
in a container component, and write it to a template JSON for Vertex PipelineJob."""
import json, os

# The heartbeat body, run inside a stock python container on the platform.
HEARTBEAT_SRC = (
    "import datetime\n"
    "TOKEN='HB-14626077'\n"
    "print(f'heartbeat {TOKEN} alive at "
    "{datetime.datetime.now(datetime.timezone.utc).isoformat()}')\n"
)

pipeline_spec = {
    "components": {
        "comp-heartbeat": {
            "executorLabel": "exec-heartbeat",
        }
    },
    "deploymentSpec": {
        "executors": {
            "exec-heartbeat": {
                "container": {
                    "image": "python:3.11-slim",
                    "command": ["python3", "-c", HEARTBEAT_SRC],
                    "args": [],
                }
            }
        }
    },
    "pipelineInfo": {"name": "heartbeat518d73"},
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

path = os.path.join(os.path.dirname(__file__), "heartbeat_pipeline.json")
with open(path, "w") as f:
    json.dump(pipeline_spec, f, indent=2)
print("wrote", path)
