"""Wait for the DLT pipeline to complete syncing the online table."""
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.pipelines import PipelineState, UpdateStateInfoState

PIPELINE_ID = "e90e30b2-455c-4a90-aa8d-f00f7fc02ab2"

w = WorkspaceClient()

print(f"Monitoring pipeline {PIPELINE_ID}...")
deadline = time.time() + 1200  # 20 minute timeout

while time.time() < deadline:
    pipeline = w.pipelines.get(pipeline_id=PIPELINE_ID)
    state = pipeline.state
    latest = pipeline.latest_updates[0] if pipeline.latest_updates else None
    update_state = latest.state if latest else None
    print(f"  Pipeline state: {state}, update state: {update_state}")

    # Check if the update is complete (COMPLETED or FAILED or CANCELED)
    if update_state in (
        UpdateStateInfoState.COMPLETED,
        UpdateStateInfoState.FAILED,
        UpdateStateInfoState.CANCELED,
    ):
        print(f"Update finished with state: {update_state}")
        break

    # If pipeline is idle (no active run), that might mean it completed already
    if state == PipelineState.IDLE and update_state == UpdateStateInfoState.COMPLETED:
        print("Pipeline completed successfully!")
        break

    time.sleep(20)
else:
    print("Timeout - pipeline may still be running")

# Final status
pipeline = w.pipelines.get(pipeline_id=PIPELINE_ID)
print(f"\nFinal pipeline state: {pipeline.state}")
if pipeline.latest_updates:
    print(f"Latest update: {pipeline.latest_updates[0]}")

print("\nOnline table should be populated.")
