import databricks.sdk
import io

w = databricks.sdk.WorkspaceClient()
SCHEMA = "workspace.mlpab98b0c2"
WH = "4dfab06c923fe3cc"

vol = "compare_vol"
w.statement_execution.execute_statement(
    warehouse_id=WH,
    statement=f"CREATE VOLUME IF NOT EXISTS {SCHEMA}.{vol}",
    wait_timeout="30s",
)
print("volume created")

base = f"/Volumes/workspace/mlpab98b0c2/{vol}"
for fn in ["training_sample.csv", "serving_log.csv"]:
    with open(f"data/{fn}", "rb") as f:
        data = f.read()
    w.files.upload(f"{base}/{fn}", io.BytesIO(data), overwrite=True)
    print("uploaded", fn)
