from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
schema = "workspace.mlpab60c44c"

r = w.statement_execution.execute_statement(
    statement=f"CREATE VOLUME IF NOT EXISTS {schema}.raw",
    warehouse_id="8a93fc195da2ceb1", wait_timeout="50s")
print("create volume:", r.status.state, r.status.error)

for f in ["transactions.csv", "fx_rates.csv"]:
    with open(f"data/{f}", "rb") as fh:
        w.files.upload(f"/Volumes/workspace/mlpab60c44c/raw/{f}", fh, overwrite=True)
    print("uploaded", f)
