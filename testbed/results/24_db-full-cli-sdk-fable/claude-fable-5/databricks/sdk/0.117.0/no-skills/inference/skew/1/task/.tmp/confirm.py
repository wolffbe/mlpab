import databricks.sdk

w = databricks.sdk.WorkspaceClient()
WH = "8a93fc195da2ceb1"
SCHEMA = "workspace.mlpaba23cda"

r = w.statement_execution.execute_statement(
    statement=f"""
    SELECT avg(abs(t.f2 - ln(1 + s.f2))) AS mad_log1p,
           max(abs(t.f2 - ln(1 + s.f2))) AS max_log1p
    FROM {SCHEMA}.skew_training t JOIN {SCHEMA}.skew_serving s USING (entity_id)
    """,
    warehouse_id=WH,
    wait_timeout="50s",
)
assert r.status.state.value == "SUCCEEDED", r.status.error
for c, v in zip(r.manifest.schema.columns, r.result.data_array[0]):
    print(c.name, "=", v)
