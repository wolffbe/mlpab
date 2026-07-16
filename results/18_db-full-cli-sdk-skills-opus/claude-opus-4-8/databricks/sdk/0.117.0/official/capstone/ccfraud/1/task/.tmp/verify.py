from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
SCHEMA = "workspace.mlpabea3b07"
wh = "4dfab06c923fe3cc"

def q(sql):
    r = w.statement_execution.execute_statement(warehouse_id=wh, statement=sql, wait_timeout="50s")
    if r.result and r.result.data_array:
        return r.result.data_array
    return []

for t in ["cctxn2dbe0a", "cctd2dbe0a", "ccpred2dbe0a"]:
    print(t, q(f"SELECT count(*) FROM {SCHEMA}.{t}"))
print("pred sample", q(f"SELECT transaction_id, fraud_probability FROM {SCHEMA}.ccpred2dbe0a LIMIT 3"))
print("prob range", q(f"SELECT min(fraud_probability), max(fraud_probability), avg(fraud_probability) FROM {SCHEMA}.ccpred2dbe0a"))
print("models:")
for m in w.registered_models.list(catalog_name="workspace", schema_name="mlpabea3b07"):
    print(" ", m.full_name)
    for v in (w.model_versions.list(full_name=m.full_name) or []):
        print("    v", v.version)
