import databricks.sdk

w = databricks.sdk.WorkspaceClient()
SCHEMA = "workspace.mlpab98b0c2"
WH = "4dfab06c923fe3cc"


def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout="50s")
    return r.result.data_array


# Test hypothesis: serve_f3 = expm1(train_f3)  i.e. train_f3 = log1p(serve_f3)
rows = run(f"""
SELECT
  max(abs(s.f3 - (exp(t.f3)-1))) max_err_expm1,
  avg(abs(s.f3 - (exp(t.f3)-1))) avg_err_expm1,
  max(abs(t.f3 - ln(s.f3+1))) max_err_log1p,
  avg(abs(t.f3 - ln(s.f3+1))) avg_err_log1p
FROM {SCHEMA}.train t JOIN {SCHEMA}.serve s ON t.entity_id=s.entity_id
""")
print("max_err(serve=expm1(train)), avg, max_err(train=log1p(serve)), avg")
print([round(float(x),8) for x in rows[0]])
