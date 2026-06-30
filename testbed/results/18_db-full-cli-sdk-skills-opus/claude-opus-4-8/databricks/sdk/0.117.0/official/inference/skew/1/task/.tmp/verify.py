import databricks.sdk
from databricks.sdk.service.sql import StatementState
import time

w = databricks.sdk.WorkspaceClient()
SCHEMA = "workspace.mlpab9f3ba9"
WH = "4dfab06c923fe3cc"


def run_sql(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout="50s")
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {r.status.state} {r.status.error}")
    return r


# Hypothesis: train_f3 = log1p(serve_f3)  i.e. serve_f3 = exp(train_f3) - 1
q = f"""
SELECT
  max(abs(t.f3 - ln(1 + s.f3))) AS max_err_log1p,
  avg(abs(t.f3 - ln(1 + s.f3))) AS avg_err_log1p,
  max(abs(s.f3 - (exp(t.f3) - 1))) AS max_err_expm1
FROM {SCHEMA}.train_t t
JOIN {SCHEMA}.serve_t s ON t.entity_id = s.entity_id
"""
r = run_sql(q)
cols = [c.name for c in r.manifest.schema.columns]
for c, v in zip(cols, r.result.data_array[0]):
    print(f"{c} = {v}")
