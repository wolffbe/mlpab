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


# Look at sample of matched f3 values and candidate transforms
q = f"""
SELECT t.entity_id, t.f3 AS train_f3, s.f3 AS serve_f3,
       s.f3 - t.f3 AS diff,
       s.f3 / t.f3 AS ratio,
       ln(t.f3) AS ln_train,
       exp(t.f3) AS exp_train
FROM {SCHEMA}.train_t t
JOIN {SCHEMA}.serve_t s ON t.entity_id = s.entity_id
ORDER BY t.f3
LIMIT 15
"""
r = run_sql(q)
cols = [c.name for c in r.manifest.schema.columns]
print("\t".join(cols))
for row in r.result.data_array:
    print("\t".join(str(x) for x in row))

# Test specific hypotheses across all rows: is serve_f3 ~ train_f3^2, or *some factor, or train_f3 + f-of-other?
# correlations of serve_f3 with transforms of train_f3
q2 = f"""
SELECT
  corr(s.f3, t.f3) AS c_identity,
  corr(s.f3, t.f3*t.f3) AS c_square,
  corr(s.f3, exp(t.f3)) AS c_exp,
  corr(s.f3, ln(t.f3)) AS c_ln,
  corr(s.f3, sqrt(t.f3)) AS c_sqrt,
  avg(s.f3 / (t.f3*t.f3)) AS avg_ratio_square,
  avg(s.f3 - t.f3*t.f3) AS avg_diff_square,
  stddev(s.f3 - t.f3*t.f3) AS std_diff_square
FROM {SCHEMA}.train_t t
JOIN {SCHEMA}.serve_t s ON t.entity_id = s.entity_id
"""
r2 = run_sql(q2)
cols2 = [c.name for c in r2.manifest.schema.columns]
print("\n=== transform fit ===")
for c, v in zip(cols2, r2.result.data_array[0]):
    print(f"{c} = {v}")
