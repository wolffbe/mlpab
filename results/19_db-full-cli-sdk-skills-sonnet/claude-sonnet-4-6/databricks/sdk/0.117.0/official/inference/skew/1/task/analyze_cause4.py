import os
import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
schema = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.default")
warehouses = list(w.warehouses.list())
warehouse_id = warehouses[0].id

def run_sql(sql):
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
    )
    state = resp.status.state.value
    stmt_id = resp.statement_id
    while state not in ["SUCCEEDED", "FAILED", "CANCELED", "CLOSED"]:
        time.sleep(2)
        resp = w.statement_execution.get_statement(stmt_id)
        state = resp.status.state.value
    if state != "SUCCEEDED":
        raise Exception(f"SQL failed: {resp.status.error}")
    return resp

def print_result(resp, title=""):
    if title:
        print(f"\n{title}:")
    if resp.result and resp.result.data_array:
        cols = [c.name for c in resp.manifest.schema.columns]
        print("  " + " | ".join(cols))
        for row in resp.result.data_array:
            print("  " + " | ".join(str(v)[:16] for v in row))

# Hypothesis: s = exp(t) - 1 = expm1(t)
# meaning training applied log1p(x) -> t, serving uses x -> s = exp(t) - 1
resp = run_sql(f"""
WITH joined AS (
    SELECT t.entity_id, t.f4 AS t, s.f4 AS s
    FROM {schema}.training_sample t
    JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
),
check_expm1 AS (
    SELECT
        entity_id,
        ROUND(t, 6) as t_f4,
        ROUND(s, 6) as s_f4,
        ROUND(EXP(t) - 1, 6) as expm1_t,
        ROUND(s - (EXP(t) - 1), 8) as diff_expm1
    FROM joined
)
SELECT * FROM check_expm1
ORDER BY entity_id
LIMIT 20
""")
print_result(resp, "Check hypothesis s = expm1(t_f4)")

# Statistical confirmation
resp2 = run_sql(f"""
WITH joined AS (
    SELECT t.f4 AS t, s.f4 AS s
    FROM {schema}.training_sample t
    JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
)
SELECT
    AVG(ABS(s - (EXP(t) - 1))) AS mean_abs_err_expm1,
    MAX(ABS(s - (EXP(t) - 1))) AS max_abs_err_expm1,
    STDDEV(s - (EXP(t) - 1)) AS std_err_expm1,
    AVG(ABS(s - t*t)) AS mean_abs_err_t2,
    COUNT(*) AS n
FROM joined
""")
print_result(resp2, "Error comparison: expm1(t) vs t^2")
