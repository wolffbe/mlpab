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
            print("  " + " | ".join(str(v)[:14] for v in row))

# Get log-log slope (power law exponent)
resp = run_sql(f"""
WITH joined AS (
    SELECT LN(t.f4) AS lt, LN(s.f4) AS ls
    FROM {schema}.training_sample t
    JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
)
SELECT
    REGR_SLOPE(ls, lt) AS power_exponent,
    REGR_INTERCEPT(ls, lt) AS log_scale_factor,
    REGR_R2(ls, lt) AS r2_log_log,
    EXP(REGR_INTERCEPT(ls, lt)) AS scale_factor,
    AVG(ls - lt) AS avg_log_ratio,
    STDDEV(ls - lt) AS std_log_ratio
FROM joined
""")
print_result(resp, "Log-log regression (power law fit: s = A * t^alpha)")

# Check if s = t^2 exactly or approximately
# Look at std of ratio s/t^alpha for alpha=2
resp2 = run_sql(f"""
WITH joined AS (
    SELECT t.f4 AS t, s.f4 AS s
    FROM {schema}.training_sample t
    JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
),
powered AS (
    SELECT
        t, s,
        s / (t*t) AS ratio_t2,
        s / (t*t*t) AS ratio_t3,
        s / (t*t*SQRT(t)) AS ratio_t25,
        ABS(s - t*t) as err_t2,
        ABS(s - t*t*t) as err_t3,
        ABS(s - t*t*SQRT(t)) as err_t25
    FROM joined
)
SELECT
    AVG(ratio_t2) as avg_s_over_t2,
    STDDEV(ratio_t2) as std_s_over_t2,
    AVG(ratio_t3) as avg_s_over_t3,
    STDDEV(ratio_t3) as std_s_over_t3,
    AVG(ratio_t25) as avg_s_over_t25,
    STDDEV(ratio_t25) as std_s_over_t25,
    AVG(err_t2) as mean_abs_err_t2,
    AVG(err_t3) as mean_abs_err_t3,
    AVG(err_t25) as mean_abs_err_t25
FROM powered
""")
print_result(resp2, "Check power law s ~ t^alpha (which alpha minimizes error?)")

# Sample: show actual values to see if there's a clean pattern
resp3 = run_sql(f"""
WITH joined AS (
    SELECT t.entity_id, t.f4 AS t, s.f4 AS s
    FROM {schema}.training_sample t
    JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
)
SELECT
    entity_id,
    ROUND(t, 4) as t_f4,
    ROUND(s, 4) as s_f4,
    ROUND(s/t, 4) as ratio,
    ROUND(s - t*t, 4) as err_t2,
    ROUND(LN(s/t), 4) as log_ratio
FROM joined
ORDER BY t
LIMIT 10
""")
print_result(resp3, "Sample sorted by t_f4 (smallest)")

resp4 = run_sql(f"""
WITH joined AS (
    SELECT t.entity_id, t.f4 AS t, s.f4 AS s
    FROM {schema}.training_sample t
    JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
)
SELECT
    entity_id,
    ROUND(t, 4) as t_f4,
    ROUND(s, 4) as s_f4,
    ROUND(s/t, 4) as ratio,
    ROUND(s - t*t, 4) as err_t2,
    ROUND(LN(s/t), 4) as log_ratio
FROM joined
ORDER BY t DESC
LIMIT 10
""")
print_result(resp4, "Sample sorted by t_f4 (largest)")
