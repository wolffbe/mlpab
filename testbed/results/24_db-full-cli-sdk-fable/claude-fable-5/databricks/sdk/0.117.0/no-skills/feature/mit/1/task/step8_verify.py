from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
schema = "workspace.mlpab60c44c"
WH = "8a93fc195da2ceb1"

def sql(stmt):
    r = w.statement_execution.execute_statement(
        statement=stmt, warehouse_id=WH, wait_timeout="50s")
    if r.status.error:
        raise RuntimeError(r.status.error.message)
    return r.result.data_array

print("columns:", [(c[0], c[1]) for c in sql(f"DESCRIBE TABLE {schema}.features3bde51") if c[0] and not c[0].startswith('#')])

# cross-check amount_7d against a correlated-subquery recomputation
bad = sql(f"""
WITH src AS (
  SELECT CAST(row_id AS STRING) row_id, CAST(account_id AS STRING) account_id,
         CAST(event_time AS BIGINT) event_time, CAST(amount AS DOUBLE) amount
  FROM read_files('/Volumes/workspace/mlpab60c44c/raw/transactions.csv', format=>'csv', header=>true)
),
expected AS (
  SELECT a.row_id,
         (SELECT SUM(b.amount) FROM src b
          WHERE b.account_id = a.account_id
            AND b.event_time BETWEEN a.event_time - 604800000 AND a.event_time) AS exp_7d
  FROM src a
)
SELECT COUNT(*) FROM {schema}.features3bde51 f JOIN expected e ON f.row_id = e.row_id
WHERE ABS(f.amount_7d - e.exp_7d) > 1e-9
""")
print("amount_7d mismatches:", bad)

print("row count:", sql(f"SELECT COUNT(*), COUNT(DISTINCT row_id) FROM {schema}.features3bde51"))
