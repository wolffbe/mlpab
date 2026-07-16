from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
schema = "workspace.mlpab60c44c"
WH = "8a93fc195da2ceb1"

def sql(stmt):
    r = w.statement_execution.execute_statement(
        statement=stmt, warehouse_id=WH, wait_timeout="50s")
    state = r.status.state.value if r.status.state else None
    print(stmt.strip().split("\n")[0][:90], "->", state)
    if r.status.error:
        raise RuntimeError(r.status.error.message)
    return r

# Build the feature table. is_weekend computed from epoch days (1970-01-01 was
# Thursday, so pmod(days,7): 2=Saturday, 3=Sunday) — independent of session TZ.
sql(f"""
CREATE OR REPLACE TABLE {schema}.features3bde51 AS
WITH tx AS (
  SELECT
    CAST(row_id AS STRING)     AS row_id,
    CAST(account_id AS STRING) AS account_id,
    CAST(event_time AS BIGINT) AS event_time,
    CAST(amount AS DOUBLE)     AS amount,
    CAST(currency AS STRING)   AS currency
  FROM read_files('/Volumes/workspace/mlpab60c44c/raw/transactions.csv',
                  format => 'csv', header => true)
),
fx AS (
  SELECT CAST(currency AS STRING) AS currency, CAST(fx_rate AS DOUBLE) AS fx_rate
  FROM read_files('/Volumes/workspace/mlpab60c44c/raw/fx_rates.csv',
                  format => 'csv', header => true)
)
SELECT
  tx.row_id,
  tx.account_id,
  tx.event_time,
  tx.amount * fx.fx_rate AS amount_usd,
  CASE WHEN pmod(FLOOR(tx.event_time / 86400000), 7) IN (2, 3) THEN 1 ELSE 0 END AS is_weekend,
  SUM(tx.amount) OVER (
    PARTITION BY tx.account_id
    ORDER BY tx.event_time
    RANGE BETWEEN 604800000 PRECEDING AND CURRENT ROW
  ) AS amount_7d
FROM tx
JOIN fx ON tx.currency = fx.currency
""")

sql(f"ALTER TABLE {schema}.features3bde51 ALTER COLUMN row_id SET NOT NULL")
sql(f"ALTER TABLE {schema}.features3bde51 ADD CONSTRAINT features3bde51_pk PRIMARY KEY (row_id)")

r = sql(f"SELECT COUNT(*) AS n, MIN(is_weekend), MAX(is_weekend) FROM {schema}.features3bde51")
print("counts:", r.result.data_array)
r = sql(f"SELECT * FROM {schema}.features3bde51 ORDER BY row_id LIMIT 5")
print("sample:")
for row in r.result.data_array:
    print(row)
