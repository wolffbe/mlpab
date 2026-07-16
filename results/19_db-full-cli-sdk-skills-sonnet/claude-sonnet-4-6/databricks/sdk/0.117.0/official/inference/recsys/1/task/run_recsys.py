import os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState, ExecuteStatementRequestOnWaitTimeout

w = WorkspaceClient()
warehouse_id = '4dfab06c923fe3cc'

def run_sql(sql, description=""):
    print(f"Running: {description or sql[:80]}")
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout='50s',
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE
    )
    stmt_id = resp.statement_id
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL FAILED ({resp.status.state}): {resp.status.error}")
    print(f"  OK")
    return resp

# Step 1: Load CSVs as Delta tables
run_sql("""
CREATE OR REPLACE TABLE workspace.mlpab3f17ca.interactions_raw AS
SELECT * FROM read_files(
  '/Volumes/workspace/mlpab3f17ca/recsys_data/interactions.csv',
  format => 'csv', header => true
)
""", "Load interactions")

run_sql("""
CREATE OR REPLACE TABLE workspace.mlpab3f17ca.user_emb_raw AS
SELECT * FROM read_files(
  '/Volumes/workspace/mlpab3f17ca/recsys_data/user_embeddings.csv',
  format => 'csv', header => true
)
""", "Load user embeddings")

run_sql("""
CREATE OR REPLACE TABLE workspace.mlpab3f17ca.item_emb_raw AS
SELECT * FROM read_files(
  '/Volumes/workspace/mlpab3f17ca/recsys_data/item_embeddings.csv',
  format => 'csv', header => true
)
""", "Load item embeddings")

# Step 2: Compute dot products for all user-item pairs, exclude interactions, rank top-5
run_sql("""
CREATE OR REPLACE TABLE workspace.mlpab3f17ca.recs_computed AS
WITH scores AS (
  SELECT
    u.user_id,
    i.item_id,
    (CAST(u.e1 AS DOUBLE) * CAST(i.e1 AS DOUBLE) +
     CAST(u.e2 AS DOUBLE) * CAST(i.e2 AS DOUBLE) +
     CAST(u.e3 AS DOUBLE) * CAST(i.e3 AS DOUBLE) +
     CAST(u.e4 AS DOUBLE) * CAST(i.e4 AS DOUBLE) +
     CAST(u.e5 AS DOUBLE) * CAST(i.e5 AS DOUBLE) +
     CAST(u.e6 AS DOUBLE) * CAST(i.e6 AS DOUBLE) +
     CAST(u.e7 AS DOUBLE) * CAST(i.e7 AS DOUBLE) +
     CAST(u.e8 AS DOUBLE) * CAST(i.e8 AS DOUBLE)) AS score
  FROM workspace.mlpab3f17ca.user_emb_raw u
  CROSS JOIN workspace.mlpab3f17ca.item_emb_raw i
),
excluded AS (
  SELECT user_id, item_id FROM workspace.mlpab3f17ca.interactions_raw
),
filtered AS (
  SELECT s.*
  FROM scores s
  LEFT ANTI JOIN excluded e ON s.user_id = e.user_id AND s.item_id = e.item_id
),
ranked AS (
  SELECT
    user_id,
    item_id,
    score,
    ROW_NUMBER() OVER (
      PARTITION BY user_id
      ORDER BY score DESC, item_id ASC
    ) AS rank
  FROM filtered
)
SELECT user_id, item_id, rank
FROM ranked
WHERE rank <= 5
""", "Compute top-5 recs with dot product scores")

# Step 3: Create the feature table recs708df6
run_sql("""
CREATE OR REPLACE TABLE workspace.mlpab3f17ca.recs708df6 (
  rec_id STRING NOT NULL,
  user_id STRING,
  rank INT,
  item_id STRING
) TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""", "Create feature table recs708df6")

run_sql("""
INSERT INTO workspace.mlpab3f17ca.recs708df6
SELECT
  CONCAT(user_id, '#', CAST(rank AS STRING)) AS rec_id,
  user_id,
  rank,
  item_id
FROM workspace.mlpab3f17ca.recs_computed
ORDER BY user_id, rank
""", "Insert recommendations into feature table")

# Verify row count
resp = run_sql("""
SELECT COUNT(*) AS cnt, COUNT(DISTINCT user_id) AS users FROM workspace.mlpab3f17ca.recs708df6
""", "Count rows")
if resp.result and resp.result.data_array:
    row = resp.result.data_array[0]
    print(f"  Rows: {row[0]}, Users: {row[1]}")

# Sample
resp2 = run_sql("""
SELECT * FROM workspace.mlpab3f17ca.recs708df6 ORDER BY user_id, rank LIMIT 10
""", "Sample rows")
if resp2.result and resp2.result.data_array:
    for row in resp2.result.data_array:
        print(f"  {row}")

print("Feature table created successfully!")
