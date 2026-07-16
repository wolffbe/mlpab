-- Databricks SQL notebook to deduplicate and register the accounts feature table.

-- Create a temporary view for each batch.
CREATE OR REPLACE TEMPORARY VIEW batch_1 AS
SELECT * FROM csv.`/Volumes/workspace/mlpab58df8e/task_volume/batch_1.csv`;

CREATE OR REPLACE TEMPORARY VIEW batch_2 AS
SELECT * FROM csv.`/Volumes/workspace/mlpab58df8e/task_volume/batch_2.csv`;

CREATE OR REPLACE TEMPORARY VIEW batch_3 AS
SELECT * FROM csv.`/Volumes/workspace/mlpab58df8e/task_volume/batch_3.csv`;

-- Union all batches and deduplicate: keep the latest revision per row_id.
CREATE OR REPLACE TEMPORARY VIEW all_batches AS
SELECT * FROM batch_1
UNION ALL
SELECT * FROM batch_2
UNION ALL
SELECT * FROM batch_3;

CREATE OR REPLACE TEMPORARY VIEW latest_revisions AS
WITH ranked AS (
  SELECT 
    *,
    ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY updated_at DESC) as rn
  FROM all_batches
)
SELECT row_id, status, balance, updated_at
FROM ranked
WHERE rn = 1;

-- Create or replace the feature table in the target schema.
CREATE OR REPLACE TABLE workspace.mlpab58df8e.accounts7b3169 AS
SELECT * FROM latest_revisions;

-- Register the table as a feature table with record key and event-time column.
CREATE FEATURE TABLE IF NOT EXISTS workspace.mlpab58df8e.accounts7b3169
AS SELECT * FROM workspace.mlpab58df8e.accounts7b3169
WITH PRIMARY KEY (row_id)
WITH TIMESTAMP KEY (updated_at);