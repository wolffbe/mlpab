-- Create source tables
CREATE TABLE IF NOT EXISTS workspace.mlpab5e2cc3.rawa9a7eb8 (
  row_id STRING,
  a_val DOUBLE
) USING CSV
OPTIONS (header = 'true', inferSchema = 'false');

-- Load data into rawa9a7eb8
COPY INTO workspace.mlpab5e2cc3.rawa9a7eb8
FROM 'dbfs:/Volumes/workspace/mlpab5e2cc3/staging/raw_a.csv'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false');

-- Create rawb9a7eb8
CREATE TABLE IF NOT EXISTS workspace.mlpab5e2cc3.rawb9a7eb8 (
  row_id STRING,
  b_val DOUBLE
) USING CSV
OPTIONS (header = 'true', inferSchema = 'false');

-- Load data into rawb9a7eb8
COPY INTO workspace.mlpab5e2cc3.rawb9a7eb8
FROM 'dbfs:/Volumes/workspace/mlpab5e2cc3/staging/raw_b.csv'
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false');

-- Create derived table
CREATE OR REPLACE TABLE workspace.mlpab5e2cc3.derived9a7eb8 AS
SELECT
  a.row_id,
  ROUND(a.a_val + b.b_val, 6) AS col_sum
FROM workspace.mlpab5e2cc3.rawa9a7eb8 a
JOIN workspace.mlpab5e2cc3.rawb9a7eb8 b
ON a.row_id = b.row_id;

-- Enable online table for derived9a7eb8
CREATE OR REFRESH ONLINE TABLE workspace.mlpab5e2cc3.derived9a7eb8;