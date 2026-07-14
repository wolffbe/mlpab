-- Create the table
CREATE TABLE IF NOT EXISTS ${MLPAB_DATABRICKS_SCHEMA}.incrementala59b19 (
  row_id STRING,
  account_id STRING,
  event_time BIGINT,
  amount DOUBLE,
  category STRING
);

-- Load all increment files
COPY INTO ${MLPAB_DATABRICKS_SCHEMA}.incrementala59b19
FROM "dbfs:/Volumes/workspace/${MLPAB_DATABRICKS_PREFIX}/incremental_volume"
FILEFORMAT = CSV
PATTERN = 'increment_*.csv'
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true');