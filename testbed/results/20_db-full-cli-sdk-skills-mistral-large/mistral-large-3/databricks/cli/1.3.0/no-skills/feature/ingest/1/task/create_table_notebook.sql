-- Create the table
CREATE TABLE IF NOT EXISTS workspace.mlpabd62957.transactions4adadd (
    row_id STRING,
    account_id STRING,
    event_time BIGINT,
    amount DOUBLE,
    category STRING
) USING DELTA;

-- Load data from the first export, ignoring duplicates
COPY INTO workspace.mlpabd62957.transactions4adadd
FROM (
    SELECT row_id, account_id, event_time, amount, category
    FROM 'dbfs:/Volumes/workspace/mlpabd62957/ingest_volume/transactions_export_1.csv'
)
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false')
COPY_OPTIONS ('mergeSchema' = 'true');

-- Load data from the second export, ignoring duplicates
COPY INTO workspace.mlpabd62957.transactions4adadd
FROM (
    SELECT row_id, account_id, event_time, amount, category
    FROM 'dbfs:/Volumes/workspace/mlpabd62957/ingest_volume/transactions_export_2.csv'
)
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false')
COPY_OPTIONS ('mergeSchema' = 'true');

-- Register the table as a feature table
CREATE FEATURE TABLE IF NOT EXISTS workspace.mlpabd62957.transactions4adadd
COMMENT "Feature table for transactions data"
AS SELECT * FROM workspace.mlpabd62957.transactions4adadd;

-- Enable online access for low-latency lookup
CREATE OR REFRESH ONLINE TABLE workspace.mlpabd62957.transactions4adadd
FROM FEATURES workspace.mlpabd62957.transactions4adadd;