-- Daily incremental ingestion of new event increments into the feature table.
-- COPY INTO is idempotent: it tracks already-loaded files and only ingests
-- newly-arrived increment files from the source volume.
COPY INTO workspace.mlpab2b7985.incrementalb48074
FROM (
  SELECT
    CAST(row_id AS STRING)     AS row_id,
    CAST(account_id AS STRING) AS account_id,
    CAST(event_time AS BIGINT) AS event_time,
    CAST(amount AS DOUBLE)     AS amount,
    CAST(category AS STRING)   AS category
  FROM '/Volumes/workspace/mlpab2b7985/incr_raw/'
)
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false')
COPY_OPTIONS ('mergeSchema' = 'false');
