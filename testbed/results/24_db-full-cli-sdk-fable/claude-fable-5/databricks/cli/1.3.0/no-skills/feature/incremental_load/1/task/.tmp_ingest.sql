MERGE INTO workspace.mlpab679845.incrementalf3c1bf AS t
USING (
  SELECT row_id, account_id, event_time, amount, category
  FROM read_files(
    '/Volumes/workspace/mlpab679845/raw/increment_*.csv',
    format => 'csv',
    header => true,
    schema => 'row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING'
  )
) AS s
ON t.row_id = s.row_id AND t.event_time = s.event_time
WHEN NOT MATCHED THEN INSERT (row_id, account_id, event_time, amount, category)
VALUES (s.row_id, s.account_id, s.event_time, s.amount, s.category);
