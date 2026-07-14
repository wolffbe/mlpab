-- Databricks SQL Notebook to ingest transactions data
-- This notebook creates a Delta table from the CSV files and deduplicates by `row_id`.

-- Create a temporary view for each CSV file
CREATE OR REPLACE TEMP VIEW transactions_export_1 AS
SELECT * FROM csv.`/tmp/mlpaba52a53/transactions_export_1.csv`
WITH (
  header = "true",
  inferSchema = "true"
);

CREATE OR REPLACE TEMP VIEW transactions_export_2 AS
SELECT * FROM csv.`/tmp/mlpaba52a53/transactions_export_2.csv`
WITH (
  header = "true",
  inferSchema = "true"
);

-- Union and deduplicate by row_id
CREATE OR REPLACE TEMP VIEW combined_transactions AS
SELECT * FROM transactions_export_1
UNION ALL
SELECT * FROM transactions_export_2;

-- Create or replace the Delta table
CREATE OR REPLACE TABLE workspace.mlpaba52a53.transactions4adadd AS
SELECT * FROM combined_transactions
QUALIFY ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY event_time) = 1;

-- Set the record key and event-time column properties
ALTER TABLE workspace.mlpaba52a53.transactions4adadd
SET TBLPROPERTIES (
  'delta.feature.recordKey' = 'row_id',
  'delta.feature.eventTimeColumn' = 'event_time'
);