-- Create a Delta table for item embeddings
CREATE TABLE IF NOT EXISTS `$(echo $MLPAB_DATABRICKS_SCHEMA)`.items568016_table (
  item_id STRING,
  embedding ARRAY<FLOAT>,
  label STRING
) USING DELTA;

-- Load data from the workspace file into the table
CREATE OR REPLACE TEMPORARY VIEW items_temp_view
USING CSV
OPTIONS (
  path "/Shared/$(echo $MLPAB_DATABRICKS_PREFIX)/items.csv",
  header "true",
  inferSchema "false"
);

INSERT INTO `$(echo $MLPAB_DATABRICKS_SCHEMA)`.items568016_table
SELECT
  item_id,
  CAST(from_json(embedding, 'ARRAY<FLOAT>') AS ARRAY<FLOAT>) AS embedding,
  label
FROM items_temp_view;