-- Create a Delta table for item embeddings
CREATE TABLE IF NOT EXISTS `$(echo $MLPAB_DATABRICKS_SCHEMA)`.items568016_table (
  item_id STRING,
  embedding ARRAY<FLOAT>,
  label STRING
) USING DELTA;

-- Load data from the local file into the table
COPY INTO `$(echo $MLPAB_DATABRICKS_SCHEMA)`.items568016_table
FROM (
  SELECT
    item_id,
    CAST(from_json(embedding, 'ARRAY<FLOAT>') AS ARRAY<FLOAT>) AS embedding,
    label
  FROM
    json.`dbfs:/tmp/$(echo $MLPAB_DATABRICKS_PREFIX)/items.csv`
)
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false');