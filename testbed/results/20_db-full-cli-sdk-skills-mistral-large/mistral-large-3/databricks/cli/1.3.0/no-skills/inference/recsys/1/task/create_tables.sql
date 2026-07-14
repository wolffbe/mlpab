-- Create and load interactions
CREATE TABLE IF NOT EXISTS workspace.mlpab5a0819.interactions (
  user_id STRING,
  item_id STRING
);

COPY INTO workspace.mlpab5a0819.interactions
FROM "dbfs:/Volumes/workspace/mlpab5a0819/input_volume/interactions.csv"
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
COPY_OPTIONS ('mergeSchema' = 'true');

-- Create and load user_embeddings
CREATE TABLE IF NOT EXISTS workspace.mlpab5a0819.user_embeddings (
  user_id STRING,
  e1 DOUBLE, e2 DOUBLE, e3 DOUBLE, e4 DOUBLE, e5 DOUBLE, e6 DOUBLE, e7 DOUBLE, e8 DOUBLE
);

COPY INTO workspace.mlpab5a0819.user_embeddings
FROM "dbfs:/Volumes/workspace/mlpab5a0819/input_volume/user_embeddings.csv"
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
COPY_OPTIONS ('mergeSchema' = 'true');

-- Create and load item_embeddings
CREATE TABLE IF NOT EXISTS workspace.mlpab5a0819.item_embeddings (
  item_id STRING,
  e1 DOUBLE, e2 DOUBLE, e3 DOUBLE, e4 DOUBLE, e5 DOUBLE, e6 DOUBLE, e7 DOUBLE, e8 DOUBLE
);

COPY INTO workspace.mlpab5a0819.item_embeddings
FROM "dbfs:/Volumes/workspace/mlpab5a0819/input_volume/item_embeddings.csv"
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
COPY_OPTIONS ('mergeSchema' = 'true');