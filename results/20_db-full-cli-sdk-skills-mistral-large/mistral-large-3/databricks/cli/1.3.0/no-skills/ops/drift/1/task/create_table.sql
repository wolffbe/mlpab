-- Create a table in Unity Catalog for feature data
CREATE TABLE IF NOT EXISTS workspace.mlpabf81e30.mlpabf81e30_features (
    entity_id STRING,
    event_time TIMESTAMP,
    f1 DOUBLE,
    f2 DOUBLE,
    f3 DOUBLE,
    f4 DOUBLE,
    f5 DOUBLE,
    f6 DOUBLE
)
USING CSV
OPTIONS (
    header = "true",
    inferSchema = "false",
    path = "dbfs:/Volumes/workspace/mlpabf81e30/mlpabf81e30_volume/features.csv"
);