CREATE TABLE ${MLPAB_DATABRICKS_SCHEMA}.${MLPAB_DATABRICKS_PREFIX}_training_data
USING CSV
OPTIONS (
  path 'dbfs:/Volumes/workspace/${MLPAB_DATABRICKS_SCHEMA#*.}/${MLPAB_DATABRICKS_PREFIX}_volume/training_data.csv',
  header 'true',
  inferSchema 'true'
);