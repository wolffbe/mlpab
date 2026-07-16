# Databricks notebook source
# MAGIC %sql
# MAGIC -- Create a Delta table for item embeddings
# MAGIC CREATE TABLE IF NOT EXISTS `$(echo $MLPAB_DATABRICKS_SCHEMA)`.items568016_table (
# MAGIC   item_id STRING,
# MAGIC   embedding ARRAY<FLOAT>,
# MAGIC   label STRING
# MAGIC ) USING DELTA;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Load data from the workspace file into the table
# MAGIC CREATE OR REPLACE TEMPORARY VIEW items_temp_view
# MAGIC USING CSV
# MAGIC OPTIONS (
# MAGIC   path "/Shared/$(echo $MLPAB_DATABRICKS_PREFIX)/items.csv",
# MAGIC   header "true",
# MAGIC   inferSchema "false"
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO `$(echo $MLPAB_DATABRICKS_SCHEMA)`.items568016_table
# MAGIC SELECT
# MAGIC   item_id,
# MAGIC   CAST(from_json(embedding, 'ARRAY<FLOAT>') AS ARRAY<FLOAT>) AS embedding,
# MAGIC   label
# MAGIC FROM items_temp_view;