# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE TABLE workspace.mlpab48540d.items568016_table AS
# MAGIC SELECT item_id, CAST(from_json(embedding, 'ARRAY<FLOAT>') AS ARRAY<FLOAT>) AS embedding, label
# MAGIC FROM csv."/Volumes/workspace/mlpab48540d/items568016_volume/items.csv";