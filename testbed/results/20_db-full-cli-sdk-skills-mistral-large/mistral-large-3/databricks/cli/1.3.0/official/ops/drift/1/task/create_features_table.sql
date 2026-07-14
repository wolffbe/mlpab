-- Databricks notebook source
CREATE TABLE IF NOT EXISTS ${MLPAB_DATABRICKS_SCHEMA}.${MLPAB_DATABRICKS_PREFIX}_features (
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
LOCATION 'dbfs:/Volumes/$(echo $MLPAB_DATABRICKS_SCHEMA | sed "s/\\./\\//g")/${MLPAB_DATABRICKS_PREFIX}_drift_volume/features.csv'
OPTIONS (header = 'true');