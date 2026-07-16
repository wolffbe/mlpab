# Databricks notebook source
import dlt
from pyspark.sql.functions import col

@dlt.table(
  name="workspace.mlpabbc4768.prediction_log",
  comment="Table created from prediction_log.csv"
)
def create_prediction_log():
    return spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/Volumes/workspace/mlpabbc4768/prediction_volume/prediction_log.csv")