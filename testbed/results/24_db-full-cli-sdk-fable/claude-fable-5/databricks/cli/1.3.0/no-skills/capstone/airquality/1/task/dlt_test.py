# Databricks notebook source
import dlt
from pyspark.sql import functions as F

@dlt.table(name="dlt_probe")
def dlt_probe():
    return spark.range(1).withColumn("ok", F.lit(1))
