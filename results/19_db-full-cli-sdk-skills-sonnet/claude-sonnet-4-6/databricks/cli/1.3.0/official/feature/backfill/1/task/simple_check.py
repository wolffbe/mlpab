# Databricks notebook source
import inspect
from databricks.sdk.service.serving import CreateServingEndpoint

params = list(inspect.signature(CreateServingEndpoint.__init__).parameters.keys())
spark.createDataFrame([(p,) for p in params], ["param"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.cse_params")
