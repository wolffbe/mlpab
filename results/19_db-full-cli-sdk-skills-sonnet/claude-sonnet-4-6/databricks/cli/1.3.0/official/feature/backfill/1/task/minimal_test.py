# Databricks notebook source
spark.createDataFrame([("hello",)], ["msg"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.minimal_test")
