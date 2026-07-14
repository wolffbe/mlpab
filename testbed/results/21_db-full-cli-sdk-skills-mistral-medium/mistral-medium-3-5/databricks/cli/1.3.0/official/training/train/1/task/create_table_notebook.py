# Databricks notebook source
# MAGIC %python
# MAGIC # First, let's read the predictions.csv file from workspace
# MAGIC import pandas as pd
# MAGIC 
# MAGIC # Read the predictions.csv file
# MAGIC predictions_df = pd.read_csv("/Workspace/Users/benedict@logicalclocks.com/mlpab155832/data/predictions.csv")
# MAGIC print("Predictions data:")
# MAGIC print(predictions_df.head())
# MAGIC print(f"Shape: {predictions_df.shape}")
# MAGIC 
# MAGIC # Create a Spark DataFrame and save as Delta table
# MAGIC spark_df = spark.createDataFrame(predictions_df)
# MAGIC spark_df.write.mode("overwrite").saveAsTable("workspace.mlpab155832.predictionsd631cc")
# MAGIC 
# MAGIC # Verify the table was created
# MAGIC display(spark.sql("SELECT * FROM workspace.mlpab155832.predictionsd631cc LIMIT 10"))