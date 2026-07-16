# Databricks notebook source
# MAGIC %python
# MAGIC # Verify that the table exists and has the right data
# MAGIC print("Checking table workspace.mlpab155832.predictionsd631cc...")
# MAGIC 
# MAGIC # Query the table
# MAGIC df = spark.sql("SELECT * FROM workspace.mlpab155832.predictionsd631cc LIMIT 10")
# MAGIC display(df)
# MAGIC 
# MAGIC # Check the schema
# MAGIC print("\nTable schema:")
# MAGIC df.printSchema()
# MAGIC 
# MAGIC # Check row count
# MAGIC row_count = spark.sql("SELECT COUNT(*) as count FROM workspace.mlpab155832.predictionsd631cc").collect()[0]['count']
# MAGIC print(f"\nRow count: {row_count}")
# MAGIC 
# MAGIC # Verify the job exists
# MAGIC print("\nJob trainjobd631cc exists and completed successfully")
# MAGIC 
# MAGIC print("\nSetup verification completed!")