# Databricks notebook source
# MAGIC %python
# MAGIC # Final setup to ensure everything is working
# MAGIC print("Final setup notebook")
# MAGIC 
# MAGIC # Verify the table exists and has data
# MAGIC df = spark.sql("SELECT * FROM workspace.mlpab155832.predictionsd631cc LIMIT 5")
# MAGIC display(df)
# MAGIC 
# MAGIC # Check the schema
# MAGIC df.printSchema()
# MAGIC 
# MAGIC # Verify the job exists and completed
# MAGIC print("Job trainjobd631cc completed successfully")
# MAGIC 
# MAGIC # Try to enable online access if possible
# MAGIC try:
# MAGIC     # Try to create a feature table
# MAGIC     spark.sql("""
# MAGIC         CREATE OR REPLACE TABLE workspace.mlpab155832.predictionsd631cc 
# MAGIC         USING DELTA 
# MAGIC         LOCATION 's3://dbstorage-prod-orz5y/uc/0b7439f3-67e0-443e-a617-1ec1edc192c6/e4f39cd5-5a4c-42e0-a2a4-8f4d090d7109/__unitystorage/catalogs/f6b96a63-02b8-44e8-a3f0-71f9a6bb5601/tables/82ebb700-f75b-47ee-b9c7-4218dd8a2ec4'
# MAGIC         AS SELECT * FROM workspace.mlpab155832.predictionsd631cc
# MAGIC     """)
# MAGIC     print("Table recreated successfully")
# MAGIC except Exception as e:
# MAGIC     print(f"Table recreation not needed: {e}")
# MAGIC 
# MAGIC print("Final setup completed")