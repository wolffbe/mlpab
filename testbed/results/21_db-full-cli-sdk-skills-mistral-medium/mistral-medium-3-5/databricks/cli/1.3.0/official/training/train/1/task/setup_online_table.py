# Databricks notebook source
# MAGIC %python
# MAGIC # Try to create an online table or synced table for the predictions
# MAGIC # First, let's check if we can use the Lakebase synced tables
# MAGIC 
# MAGIC # Import necessary libraries
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.postgres import *
# MAGIC 
# MAGIC # Initialize the workspace client
# MAGIC w = WorkspaceClient()
# MAGIC 
# MAGIC # Try to create a synced table
# MAGIC try:
# MAGIC     # First, let's see if we can create a synced table using the SDK
# MAGIC     # We need to specify the project, branch, database, and table
# MAGIC     project_id = "mlpab155832"
# MAGIC     branch_id = "production"
# MAGIC     database_id = "mlpab155832"
# MAGIC     table_id = "predictionsd631cc"
# MAGIC     
# MAGIC     # Create the synced table
# MAGIC     synced_table = w.postgres.create_synced_table(
# MAGIC         parent=f"projects/{project_id}/branches/{branch_id}",
# MAGIC         synced_table_id=table_id,
# MAGIC         synced_table={
# MAGIC             "source_table_name": f"workspace.mlpab155832.{table_id}",
# MAGIC             "primary_key_columns": ["row_id"]
# MAGIC         }
# MAGIC     )
# MAGIC     print(f"Created synced table: {synced_table}")
# MAGIC except Exception as e:
# MAGIC     print(f"Error creating synced table: {e}")
# MAGIC     
# MAGIC # If that doesn't work, let's try to use SQL to create a feature table
# MAGIC try:
# MAGIC     # Create a feature table using SQL
# MAGIC     spark.sql(f"""
# MAGIC         CREATE FEATURE TABLE IF NOT EXISTS workspace.mlpab155832.predictionsd631cc 
# MAGIC         USING DELTA 
# MAGIC         LOCATION 's3://dbstorage-prod-orz5y/uc/0b7439f3-67e0-443e-a617-1ec1edc192c6/e4f39cd5-5a4c-42e0-a2a4-8f4d090d7109/__unitystorage/catalogs/f6b96a63-02b8-44e8-a3f0-71f9a6bb5601/tables/82ebb700-f75b-47ee-b9c7-4218dd8a2ec4'
# MAGIC         AS SELECT * FROM workspace.mlpab155832.predictionsd631cc
# MAGIC     """)
# MAGIC     print("Created feature table using SQL")
# MAGIC except Exception as e:
# MAGIC     print(f"Error creating feature table: {e}")
# MAGIC 
# MAGIC print("Setup completed")