# Databricks notebook source
# MAGIC %python
# MAGIC # Create a synced table for low-latency access
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.postgres import SyncedTable, SyncedTableSource
# MAGIC 
# MAGIC # Initialize the workspace client
# MAGIC w = WorkspaceClient()
# MAGIC 
# MAGIC # Create a synced table
# MAGIC try:
# MAGIC     synced_table = w.postgres.create_synced_table(
# MAGIC         parent="projects/mlpab155832/branches/production",
# MAGIC         synced_table_id="predictionsd631cc",
# MAGIC         synced_table={
# MAGIC             "source_table_name": "workspace.mlpab155832.predictionsd631cc",
# MAGIC             "primary_key_columns": ["row_id"]
# MAGIC         }
# MAGIC     )
# MAGIC     print(f"Created synced table: {synced_table}")
# MAGIC except Exception as e:
# MAGIC     print(f"Error creating synced table: {e}")
# MAGIC     
# MAGIC     # Try alternative approach
# MAGIC     try:
# MAGIC         # Maybe we need to create a database first
# MAGIC         database = w.postgres.create_database(
# MAGIC             parent="projects/mlpab155832/branches/production",
# MAGIC             database_id="mlpab155832"
# MAGIC         )
# MAGIC         print(f"Created database: {database}")
# MAGIC         
# MAGIC         # Now create the synced table
# MAGIC         synced_table = w.postgres.create_synced_table(
# MAGIC             parent="projects/mlpab155832/branches/production",
# MAGIC             synced_table_id="predictionsd631cc",
# MAGIC             synced_table={
# MAGIC                 "source_table_name": "workspace.mlpab155832.predictionsd631cc",
# MAGIC                 "primary_key_columns": ["row_id"]
# MAGIC             }
# MAGIC         )
# MAGIC         print(f"Created synced table: {synced_table}")
# MAGIC     except Exception as e2:
# MAGIC         print(f"Error in alternative approach: {e2}")
# MAGIC 
# MAGIC print("Synced table creation completed")