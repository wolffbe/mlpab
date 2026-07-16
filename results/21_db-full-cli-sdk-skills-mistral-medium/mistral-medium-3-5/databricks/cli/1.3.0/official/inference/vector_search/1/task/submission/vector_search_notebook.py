# Databricks notebook source
# MAGIC %python
# MAGIC 
# MAGIC # Step 1: Read the CSV files
# MAGIC import pandas as pd
# MAGIC import ast
# MAGIC 
# MAGIC # Read items.csv
# MAGIC items_df = pd.read_csv('/Workspace/Users/benedict@hopsworks.ai/mlpabb9b3c4/data/items.csv')
# MAGIC queries_df = pd.read_csv('/Workspace/Users/benedict@hopsworks.ai/mlpabb9b3c4/data/queries.csv')
# MAGIC 
# MAGIC # Convert embedding strings to arrays
# MAGIC items_df['embedding'] = items_df['embedding'].apply(lambda x: ast.literal_eval(x))
# MAGIC queries_df['embedding'] = queries_df['embedding'].apply(lambda x: ast.literal_eval(x))
# MAGIC 
# MAGIC # Step 2: Create Delta table for items
# MAGIC spark.createDataFrame(items_df).write.mode('overwrite').saveAsTable('workspace.mlpabb9b3c4.itemsffc8a7')
# MAGIC 
# MAGIC # Step 3: Create vector search index
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC vsc = VectorSearchClient()
# MAGIC 
# MAGIC # Create index
# MAGIC vs_index = vsc.create_delta_sync_index(
# MAGIC   name='workspace.mlpabb9b3c4.itemsffc8a7_vsi',
# MAGIC   source_table_name='workspace.mlpabb9b3c4.itemsffc8a7',
# MAGIC   pipeline_type='CONTINUOUS',
# MAGIC   primary_key='item_id',
# MAGIC   embedding_source_column='embedding',
# MAGIC   embedding_dimension=16,
# MAGIC   endpoint_name='mlpabb9b3c4_itemsffc8a7'
# MAGIC )
# MAGIC 
# MAGIC # Wait for index to be ready
# MAGIC import time
# MAGIC while True:
# MAGIC   index = vsc.get_index('workspace.mlpabb9b3c4.itemsffc8a7_vsi')
# MAGIC   if index.get('status', {}).get('ready', False):
# MAGIC     break
# MAGIC   time.sleep(5)
# MAGIC 
# MAGIC # Step 4: Query for each query vector
# MAGIC results = {}
# MAGIC for _, query_row in queries_df.iterrows():
# MAGIC   query_id = query_row['query_id']
# MAGIC   query_embedding = query_row['embedding']
# MAGIC   
# MAGIC   # Query the index
# MAGIC   query_result = vs_index.similarity_search(
# MAGIC     query_vector=query_embedding,
# MAGIC     columns=['item_id'],
# MAGIC     num_results=5
# MAGIC   )
# MAGIC   
# MAGIC   # Extract item_ids
# MAGIC   item_ids = [row['item_id'] for row in query_result]
# MAGIC   results[query_id] = item_ids
# MAGIC 
# MAGIC # Step 5: Write results to workspace
# MAGIC import json
# MAGIC output = {
# MAGIC   "store": "workspace.mlpabb9b3c4.itemsffc8a7_vsi",
# MAGIC   "neighbors": results
# MAGIC }
# MAGIC 
# MAGIC # Write to workspace file
# MAGIC with open('/Workspace/Users/benedict@hopsworks.ai/mlpabb9b3c4/answers.json', 'w') as f:
# MAGIC   json.dump(output, f)
