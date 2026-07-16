# Databricks notebook source
# MAGIC %python
# MAGIC 
# MAGIC # Query the vector search index for all queries using REST API
# MAGIC import requests
# MAGIC import pandas as pd
# MAGIC import ast
# MAGIC import json
# MAGIC 
# MAGIC # Get token from environment
# MAGIC import os
# MAGIC token = os.environ.get('DATABRICKS_TOKEN', '***REDACTED***')
# MAGIC host = os.environ.get('DATABRICKS_HOST', '***REDACTED***')
# MAGIC 
# MAGIC # Read queries
# MAGIC queries_df = pd.read_csv('/Workspace/Users/benedict@hopsworks.ai/mlpabb9b3c4/data/queries.csv')
# MAGIC queries_df['embedding'] = queries_df['embedding'].apply(lambda x: ast.literal_eval(x))
# MAGIC 
# MAGIC # Query for each query vector
# MAGIC results = {}
# MAGIC index_name = 'workspace.mlpabb9b3c4.itemsffc8a7_vsi'
# MAGIC url = f'https://{host}/api/2.0/vector-search/indexes/{index_name}/query'
# MAGIC headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
# MAGIC 
# MAGIC for _, query_row in queries_df.iterrows():
# MAGIC   query_id = query_row['query_id']
# MAGIC   query_embedding = query_row['embedding']
# MAGIC   
# MAGIC   # Query the index
# MAGIC   payload = {
# MAGIC     'query_vector': query_embedding,
# MAGIC     'columns': ['item_id'],
# MAGIC     'num_results': 5
# MAGIC   }
# MAGIC   
# MAGIC   response = requests.post(url, headers=headers, json=payload)
# MAGIC   response.raise_for_status()
# MAGIC   data = response.json()
# MAGIC   
# MAGIC   # Extract item_ids from data_array
# MAGIC   item_ids = [row[0] for row in data['result']['data_array']]
# MAGIC   results[query_id] = item_ids
# MAGIC   print(f"Query {query_id}: {item_ids}")
# MAGIC 
# MAGIC # Write results
# MAGIC output = {
# MAGIC   "store": "workspace.mlpabb9b3c4.itemsffc8a7_vsi",
# MAGIC   "neighbors": results
# MAGIC }
# MAGIC 
# MAGIC # Write to workspace file
# MAGIC with open('/Workspace/Users/benedict@hopsworks.ai/mlpabb9b3c4/answers.json', 'w') as f:
# MAGIC   json.dump(output, f, indent=2)
# MAGIC 
# MAGIC print("Results written successfully")
