# Databricks notebook source
# MAGIC %python
# MAGIC 
# MAGIC # Query the vector search index for all queries
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC import pandas as pd
# MAGIC import ast
# MAGIC import json
# MAGIC 
# MAGIC # Initialize client
# MAGIC vsc = VectorSearchClient()
# MAGIC 
# MAGIC # Get the index
# MAGIC index = vsc.get_index('workspace.mlpabb9b3c4.itemsffc8a7_vsi')
# MAGIC 
# MAGIC # Read queries
# MAGIC queries_df = pd.read_csv('/Workspace/Users/benedict@hopsworks.ai/mlpabb9b3c4/data/queries.csv')
# MAGIC queries_df['embedding'] = queries_df['embedding'].apply(lambda x: ast.literal_eval(x))
# MAGIC 
# MAGIC # Query for each query vector
# MAGIC results = {}
# MAGIC for _, query_row in queries_df.iterrows():
# MAGIC   query_id = query_row['query_id']
# MAGIC   query_embedding = query_row['embedding']
# MAGIC   
# MAGIC   # Query the index
# MAGIC   query_result = index.similarity_search(
# MAGIC     query_vector=query_embedding,
# MAGIC     columns=['item_id'],
# MAGIC     num_results=5
# MAGIC   )
# MAGIC   
# MAGIC   # Extract item_ids
# MAGIC   item_ids = [row['item_id'] for row in query_result]
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
