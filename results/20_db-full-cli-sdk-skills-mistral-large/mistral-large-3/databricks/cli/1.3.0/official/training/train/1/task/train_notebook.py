# Databricks notebook source
# MAGIC %sh
# MAGIC cp "/Workspace/Users/benedict@logicalclocks.com/mlpaba53a68/train.csv" "./train.csv"
# MAGIC cp "/Workspace/Users/benedict@logicalclocks.com/mlpaba53a68/score.csv" "./score.csv"

# COMMAND ----------

import train_model

# COMMAND ----------

# Display the output file
import pandas as pd
df = pd.read_csv("predictions.csv")
display(df)