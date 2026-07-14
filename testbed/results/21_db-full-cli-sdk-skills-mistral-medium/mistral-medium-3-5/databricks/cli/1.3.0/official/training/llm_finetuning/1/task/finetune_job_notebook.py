# Databricks notebook source
# MAGIC %md {
# MAGIC   "cell_type": "code",
# MAGIC   "execution_count": null,
# MAGIC   "metadata": {},
# MAGIC   "outputs": [],
# MAGIC   "source": [
# MAGIC     "# Copy files from workspace to DBFS\n",
# MAGIC     "dbutils.fs.cp('workspace:/Users/benedict@logicalclocks.com/mlpab64367b/finetune_model.py', 'file:/tmp/finetune_model.py')\n",
# MAGIC     "dbutils.fs.cp('workspace:/Users/benedict@logicalclocks.com/mlpab64367b/base_model.npz', 'file:/tmp/base_model.npz')\n",
# MAGIC     "dbutils.fs.cp('workspace:/Users/benedict@logicalclocks.com/mlpab64367b/finetune.txt', 'file:/tmp/finetune.txt')\n",
# MAGIC     "dbutils.fs.cp('workspace:/Users/benedict@logicalclocks.com/mlpab64367b/eval.txt', 'file:/tmp/eval.txt')"
# MAGIC   ]
# MAGIC }
