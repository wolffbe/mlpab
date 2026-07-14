# Databricks notebook source
# MAGIC %md {
# MAGIC   "cell_type": "code",
# MAGIC   "execution_count": null,
# MAGIC   "metadata": {},
# MAGIC   "outputs": [],
# MAGIC   "source": [
# MAGIC     "# Copy files from workspace to local /tmp directory\n",
# MAGIC     "import os\n",
# MAGIC     "import shutil\n",
# MAGIC     "\n",
# MAGIC     "# Create /tmp directory if it doesn't exist\n",
# MAGIC     "os.makedirs('/tmp', exist_ok=True)\n",
# MAGIC     "\n",
# MAGIC     "# Copy files using dbutils\n",
# MAGIC     "dbutils.fs.cp('workspace:/Users/benedict@logicalclocks.com/mlpab64367b/finetune_model.py', 'file:/tmp/')\n",
# MAGIC     "dbutils.fs.cp('workspace:/Users/benedict@logicalclocks.com/mlpab64367b/base_model.npz', 'file:/tmp/')\n",
# MAGIC     "dbutils.fs.cp('workspace:/Users/benedict@logicalclocks.com/mlpab64367b/finetune.txt', 'file:/tmp/')\n",
# MAGIC     "dbutils.fs.cp('workspace:/Users/benedict@logicalclocks.com/mlpab64367b/eval.txt', 'file:/tmp/')"
# MAGIC   ]
# MAGIC }
# MAGIC 
# MAGIC {
# MAGIC   "cell_type": "code",
# MAGIC   "execution_count": null,
# MAGIC   "metadata": {},
# MAGIC   "outputs": [],
# MAGIC   "source": [
# MAGIC     "# Change to /tmp directory and run the script\n",
# MAGIC     "os.chdir('/tmp')\n",
# MAGIC     "import finetune_model\n",
# MAGIC     "finetune_model.main()"
# MAGIC   ]
# MAGIC }
