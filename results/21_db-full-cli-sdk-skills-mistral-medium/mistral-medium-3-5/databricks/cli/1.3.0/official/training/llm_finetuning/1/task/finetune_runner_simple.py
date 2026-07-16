# Databricks notebook source
# MAGIC %md {
# MAGIC   "cell_type": "code",
# MAGIC   "execution_count": null,
# MAGIC   "metadata": {},
# MAGIC   "outputs": [],
# MAGIC   "source": [
# MAGIC     "# Copy data files to current directory\n",
# MAGIC     "import os\n",
# MAGIC     "import shutil\n",
# MAGIC     "\n",
# MAGIC     "# Get the current notebook's directory\n",
# MAGIC     "current_dir = os.path.dirname(os.path.abspath(''))\n",
# MAGIC     "print(f'Current directory: {current_dir}')\n",
# MAGIC     "\n",
# MAGIC     "# Copy files from workspace\n",
# MAGIC     "# The files are in /Users/benedict@logicalclocks.com/mlpab64367b/\n",
# MAGIC     "# We need to copy them to the current directory\n",
# MAGIC     "\n",
# MAGIC     "# For now, let's just try to run the script directly\n",
# MAGIC     "%run /Users/benedict@logicalclocks.com/mlpab64367b/finetune_model.py"
# MAGIC   ]
# MAGIC }
