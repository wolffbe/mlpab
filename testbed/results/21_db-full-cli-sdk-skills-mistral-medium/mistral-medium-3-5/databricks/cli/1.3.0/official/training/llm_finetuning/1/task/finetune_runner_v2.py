# Databricks notebook source
# MAGIC %md {
# MAGIC   "cell_type": "code",
# MAGIC   "execution_count": null,
# MAGIC   "metadata": {},
# MAGIC   "outputs": [],
# MAGIC   "source": [
# MAGIC     "# Copy data files to /tmp\n",
# MAGIC     "import os\n",
# MAGIC     "os.makedirs('/tmp', exist_ok=True)\n",
# MAGIC     "\n",
# MAGIC     "# Use dbutils to copy files from workspace to /tmp\n",
# MAGIC     "# The files are in the same directory as this notebook\n",
# MAGIC     "# We need to use the workspace API or find another way\n",
# MAGIC     "\n",
# MAGIC     "# For now, let's try to use the workspace export API\n",
# MAGIC     "import base64\n",
# MAGIC     "import requests\n",
# MAGIC     "import json\n",
# MAGIC     "\n",
# MAGIC     "host = 'https://dbc-2a4591fe-28e4.cloud.databricks.com'\n",
# MAGIC     "token = ''  # Will use implicit auth\n",
# MAGIC     "workspace_path = '/Users/benedict@logicalclocks.com/mlpab64367b'\n",
# MAGIC     "\n",
# MAGIC     "files = ['base_model.npz', 'finetune.txt', 'eval.txt']\n",
# MAGIC     "for filename in files:\n",
# MAGIC     "    url = f'{host}/api/2.0/workspace/export'\n",
# MAGIC     "    params = {'path': f'{workspace_path}/{filename}', 'format': 'RAW'}\n",
# MAGIC     "    response = requests.get(url, params=params).json()\n",
# MAGIC     "    if 'content' in response:\n",
# MAGIC     "        content = base64.b64decode(response['content'])\n",
# MAGIC     "        with open(f'/tmp/{filename}', 'wb') as f:\n",
# MAGIC     "            f.write(content)\n",
# MAGIC     "        print(f'Downloaded {filename}')\n",
# MAGIC     "    else:\n",
# MAGIC     "        print(f'Failed: {response}')"
# MAGIC   ]
# MAGIC }
# MAGIC 
# MAGIC {
# MAGIC   "cell_type": "code",
# MAGIC   "execution_count": null,
# MAGIC   "metadata": {},
# MAGIC   "outputs": [],
# MAGIC   "source": [
# MAGIC     "# Run the finetune_model notebook\n",
# MAGIC     "%run /Users/benedict@logicalclocks.com/mlpab64367b/finetune_model.py"
# MAGIC   ]
# MAGIC }
