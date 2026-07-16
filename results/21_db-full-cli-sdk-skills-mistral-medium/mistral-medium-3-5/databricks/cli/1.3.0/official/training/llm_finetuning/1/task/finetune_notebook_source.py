# Databricks notebook source
# This is a Databricks notebook that will copy files and run the fine-tuning script

# Cell 1: Copy files from workspace to /tmp
import os
os.makedirs('/tmp', exist_ok=True)

dbutils.fs.cp('workspace:/Users/benedict@logicalclocks.com/mlpab64367b/finetune_model.py', 'file:/tmp/finetune_model.py')
dbutils.fs.cp('workspace:/Users/benedict@logicalclocks.com/mlpab64367b/base_model.npz', 'file:/tmp/base_model.npz')
dbutils.fs.cp('workspace:/Users/benedict@logicalclocks.com/mlpab64367b/finetune.txt', 'file:/tmp/finetune.txt')
dbutils.fs.cp('workspace:/Users/benedict@logicalclocks.com/mlpab64367b/eval.txt', 'file:/tmp/eval.txt')

# Cell 2: Run the fine-tuning script
os.chdir('/tmp')
import finetune_model
finetune_model.main()
