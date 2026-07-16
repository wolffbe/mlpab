# Databricks notebook source
import os

token = os.environ.get('DATABRICKS_TOKEN', 'NOT SET')
host = os.environ.get('DATABRICKS_HOST', 'NOT SET')

print(f"DATABRICKS_TOKEN: {token[:10]}..." if len(token) > 10 else f"DATABRICKS_TOKEN: {token}")
print(f"DATABRICKS_HOST: {host}")

# Try to list environment variables
print(f"Environment variables: {sorted(os.environ.keys())}")
