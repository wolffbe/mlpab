#!/usr/bin/env python3
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
import json

wc = WorkspaceClient()

# Create a simple notebook
notebook_content = """# Databricks notebook source
# MAGIC %python

print("Hello from notebook")
"""

notebook_json = {
    "cells": [
        {
            "cell_type": "code",
            "metadata": {}, 
            "source": notebook_content
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

wc.workspace.upload('/Users/benedict@hopsworks.ai/test_notebook', json.dumps(notebook_json).encode('utf-8'), format=ImportFormat.JUPYTER)
print('Notebook uploaded successfully')
