#!/usr/bin/env python3
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
import json

wc = WorkspaceClient()

# Try to create a notebook with SOURCE format
notebook_content = {
    "cells": [
        {
            "cell_type": "code",
            "language": "python",
            "metadata": {},
            "results": {},
            "source": [
                "print('Hello from Databricks notebook')"
            ]
        }
    ],
    "metadata": {
        "createdAt": 1234567890,
        "language": "python",
        "notebookVersion": "databricks-notebook-source"
    },
    "name": "test_notebook",
    "nbformat": 4,
    "nbformat_minor": 4
}

try:
    wc.workspace.upload('/Users/benedict@hopsworks.ai/mlpabf21a49/test_source_format', json.dumps(notebook_content).encode('utf-8'), format=ImportFormat.SOURCE)
    print('SOURCE format upload successful')
except Exception as e:
    print(f'SOURCE format upload failed: {e}')
