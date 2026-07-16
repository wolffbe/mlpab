#!/usr/bin/env python3
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
import json

wc = WorkspaceClient()

# Create a minimal valid Jupyter notebook
notebook_content = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Hello from minimal notebook')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

try:
    wc.workspace.upload('/Users/benedict@hopsworks.ai/mlpabf21a49/test_minimal', json.dumps(notebook_content).encode('utf-8'), format=ImportFormat.JUPYTER)
    print('Minimal Jupyter notebook upload successful')
except Exception as e:
    print(f'Minimal Jupyter notebook upload failed: {e}')
