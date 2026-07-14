"""Runs ON the Hopsworks cluster as a Python job: point-in-time batch scoring.

Feature history (data/feature_history.csv) and model (data/model.json) are
embedded below since the job container has no project filesystem mount.
"""

import io
import math

import pandas as pd

import hopsworks

T = 1773565200000
WEIGHTS = {"f1": -0.1306, "f2": 0.0121, "f3": -0.7418}
BIAS = -0.8397

CSV_DATA = """\
