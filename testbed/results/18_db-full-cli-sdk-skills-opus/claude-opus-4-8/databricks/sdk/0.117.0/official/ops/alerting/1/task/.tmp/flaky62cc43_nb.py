# Databricks notebook source
# Runs the provided failing_job.py logic — ALWAYS fails (seeded error ERR-292576)
import sys

ERROR_CODE = "ERR-292576"
sys.stderr.write(f"flaky62cc43 starting (error code {ERROR_CODE})\n")
raise RuntimeError(
    f"seeded failure {ERROR_CODE}: upstream source unavailable — "
    "this job is EXPECTED to fail"
)
