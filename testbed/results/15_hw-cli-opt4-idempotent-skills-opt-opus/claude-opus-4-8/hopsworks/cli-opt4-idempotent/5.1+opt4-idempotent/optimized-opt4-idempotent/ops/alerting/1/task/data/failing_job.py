"""Flaky job — this task ALWAYS fails (that is the point)."""
import sys

ERROR_CODE = "ERR-583654"

if __name__ == "__main__":
    print(f"flakyfc19f7 starting (error code {ERROR_CODE})", file=sys.stderr)
    raise RuntimeError(
        f"seeded failure {ERROR_CODE}: upstream source unavailable — "
        "this job is EXPECTED to fail"
    )
