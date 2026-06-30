"""Flaky job — this task ALWAYS fails (that is the point)."""
import sys

ERROR_CODE = "ERR-292576"

if __name__ == "__main__":
    print(f"flaky62cc43 starting (error code {ERROR_CODE})", file=sys.stderr)
    raise RuntimeError(
        f"seeded failure {ERROR_CODE}: upstream source unavailable — "
        "this job is EXPECTED to fail"
    )
