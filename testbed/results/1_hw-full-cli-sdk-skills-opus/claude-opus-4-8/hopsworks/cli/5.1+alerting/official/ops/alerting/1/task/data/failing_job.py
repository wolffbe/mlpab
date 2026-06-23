"""Flaky job — this task ALWAYS fails (that is the point)."""
import sys

ERROR_CODE = "ERR-639672"

if __name__ == "__main__":
    print(f"flakya9c625 starting (error code {ERROR_CODE})", file=sys.stderr)
    raise RuntimeError(
        f"seeded failure {ERROR_CODE}: upstream source unavailable — "
        "this job is EXPECTED to fail"
    )
