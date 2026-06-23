"""Flaky job — this task ALWAYS fails (that is the point)."""
import sys

ERROR_CODE = "ERR-936362"

if __name__ == "__main__":
    print(f"flakye0f50e starting (error code {ERROR_CODE})", file=sys.stderr)
    raise RuntimeError(
        f"seeded failure {ERROR_CODE}: upstream source unavailable — "
        "this job is EXPECTED to fail"
    )
