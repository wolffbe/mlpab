"""Capstone airquality — grader: hybrid metric (held-out RMSE) + on-platform
FTI artifacts, via `evals.capstone.common.grade_capstone`."""
from __future__ import annotations

import sys

from evals.capstone.common import grade_main


def main(argv=None) -> int:
    return grade_main("airquality", argv)


if __name__ == "__main__":
    sys.exit(main())
