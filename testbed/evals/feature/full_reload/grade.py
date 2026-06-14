"""Full-reload task — grader: the standard table suite (columns/rows/content)
over feature table `customers` VERSION 2 read back through the platform (or
--csv locally). Stale rows or old column names fail with named diagnoses."""

from __future__ import annotations

import sys

from evals.common import grade_table_content, grade_table_main


def grade(instance_dir, produced, adapter=None):
    return grade_table_content("full_reload", instance_dir, produced)


def main(argv=None) -> int:
    return grade_table_main("full_reload", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
