"""Recsys task — grader: the standard table suite over the recommendations
table (truth.json `table_name`) read back through the platform (or --csv locally)."""

from __future__ import annotations

import sys

from evals.common import grade_table_content, grade_table_main


def grade(instance_dir, produced, adapter=None):
    return grade_table_content("recsys", instance_dir, produced)


def main(argv=None) -> int:
    return grade_table_main("recsys", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
