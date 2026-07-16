"""MDT (model-dependent transformations) task — grader: the standard table
suite over the scaled-features table (truth.json `table_name`) read back
through the platform (or --csv)."""

from __future__ import annotations

import sys

from evals.common import grade_table_content, grade_table_main


def grade(instance_dir, produced, adapter=None):
    return grade_table_content("mdt", instance_dir, produced)


def main(argv=None) -> int:
    return grade_table_main("mdt", grade, argv)


if __name__ == "__main__":
    sys.exit(main())
