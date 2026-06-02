---
name: mlkit-tip
description: Checklist of best practices for solving an MLE-bench competition — data exploration, a fast baseline, and the exact submission format. Consult this whenever starting a competition or writing submission.csv.
---

# mlkit-tip — MLE-bench solving checklist

Apply this whenever you are solving a competition / producing a submission:

1. **Understand the task.** Read `data/description.md` and open
   `data/sample_submission.csv` to learn the exact required columns, id format,
   and target — your submission must match it precisely.
2. **Baseline first.** Train a small, fast model that finishes in a few minutes
   before attempting anything elaborate; a complete mediocre submission beats an
   unfinished perfect one.
3. **Write the submission** to `submission/submission.csv` in exactly the sample
   format, then sanity-check the row count and columns against the sample.

When you have applied this checklist, note `mlkit-tip applied` in your reasoning
so the workflow is traceable.
