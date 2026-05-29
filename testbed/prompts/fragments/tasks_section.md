## Tasks (RQ2-style: per-task specialisation)

This session has {n_tasks} TASK(s) (ML task types). Process them sequentially:
for each task, run {max_increments} increments using ONLY that task's
challenges, producing a task-specialised interface version.

Task order + challenges:
{tasks_list}

Each `banter run` already records its `task` column from the `--task` flag;
that's the grouping key for per-task analysis in results.csv.

Total runs in this session: {n_tasks} tasks × {max_increments} increments ×
n_interfaces × challenges_per_task.

---
