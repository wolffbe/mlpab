## Tasks (RQ2-style: per-task specialisation)

This session has {n_tasks} TASK(s) (ML task types). Process them sequentially:
for each task, run {max_increments} increments using ONLY that task's
challenges, producing a task-specialised interface version.

Task order + challenges:
{tasks_list}

Tag every increments.jsonl entry with `"task": "<task_name>"` so the resulting
interface versions can be cross-referenced to ML task type during analysis.

Total runs in this session: {n_tasks} tasks × {max_increments} increments ×
n_interfaces × challenges_per_task.

---
