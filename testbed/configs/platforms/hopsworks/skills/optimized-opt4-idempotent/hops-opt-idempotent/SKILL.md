---
name: hops-opt-idempotent
description: This CLI makes feature-group creation idempotent via `hops fg create --if-exists get|skip`, with actionable conflict/not-found errors. Auto-invoke for any create-or-get step.
---

# opt4: idempotent create

This CLI build adds **`--if-exists`** to creation:

```bash
hops fg create <name> --version 1 ... --if-exists get   # create, or fetch if it exists
hops fg create <name> --version 1 ... --if-exists skip   # create, or no-op if it exists
```

Use `--if-exists get` to **create-or-fetch in one step** instead of `list` →
branch → create. Re-running is safe: no crash on conflict. Errors are actionable
(distinguish not-found from conflict) — read them and adjust rather than retrying
the same command. This removes the most common source of wasted retries.
