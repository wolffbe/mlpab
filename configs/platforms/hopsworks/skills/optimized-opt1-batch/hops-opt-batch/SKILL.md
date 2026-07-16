---
name: hops-opt-batch
description: Run many `hops` subcommands in ONE process / ONE login via `hops batch`. Auto-invoke whenever you will issue several CLI calls in a row.
---

# opt1: batch many calls into one process

This CLI build adds **`hops batch`**: it runs a list of subcommands in a single
process with a single login, instead of paying process-start + auth on every call.

```bash
hops batch --help
# feed multiple commands (one per line / as documented by --help), e.g.:
hops batch <<'EOF'
fg list
fg info my_fg --version 1
fg preview my_fg --version 1 --n 10
EOF
```

When you have several reads or several mutations to do, **group them into one
`hops batch`** rather than many separate `hops ...` invocations. Fewer logins and
process starts = less wall time and lower cost. Confirm exact syntax with
`hops batch --help`.
