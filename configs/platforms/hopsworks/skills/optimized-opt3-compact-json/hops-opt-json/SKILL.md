---
name: hops-opt-json
description: This CLI emits compact JSON by default when piped; parse `--json` output directly. Auto-invoke whenever you read CLI output programmatically.
---

# opt3: compact JSON output

This CLI build emits **compact JSON by default when stdout is not a terminal**
(i.e. when you capture it), and supports `--json` explicitly. Compact = no pretty
whitespace, so output is smaller and cheaper to read.

- Prefer `--json` (or just capture the piped output) over human tables when you
  need to extract a value — parse it directly, don't eyeball a table.
- Pipe to a filter to grab exactly what you need, e.g.
  `hops fg list --json | python -c "import sys,json;..."`.
- Smaller output means you can afford more inspect calls for the same cost.
