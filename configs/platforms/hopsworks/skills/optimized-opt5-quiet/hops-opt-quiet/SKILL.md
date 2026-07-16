---
name: hops-opt-quiet
description: This CLI supports global `--quiet` and drops null-valued keys from JSON, cutting output noise. Auto-invoke on every CLI call where you don't need progress logs.
---

# opt5: quiet output

This CLI build adds a global **`--quiet`** flag (suppresses progress/log chatter)
and **drops null-valued keys** from JSON payloads.

- Add `--quiet` to essentially every `hops` call — you rarely need the progress
  logs, and suppressing them shrinks the output you pay to read.
- JSON is already trimmed of null keys, so captured output is smaller and the
  fields that remain are the meaningful ones.
- Quiet + capture = the cheapest way to drive the CLI.
