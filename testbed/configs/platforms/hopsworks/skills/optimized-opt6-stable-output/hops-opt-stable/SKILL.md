---
name: hops-opt-stable
description: This CLI emits stable, sorted JSON keys and sorted `fg list` rows, so output is deterministic. Auto-invoke when comparing or caching CLI output across calls.
---

# opt6: stable / deterministic output

This CLI build sorts JSON keys (`sort_keys`) and sorts `hops fg list` rows, so
output is **deterministic across calls**.

- You can rely on ordering: the first row of `hops fg list` is stable, fields
  appear in a fixed order, so you can parse by position and **cache/compare**
  results between calls without re-reading everything.
- When you've already listed once, trust the stable order instead of re-listing
  to re-confirm — it won't have reshuffled. Fewer redundant reads = less cost.
