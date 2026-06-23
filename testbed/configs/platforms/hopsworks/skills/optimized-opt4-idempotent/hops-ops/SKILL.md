---
name: hops-ops
description: Operating Hopsworks ML systems — jobs/scheduling, feature monitoring & data drift, alerting, and data lineage via CLI + SDK. Auto-invoke for scheduled jobs, monitoring, drift detection, prediction monitoring, alerts, or lineage/provenance tasks.
---

# Ops: jobs, monitoring, alerting, lineage

## Jobs & scheduling (CLI)

```bash
hops job list ; hops job info <name>
hops job deploy <name> <entrypoint> --env <env>   # register a pipeline as a job
hops job schedule <name> --cron "<expr>"          # run on a schedule
hops job logs <name> ; hops job history <name>
```

A feature/training/inference pipeline becomes operational by being deployed as a job and (optionally) scheduled. Read logs/history to confirm a run succeeded rather than assuming.

## Feature monitoring & drift

Monitoring tracks statistics (distributions, null rates) on a feature group/view over time and compares windows to a reference to detect **drift**. Configure it on the FG/FV via the SDK (`fg.create_statistics_monitoring(...)` / feature-monitoring config), then let scheduled runs populate results. Inspect baseline stats cheaply with `hops fg stats <name> --version 1`.

## Prediction monitoring

Log inference inputs/outputs (the online inference pipeline does this), then monitor the logged predictions the same way as features — distribution shift in predictions signals model or data drift.

## Alerting (CLI)

```bash
hops alert --help          # list/create/route alerts (mirrors the SDK AlertsApi)
```

Attach alerts to monitoring results so quality degradation (drift, missing values, failed jobs) notifies a receiver. Create the receiver/route, then bind the trigger.

## Lineage / provenance

Lineage is established by declaring relationships, not reconstructed after the fact: pass `parents=[...]` when creating derived feature groups, and use feature views (which record their source FGs). Inspect provenance via the SDK (`fg.get_parent_feature_groups()` / `get_generated_feature_groups()`) or the UI lineage graph.
