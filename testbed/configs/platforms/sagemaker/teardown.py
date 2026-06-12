"""SageMaker platform teardown — delete every resource the agent created.

Run automatically by `banter run` (the interface `teardown:` step) at the START
and END of every challenge, so each run — and therefore each autoresearch
version — starts and ends with a clean platform (same contract as the hopsworks
teardown).

Unlike Hopsworks there is NO project container whose deletion cascades:
SageMaker resources live flat in the (account, region) and must be enumerated
and deleted PER TYPE. This script therefore ASSUMES the AWS account/region
behind the .env credentials is DEDICATED to the testbed — it deletes ALL
resources of the swept types in AWS_REGION, regardless of name (the agent picks
its own names, exactly like the hopsworks teardown deletes whatever project the
agent created).

Sweeps in COST order — the runner caps each aux step (~60 s), so if the script
is cut short the billable resources are already gone:
  1. endpoints + MLflow tracking servers (billed per hour while they exist)
  2. stop in-flight training / processing / transform / tuning / AutoML jobs
     (billed per second; finished jobs are inert history and cannot be deleted)
  3. free metadata: endpoint configs, models, feature groups, pipelines,
     model packages (+ groups), experiments (+ trials + components)
  4. S3 Vectors vector buckets + their indexes (vector storage bills per GB;
     stale same-named indexes would confound the vector_search task)
  5. contents of the default SageMaker bucket `sagemaker-<region>-<account>`
     (stale datasets/artifacts confound later runs the same way a stale
     hopsworks feature view zero-filled SDK v0's features)
  6. CloudWatch log groups under `/aws/sagemaker/` — jobs themselves cannot be
     deleted (inert history), but their log groups can and do accrue storage
     charges (the "Clean up" page of the SageMaker developer guide lists them
     alongside endpoints/configs/models/notebook instances/the S3 bucket)

Uses botocore DIRECTLY (not boto3): botocore is the one AWS lib present in all
three run venvs (awscli v1 and the sagemaker SDK / MCP server all depend on it),
so the same script works for cli, sdk, and mcp runs.

Best-effort by design: invoked via the interface `teardown:` step, whose runner
(`_run_aux`) ignores failures and discards output. Nothing here may raise out of
`main()` — a teardown hiccup must never fail an engineer run. Resources mid
state-transition (e.g. an endpoint still Creating) refuse deletion; the next
run's start-of-run sweep catches them.
"""
from __future__ import annotations

import os


def _list(client, op: str, result_key: str, **kwargs) -> list[dict]:
    """All items from a NextToken-paginated SageMaker list call; [] on failure."""
    items: list[dict] = []
    token = None
    try:
        while True:
            params = dict(kwargs, NextToken=token) if token else kwargs
            resp = getattr(client, op)(**params)
            items += resp.get(result_key) or []
            token = resp.get("NextToken")
            if not token:
                break
    except Exception as e:
        print(f"[sagemaker teardown] {op} failed: {e}")
    return items


def _sweep(client, label: str, list_op: str, result_key: str, name_key: str,
           delete_op: str, delete_param: str, **list_kwargs) -> None:
    """List every <name_key> and call <delete_op> on each, best-effort."""
    for item in _list(client, list_op, result_key, **list_kwargs):
        name = item.get(name_key)
        if not name:
            continue
        try:
            getattr(client, delete_op)(**{delete_param: name})
            print(f"[sagemaker teardown] {delete_op} {label} {name!r}")
        except Exception as e:
            print(f"[sagemaker teardown] {delete_op} {label} {name!r} failed: {e}")


def _empty_default_bucket(session, region: str) -> None:
    """Delete all objects in the default SageMaker bucket (bucket itself stays —
    it is plumbing, and keeping it avoids first-write name-propagation flakes)."""
    try:
        sts = session.create_client("sts", region_name=region)
        account = sts.get_caller_identity()["Account"]
        s3 = session.create_client("s3", region_name=region)
    except Exception as e:
        print(f"[sagemaker teardown] s3 client unavailable: {e}")
        return
    bucket = f"sagemaker-{region}-{account}"
    deleted = 0
    token = None
    try:
        while True:
            params = {"Bucket": bucket}
            if token:
                params["ContinuationToken"] = token
            resp = s3.list_objects_v2(**params)
            keys = [{"Key": o["Key"]} for o in resp.get("Contents") or []]
            if keys:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": keys, "Quiet": True})
                deleted += len(keys)
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
    except Exception as e:
        # Bucket may simply not exist yet (setup.py creates it) — that's fine.
        print(f"[sagemaker teardown] bucket {bucket!r} sweep stopped: {e}")
        return
    print(f"[sagemaker teardown] emptied bucket {bucket!r} ({deleted} object(s))")


def _delete_sagemaker_log_groups(session, region: str) -> None:
    """Delete every CloudWatch log group under `/aws/sagemaker/` (endpoint,
    training-job, processing-job, notebook logs, …). Jobs are undeletable
    history, but their logs are not — and they bill for storage."""
    try:
        logs = session.create_client("logs", region_name=region)
    except Exception as e:
        print(f"[sagemaker teardown] logs client unavailable: {e}")
        return
    deleted = 0
    token = None
    try:
        while True:
            params = {"logGroupNamePrefix": "/aws/sagemaker/"}
            if token:
                params["nextToken"] = token
            resp = logs.describe_log_groups(**params)
            for group in resp.get("logGroups") or []:
                name = group.get("logGroupName")
                if not name:
                    continue
                try:
                    logs.delete_log_group(logGroupName=name)
                    deleted += 1
                    print(f"[sagemaker teardown] deleted log group {name!r}")
                except Exception as e:
                    print(f"[sagemaker teardown] delete of log group {name!r} failed: {e}")
            token = resp.get("nextToken")
            if not token:
                break
    except Exception as e:
        print(f"[sagemaker teardown] log group sweep stopped: {e}")
        return
    if deleted:
        print(f"[sagemaker teardown] deleted {deleted} /aws/sagemaker/ log group(s)")


def main() -> None:
    try:
        import botocore.session
    except Exception as e:  # no AWS lib in this venv → nothing to do
        print(f"[sagemaker teardown] botocore unavailable: {e}")
        return

    session = botocore.session.get_session()
    region = os.environ.get("AWS_REGION") or session.get_config_variable("region")
    try:
        sm = session.create_client("sagemaker", region_name=region)
    except Exception as e:
        print(f"[sagemaker teardown] sagemaker client unavailable: {e}")
        return

    # 1) Billed-per-hour resources first.
    _sweep(sm, "endpoint", "list_endpoints", "Endpoints", "EndpointName",
           "delete_endpoint", "EndpointName")
    _sweep(sm, "mlflow server", "list_mlflow_tracking_servers", "TrackingServerSummaries",
           "TrackingServerName", "delete_mlflow_tracking_server", "TrackingServerName")
    # Notebook instances also bill per hour while InService: stop running ones,
    # delete stopped ones. Delete requires Stopped, so a just-stopped instance
    # (Stopping) is picked up by the NEXT sweep — same convergence story as an
    # endpoint mid-Creating (this runs at start AND end of every run).
    _sweep(sm, "notebook instance", "list_notebook_instances", "NotebookInstances",
           "NotebookInstanceName", "stop_notebook_instance", "NotebookInstanceName",
           StatusEquals="InService")
    _sweep(sm, "notebook instance", "list_notebook_instances", "NotebookInstances",
           "NotebookInstanceName", "delete_notebook_instance", "NotebookInstanceName",
           StatusEquals="Stopped")

    # 2) Stop billed-per-second jobs still running (jobs cannot be deleted —
    #    finished ones are inert history).
    _sweep(sm, "training job", "list_training_jobs", "TrainingJobSummaries",
           "TrainingJobName", "stop_training_job", "TrainingJobName",
           StatusEquals="InProgress")
    _sweep(sm, "processing job", "list_processing_jobs", "ProcessingJobSummaries",
           "ProcessingJobName", "stop_processing_job", "ProcessingJobName",
           StatusEquals="InProgress")
    _sweep(sm, "transform job", "list_transform_jobs", "TransformJobSummaries",
           "TransformJobName", "stop_transform_job", "TransformJobName",
           StatusEquals="InProgress")
    _sweep(sm, "tuning job", "list_hyper_parameter_tuning_jobs",
           "HyperParameterTuningJobSummaries", "HyperParameterTuningJobName",
           "stop_hyper_parameter_tuning_job", "HyperParameterTuningJobName",
           StatusEquals="InProgress")
    _sweep(sm, "automl job", "list_auto_ml_jobs", "AutoMLJobSummaries",
           "AutoMLJobName", "stop_auto_ml_job", "AutoMLJobName",
           StatusEquals="InProgress")

    # 3) Free metadata.
    _sweep(sm, "endpoint config", "list_endpoint_configs", "EndpointConfigs",
           "EndpointConfigName", "delete_endpoint_config", "EndpointConfigName")
    _sweep(sm, "model", "list_models", "Models", "ModelName",
           "delete_model", "ModelName")
    _sweep(sm, "feature group", "list_feature_groups", "FeatureGroupSummaries",
           "FeatureGroupName", "delete_feature_group", "FeatureGroupName")
    _sweep(sm, "pipeline", "list_pipelines", "PipelineSummaries", "PipelineName",
           "delete_pipeline", "PipelineName")

    # Model packages nest under groups: delete packages, then their group.
    for group in _list(sm, "list_model_package_groups", "ModelPackageGroupSummaryList"):
        gname = group.get("ModelPackageGroupName")
        if not gname:
            continue
        _sweep(sm, "model package", "list_model_packages", "ModelPackageSummaryList",
               "ModelPackageArn", "delete_model_package", "ModelPackageName",
               ModelPackageGroupName=gname)
        try:
            sm.delete_model_package_group(ModelPackageGroupName=gname)
            print(f"[sagemaker teardown] deleted model package group {gname!r}")
        except Exception as e:
            print(f"[sagemaker teardown] delete of model package group {gname!r} failed: {e}")

    # Experiments nest trials nest components: unwind inside-out.
    for exp in _list(sm, "list_experiments", "ExperimentSummaries"):
        ename = exp.get("ExperimentName")
        if not ename:
            continue
        for trial in _list(sm, "list_trials", "TrialSummaries", ExperimentName=ename):
            tname = trial.get("TrialName")
            if not tname:
                continue
            for comp in _list(sm, "list_trial_components", "TrialComponentSummaries",
                              TrialName=tname):
                cname = comp.get("TrialComponentName")
                if not cname:
                    continue
                try:
                    sm.disassociate_trial_component(TrialComponentName=cname, TrialName=tname)
                    sm.delete_trial_component(TrialComponentName=cname)
                except Exception:
                    pass  # may still be associated with another trial
            try:
                sm.delete_trial(TrialName=tname)
            except Exception:
                pass
        try:
            sm.delete_experiment(ExperimentName=ename)
            print(f"[sagemaker teardown] deleted experiment {ename!r}")
        except Exception as e:
            print(f"[sagemaker teardown] delete of experiment {ename!r} failed: {e}")

    # 4) S3 Vectors — vector indexes nest under vector buckets: delete the
    #    indexes, then their bucket (delete_vector_bucket requires it empty).
    #    The `s3vectors` client only exists in recent botocore; older venvs
    #    simply skip the sweep (same guard story as the main client above).
    try:
        s3v = session.create_client("s3vectors", region_name=region)
    except Exception as e:
        print(f"[sagemaker teardown] s3vectors client unavailable: {e}")
        s3v = None
    if s3v is not None:
        for vb in _list(s3v, "list_vector_buckets", "vectorBuckets"):
            bname = vb.get("vectorBucketName")
            if not bname:
                continue
            # delete_index needs (vectorBucketName, indexName) or the ARN;
            # _sweep passes a single delete param, so sweep by indexArn.
            _sweep(s3v, "vector index", "list_indexes", "indexes", "indexArn",
                   "delete_index", "indexArn", vectorBucketName=bname)
            try:
                s3v.delete_vector_bucket(vectorBucketName=bname)
                print(f"[sagemaker teardown] deleted vector bucket {bname!r}")
            except Exception as e:
                print(f"[sagemaker teardown] delete of vector bucket {bname!r} failed: {e}")

    # 5) Stale artifacts/datasets in the default bucket.
    _empty_default_bucket(session, region)

    # 6) CloudWatch log groups (jobs can't be deleted; their logs can).
    _delete_sagemaker_log_groups(session, region)

    print("[sagemaker teardown] done")


if __name__ == "__main__":
    main()
