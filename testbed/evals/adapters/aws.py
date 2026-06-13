"""SageMaker checker adapter — grader-side reads through AWS.

Uses botocore/boto3 with the standard AWS env credentials. Platform
realization conventions:

  * feature tables are SageMaker Feature Groups (`describe_feature_group`
    gives record key, event time, and schema — all first-class there);
  * a "versioned training dataset named X, version N" is one or more
    CSV/Parquet objects in the default bucket
    (`$SAGEMAKER_DEFAULT_BUCKET`, else `sagemaker-<region>-<account>`) whose
    key contains BOTH the dataset name and a version marker (`vN`,
    `version=N`, or `/N/`). The recommended layout is
        s3://<bucket>/training_datasets/<name>/v<version>/part-*.csv
    but any key matching the name+version search is accepted.

`read_rows` is NOT implemented: the offline store materializes asynchronously
(5–15 min) and querying it needs Athena/Glue permissions outside the testbed's
policy; content asserts on SageMaker go through the online store
(`batch_get_record`, see `get_records`) or the S3 deliverable instead.

boto3 is imported lazily so this module imports without it.

CLI (for live probing):
    python -m evals.adapters.aws describe-fg --name transactions
    python -m evals.adapters.aws read-td --name churn_training --version 1 --out /tmp/td.csv
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

from evals.adapters import TableInfo


class SageMakerChecker:
    def __init__(self) -> None:
        import boto3  # lazy

        region = os.environ.get("AWS_REGION")
        self._sm = boto3.client("sagemaker", region_name=region)
        self._s3 = boto3.client("s3", region_name=region)
        self._fsr = boto3.client("sagemaker-featurestore-runtime", region_name=region)
        bucket = os.environ.get("SAGEMAKER_DEFAULT_BUCKET")
        if not bucket:
            sts = boto3.client("sts", region_name=region)
            account = sts.get_caller_identity()["Account"]
            bucket = f"sagemaker-{region}-{account}"
        self._bucket = bucket

    # -- feature tables ----------------------------------------------------

    def get_feature_table(self, name: str, version: int | None = None) -> TableInfo | None:
        try:
            d = self._sm.describe_feature_group(FeatureGroupName=name)
        except Exception:
            return None
        return TableInfo(
            name=name,
            version=None,
            primary_key=[d["RecordIdentifierFeatureName"]],
            event_time=d.get("EventTimeFeatureName"),
            schema={f["FeatureName"]: f["FeatureType"]
                    for f in d.get("FeatureDefinitions", [])},
        )

    def read_rows(self, name: str, version: int | None = None) -> pd.DataFrame:
        raise NotImplementedError(
            "SageMaker offline-store reads need Athena/Glue permissions and "
            "materialize asynchronously — use get_records (online store) or "
            "read_training_dataset (S3 deliverable) instead."
        )

    def get_records(self, name: str, record_ids: list[str]) -> pd.DataFrame:
        """Online-store reads for known keys (content asserts on SageMaker)."""
        out = []
        for chunk in (record_ids[i:i + 100] for i in range(0, len(record_ids), 100)):
            resp = self._fsr.batch_get_record(Identifiers=[{
                "FeatureGroupName": name, "RecordIdentifiersValueAsString": chunk,
            }])
            for rec in resp.get("Records", []):
                out.append({f["FeatureName"]: f["ValueAsString"] for f in rec["Record"]})
        return pd.DataFrame(out)

    # -- training datasets ---------------------------------------------------

    def _candidate_keys(self, name: str, version: int) -> list[str]:
        markers = (f"v{version}", f"version={version}", f"/{version}/")
        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if name in k and any(m in k for m in markers) \
                        and re.search(r"\.(csv|parquet)$", k):
                    keys.append(k)
        return sorted(keys)

    def read_training_dataset(self, name: str, version: int = 1) -> pd.DataFrame:
        keys = self._candidate_keys(name, version)
        if not keys:
            raise LookupError(
                f"training dataset {name!r} v{version} not found in "
                f"s3://{self._bucket} — expected CSV/Parquet objects whose key "
                f"contains {name!r} and a v{version} marker (recommended: "
                f"training_datasets/{name}/v{version}/)"
            )
        frames = []
        for k in keys:
            body = self._s3.get_object(Bucket=self._bucket, Key=k)["Body"].read()
            if k.endswith(".parquet"):
                frames.append(pd.read_parquet(io.BytesIO(body)))
            else:
                frames.append(pd.read_csv(io.BytesIO(body)))
        return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("describe-fg")
    p.add_argument("--name", required=True)
    p = sub.add_parser("read-td")
    p.add_argument("--name", required=True)
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    checker = SageMakerChecker()
    if args.cmd == "describe-fg":
        info = checker.get_feature_table(args.name)
        print(json.dumps({"exists": info is not None,
                          **(info.__dict__ if info else {"name": args.name})},
                         default=str, indent=2))
        return 0 if info else 1
    checker.read_training_dataset(args.name, args.version).to_csv(args.out, index=False)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------
# Platform-state reads beyond the feature store (best-effort dicts: graders
# assert on `exists` plus whichever detail keys the API provides).
# --------------------------------------------------------------------------

def _state_reads(cls):
    def get_model(self, name: str, version: int = 1) -> dict:
        """Model registry realization: a model package group named `name`
        with at least `version` package versions."""
        try:
            self._sm.describe_model_package_group(ModelPackageGroupName=name)
        except Exception as e:
            return {"exists": False, "error": str(e)}
        try:
            pkgs = self._sm.list_model_packages(
                ModelPackageGroupName=name)["ModelPackageSummaryList"]
            return {"exists": True, "version": len(pkgs)}
        except Exception:
            return {"exists": True}

    def get_job(self, name: str) -> dict:
        """Job realization (checked in order): SageMaker Pipeline (scheduled
        pipelines), training job, processing job."""
        try:
            p = self._sm.describe_pipeline(PipelineName=name)
            return {"exists": True, "kind": "pipeline",
                    "scheduled": True,  # schedule lives in EventBridge; presence of
                                        # the pipeline is the gradable platform state
                    "last_run_state": p.get("PipelineStatus", "")}
        except Exception:
            pass
        for kind, describe, key in (
            ("training-job", "describe_training_job", "TrainingJobStatus"),
            ("processing-job", "describe_processing_job", "ProcessingJobStatus"),
        ):
            try:
                d = getattr(self._sm, describe)(**{
                    "TrainingJobName" if kind == "training-job" else "ProcessingJobName": name})
                return {"exists": True, "kind": kind,
                        "scheduled": False, "last_run_state": d.get(key, "")}
            except Exception:
                continue
        return {"exists": False}

    def get_endpoint(self, name: str) -> dict:
        """Real-time endpoint: {exists, status}."""
        try:
            d = self._sm.describe_endpoint(EndpointName=name)
            return {"exists": True, "status": d.get("EndpointStatus", "")}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_alert(self, name_or_hint: str) -> dict:
        """Alerting realization on SageMaker = a CloudWatch alarm whose name
        contains the hint."""
        try:
            import boto3, os
            cw = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION"))
            alarms = cw.describe_alarms()["MetricAlarms"]
            hits = [a for a in alarms if name_or_hint in a["AlarmName"]]
            return {"exists": bool(hits), "count": len(hits)}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    def get_vector_store(self, name: str) -> dict:
        """Vector store realization on SageMaker, accepted in order:

        1. An Amazon S3 Vectors index named `name` (in any vector bucket) —
           the managed path: `s3vectors` is on the CLI allowlist and ships
           in the run venvs' botocore (query-vectors does real ANN).
        2. A feature group `name` holding the vectors (online recommended) —
           storage without native search; neighbors computed interface-side.
        3. An InService real-time endpoint `name` — the self-hosted pattern
           (FAISS index served from a SageMaker endpoint, per AWS's own
           sagemaker-vector-store-microservice sample).
        """
        try:
            import boto3, os
            s3v = boto3.client("s3vectors", region_name=os.environ.get("AWS_REGION"))
            for vb in s3v.list_vector_buckets().get("vectorBuckets", []):
                bname = vb.get("vectorBucketName")
                for idx in s3v.list_indexes(
                        vectorBucketName=bname).get("indexes", []):
                    if idx.get("indexName") == name:
                        return {"exists": True, "kind": "s3vectors-index",
                                "native_ann": True, "vector_bucket": bname}
        except Exception:
            pass  # no IAM grant / old botocore → fall through to other shapes
        try:
            d = self._sm.describe_feature_group(FeatureGroupName=name)
            return {"exists": True, "kind": "feature-group", "native_ann": False,
                    "online_store": bool((d.get("OnlineStoreConfig") or {})
                                         .get("EnableOnlineStore"))}
        except Exception:
            pass
        try:
            e = self._sm.describe_endpoint(EndpointName=name)
            if e.get("EndpointStatus") == "InService":
                return {"exists": True, "kind": "endpoint",
                        "native_ann": False, "self_hosted": True}
            return {"exists": False,
                    "error": f"endpoint {name!r} status {e.get('EndpointStatus')!r}"}
        except Exception as e:
            return {"exists": False, "error": str(e)}

    cls.get_model = get_model
    cls.get_job = get_job
    cls.get_endpoint = get_endpoint
    cls.get_alert = get_alert
    cls.get_vector_store = get_vector_store
    return cls


_state_reads(SageMakerChecker)
