"""Tests for REST-endpoint coverage tracking: the `endpoint_hits` matcher, the
metric registration, and the venv request-logging shim."""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from mlpab import claude_runner, results


def _log(*rows) -> Path:
    p = Path(tempfile.mkdtemp()) / "api_calls.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return p


FG_CREATE = "POST /hopsworks-api/api/project/[0-9]+/featurestores/[0-9]+/featuregroups"
FG_GET = "GET /hopsworks-api/api/project/[0-9]+/featurestores/[0-9]+/featuregroups/[^/]+"
FV_CREATE = "POST /hopsworks-api/api/project/[0-9]+/featurestores/[0-9]+/featureview"


class EndpointHitsTests(unittest.TestCase):
    def test_distinct_coverage_not_volume(self):
        log = _log(
            {
                "method": "POST",
                "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups",
            },
            {
                "method": "GET",
                "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups/fg",
            },
            {
                "method": "GET",
                "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups/fg",
            },  # dup
        )
        r = results.endpoint_hits(log, [FG_CREATE, FG_GET, FV_CREATE], [])
        # 2 distinct whitelisted patterns hit (create + get); the dup get doesn't add.
        self.assertEqual(r["whitelist_hits"], 2)
        self.assertEqual(r["blacklist_hits"], 0)

    def test_blacklist_counts_every_matching_call(self):
        log = _log(
            {"method": "DELETE", "path": "/hopsworks-api/api/project/1"},
            {"method": "DELETE", "path": "/hopsworks-api/api/project/2"},
        )
        r = results.endpoint_hits(log, [], ["DELETE /hopsworks-api/api/project/[0-9]+$"])
        self.assertEqual(r["blacklist_hits"], 2)

    def test_method_must_match(self):
        log = _log(
            {"method": "GET", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups"}
        )
        # whitelist wants POST for create → GET shouldn't match it.
        self.assertEqual(results.endpoint_hits(log, [FG_CREATE], [])["whitelist_hits"], 0)

    def test_any_method_when_unspecified(self):
        log = _log({"method": "GET", "path": "/x/y"})
        self.assertEqual(results.endpoint_hits(log, ["/x/y"], [])["whitelist_hits"], 1)

    def test_missing_or_empty_log_zeros(self):
        self.assertEqual(
            results.endpoint_hits(Path("/nope/api.jsonl"), [FG_CREATE], [FG_GET]),
            {"whitelist_hits": 0, "blacklist_hits": 0},
        )

    def test_no_patterns_zeros(self):
        log = _log({"method": "GET", "path": "/x"})
        self.assertEqual(
            results.endpoint_hits(log, [], []), {"whitelist_hits": 0, "blacklist_hits": 0}
        )

    def test_prefix_pattern_does_not_overmatch_deeper_path(self):
        # FV-create POST .../featureview must NOT match the deeper TD-create path
        # .../featureview/{name}/version/{v}/trainingdatasets (end-anchored).
        td_create = {
            "method": "POST",
            "path": "/hopsworks-api/api/project/1/featurestores/9/featureview/fv/version/1/trainingdatasets",
        }
        log = _log(td_create)
        r = results.endpoint_hits(log, [FV_CREATE], [])
        self.assertEqual(r["whitelist_hits"], 0)  # the TD path is not an FV-create

    def test_fg_get_does_not_match_ingestion_subpath(self):
        # FG-get .../featuregroups/[^/]+ must not match .../featuregroups/5/ingestion
        log = _log(
            {
                "method": "POST",
                "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups/5/ingestion",
            }
        )
        self.assertEqual(results.endpoint_hits(log, [FG_GET], [])["whitelist_hits"], 0)

    def test_trailing_slash_tolerated(self):
        log = _log(
            {
                "method": "POST",
                "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups/",
            }
        )
        self.assertEqual(results.endpoint_hits(log, [FG_CREATE], [])["whitelist_hits"], 1)

    def test_real_sdk_get_paths_match(self):
        # The real SDK GET paths (verified against hopsworks-api source): FV get
        # is .../featureview/{name}/version/{v}; deployment get can be /serving/{id};
        # app get can be /apps/{name}. The optional-group patterns must match these.
        fv_get = "GET /hopsworks-api/api/project/[0-9]+/featurestores/[0-9]+/featureview/[^/]+(?:/version/[0-9]+)?"
        dep_get = "GET /hopsworks-api/api/project/[0-9]+/serving(?:/[0-9]+)?"
        app_get = "GET /hopsworks-api/api/project/[0-9]+/apps(?:/[^/]+)?"
        log = _log(
            {
                "method": "GET",
                "path": "/hopsworks-api/api/project/1/featurestores/9/featureview/fv/version/1",
            },
            {"method": "GET", "path": "/hopsworks-api/api/project/1/serving/5"},
            {"method": "GET", "path": "/hopsworks-api/api/project/1/apps/myapp"},
        )
        self.assertEqual(
            results.endpoint_hits(log, [fv_get, dep_get, app_get], [])["whitelist_hits"], 3
        )
        # FV-get must NOT also swallow the deeper TD path.
        td = _log(
            {
                "method": "GET",
                "path": "/hopsworks-api/api/project/1/featurestores/9/featureview/fv/version/1/trainingdatasets",
            }
        )
        self.assertEqual(results.endpoint_hits(td, [fv_get], [])["whitelist_hits"], 0)


class EndpointCoverageTests(unittest.TestCase):
    def test_covered_and_missed_split(self):
        log = _log(
            {"method": "POST", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups"}
        )
        cov = results.endpoint_coverage(log, [FG_CREATE, FG_GET, FV_CREATE], [])
        self.assertEqual(cov["whitelist_hits"], 1)
        self.assertEqual(cov["total_whitelist"], 3)
        self.assertEqual(cov["covered"], [FG_CREATE])
        self.assertEqual(set(cov["missed"]), {FG_GET, FV_CREATE})

    def test_missing_log_all_missed(self):
        cov = results.endpoint_coverage(Path("/nope/api.jsonl"), [FG_CREATE, FG_GET], [])
        self.assertEqual(cov["whitelist_hits"], 0)
        self.assertEqual(cov["covered"], [])
        self.assertEqual(cov["missed"], [FG_CREATE, FG_GET])

    def test_endpoint_hits_matches_coverage_counts(self):
        log = _log(
            {"method": "POST", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups"}
        )
        cov = results.endpoint_coverage(log, [FG_CREATE, FG_GET], ["DELETE /x"])
        hits = results.endpoint_hits(log, [FG_CREATE, FG_GET], ["DELETE /x"])
        self.assertEqual(hits["whitelist_hits"], cov["whitelist_hits"])
        self.assertEqual(hits["blacklist_hits"], cov["blacklist_hits"])

    def test_coverage_embeds_original_patterns(self):
        # endpoint_coverage.json must be self-contained once api_calls.jsonl is
        # discarded: it carries the exact whitelist/blacklist it was scored against.
        log = _log(
            {"method": "POST", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups"}
        )
        cov = results.endpoint_coverage(log, [FG_CREATE, FG_GET], ["DELETE /x"])
        self.assertEqual(cov["whitelist"], [FG_CREATE, FG_GET])
        self.assertEqual(cov["blacklist"], ["DELETE /x"])

    def test_coverage_patterns_default_to_empty_lists(self):
        cov = results.endpoint_coverage(Path("/nope/api.jsonl"), None, None)
        self.assertEqual(cov["whitelist"], [])
        self.assertEqual(cov["blacklist"], [])


class AttributionTests(unittest.TestCase):
    """Whitelist coverage is credited only to calls made THROUGH the interface
    under test (the shim's `src` tag); hand-rolled `requests` ("other") and
    server-side Job calls (never logged) don't count."""

    FG = {"method": "POST", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups"}

    def test_only_active_interface_src_counts(self):
        log = _log({**self.FG, "src": "mcp"})
        self.assertEqual(
            results.endpoint_hits(log, [FG_CREATE], [], interface="mcp")["whitelist_hits"], 1
        )
        # Same call attributed to MCP does NOT count for a CLI/SDK run.
        self.assertEqual(
            results.endpoint_hits(log, [FG_CREATE], [], interface="cli")["whitelist_hits"], 0
        )
        self.assertEqual(
            results.endpoint_hits(log, [FG_CREATE], [], interface="sdk")["whitelist_hits"], 0
        )

    def test_other_src_never_counts_when_attributed(self):
        # A hand-rolled `requests.post(...)` (not via the interface) → src "other".
        log = _log({**self.FG, "src": "other"})
        self.assertEqual(
            results.endpoint_hits(log, [FG_CREATE], [], interface="mcp")["whitelist_hits"], 0
        )
        self.assertEqual(
            results.endpoint_hits(log, [FG_CREATE], [], interface="sdk")["whitelist_hits"], 0
        )

    def test_no_interface_counts_any_src(self):
        # Back-compat: interface=None (e.g. none/none baseline or legacy logs)
        # scores any logged call regardless of attribution.
        log = _log({**self.FG, "src": "other"})
        self.assertEqual(results.endpoint_hits(log, [FG_CREATE], [])["whitelist_hits"], 1)

    def test_legacy_log_without_src_counts_only_unattributed(self):
        # A row with no `src` (legacy shim) reads as "" — counted only when
        # interface=None, never credited to a specific interface.
        log = _log(dict(self.FG))  # no "src" key
        self.assertEqual(results.endpoint_hits(log, [FG_CREATE], [])["whitelist_hits"], 1)
        self.assertEqual(
            results.endpoint_hits(log, [FG_CREATE], [], interface="mcp")["whitelist_hits"], 0
        )

    def test_blacklist_not_attribution_filtered(self):
        # A forbidden call is a violation however it was made — blacklist counts
        # every matching row regardless of `src` (even "other").
        log = _log({"method": "DELETE", "path": "/hopsworks-api/api/project/1", "src": "other"})
        bl = ["DELETE /hopsworks-api/api/project/[0-9]+$"]
        self.assertEqual(results.endpoint_hits(log, [], bl, interface="mcp")["blacklist_hits"], 1)


class EndpointSchemaTests(unittest.TestCase):
    def test_metrics_registered(self):
        # The endpoint metrics stay tracked on the Row (and in the tracked
        # call-count columns) but are NOT columns of the results CSV.
        for m in ("whitelist_hits", "blacklist_hits"):
            self.assertIn(m, results.CALL_COUNT_COLUMNS)
            self.assertNotIn(m, results.RESULTS_FIELDS)
        self.assertEqual(results.RESULTS_FIELDS[-1], "run_dir")  # invariant preserved

    def test_row_carries_endpoint_metrics_but_csv_drops_them(self):
        import csv

        out = Path(tempfile.mkdtemp()) / "results.csv"
        row = results.Row(
            started_at="2026-06-04T00:00:00+00:00",
            run="1",
            version="v1",
            platform="hopsworks",
            interface="cli",
            skills="none",
            category="t",
            task="c",
            whitelist_hits=5,
            blacklist_hits=0,
            run_dir=str(out.parent),
        )
        self.assertEqual(row.whitelist_hits, 5)  # still recorded on the Row
        results.append(out, row)
        got = list(csv.DictReader(out.open()))[0]
        self.assertNotIn("whitelist_hits", got)
        self.assertNotIn("blacklist_hits", got)


class ApiLogShimTests(unittest.TestCase):
    def test_shim_patches_requests_and_logs_method_and_path(self):
        fake = types.ModuleType("requests")

        class Session:
            def send(self, request, **kw):
                return "RESP"

        fake.Session = Session
        log = Path(tempfile.mkdtemp()) / "api.jsonl"
        old = sys.modules.get("requests")
        sys.modules["requests"] = fake
        os.environ["MLPAB_API_LOG"] = str(log)
        try:
            exec(claude_runner._API_LOG_SHIM, {})
            req = types.SimpleNamespace(
                method="post",
                url="https://h:443/hopsworks-api/api/project/1/featurestores/9/featuregroups?x=1",
            )
            self.assertEqual(fake.Session().send(req), "RESP")  # original still called
            rec = json.loads(log.read_text().splitlines()[0])
            self.assertEqual(rec["method"], "POST")
            self.assertEqual(
                rec["path"], "/hopsworks-api/api/project/1/featurestores/9/featuregroups"
            )
            self.assertIn("src", rec)  # attribution tag always present
        finally:
            if old is not None:
                sys.modules["requests"] = old
            else:
                sys.modules.pop("requests", None)
            os.environ.pop("MLPAB_API_LOG", None)

    def _origin_from(self, module_name: str, roots: str) -> str:
        """Run the shim's `_mlpab_origin` from a synthetic frame whose module
        `__name__` is `module_name`, with MLPAB_IFACE_SDK=`roots`. Returns the
        attributed `src`."""
        log = Path(tempfile.mkdtemp()) / "api.jsonl"
        os.environ["MLPAB_API_LOG"] = str(log)
        os.environ["MLPAB_IFACE_SDK"] = roots
        old = sys.modules.get("requests")
        fake = types.ModuleType("requests")
        fake.Session = type("Session", (), {"send": lambda self, r, **k: None})
        sys.modules["requests"] = fake
        try:
            g: dict = {}
            exec(claude_runner._API_LOG_SHIM, g)
            ns = {"__name__": module_name, "_origin": g["_mlpab_origin"]}
            exec("def _f():\n    return _origin()\n", ns)
            return ns["_f"]()
        finally:
            if old is not None:
                sys.modules["requests"] = old
            else:
                sys.modules.pop("requests", None)
            os.environ.pop("MLPAB_API_LOG", None)
            os.environ.pop("MLPAB_IFACE_SDK", None)

    def test_origin_attributes_mcp_cli_sdk_and_other(self):
        roots = "hopsworks,hopsworks_common,hsfs,hsml"
        # A `.mcp` / `.cli` subpackage of a first-party root wins over the SDK
        # client it wraps; any other first-party frame is "sdk"; none → "other".
        self.assertEqual(self._origin_from("hopsworks.mcp.tools.jobs", roots), "mcp")
        self.assertEqual(self._origin_from("hopsworks.cli.main", roots), "cli")
        self.assertEqual(self._origin_from("hsfs.feature_group", roots), "sdk")
        self.assertEqual(self._origin_from("hopsworks_common.core.feature_group_api", roots), "sdk")
        self.assertEqual(self._origin_from("my_glue_script", roots), "other")

    def test_origin_other_when_no_roots(self):
        # No first-party roots configured → nothing is interface-attributed.
        self.assertEqual(self._origin_from("hopsworks.mcp.tools.jobs", ""), "other")


class FirstPartyRootsTests(unittest.TestCase):
    def _venv(self, dist: str, top_level: list[str], subpkgs: dict[str, list[str]]):
        root = Path(tempfile.mkdtemp())
        sp = root / "lib" / "python3.11" / "site-packages"
        (sp / f"{dist}.dist-info").mkdir(parents=True)
        (sp / f"{dist}.dist-info" / "top_level.txt").write_text("\n".join(top_level) + "\n")
        for pkg, subs in subpkgs.items():
            for sub in subs:
                (sp / pkg / sub).mkdir(parents=True, exist_ok=True)
        return root

    def test_picks_dist_shipping_mcp_or_cli_subpackage(self):
        # The interface dist is the one whose packages include an mcp/cli subtree;
        # returns ALL of that dist's top-level packages.
        venv = self._venv(
            "hopsworks-0",
            ["hopsworks", "hopsworks_common", "hsfs", "hsml"],
            {"hopsworks": ["mcp", "cli"]},
        )
        self.assertEqual(
            set(claude_runner._first_party_roots(venv)),
            {"hopsworks", "hopsworks_common", "hsfs", "hsml"},
        )

    def test_empty_when_no_interface_dist(self):
        # A venv with only third-party dists (no mcp/cli subpackage) → no roots,
        # so the shim safely tags every call "other".
        venv = self._venv("requests-2", ["requests"], {})
        self.assertEqual(claude_runner._first_party_roots(venv), [])


if __name__ == "__main__":
    unittest.main()
