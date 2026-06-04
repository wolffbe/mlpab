"""Tests for REST-endpoint coverage tracking: the `endpoint_hits` matcher, the
metric registration, the `endpoints:` config block, and the venv request-logging
shim."""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

from banter import autoresearch as ar, claude_runner, experiments, results


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
            {"method": "POST", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups"},
            {"method": "GET", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups/fg"},
            {"method": "GET", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups/fg"},  # dup
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
        log = _log({"method": "GET", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups"})
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
        self.assertEqual(results.endpoint_hits(log, [], []), {"whitelist_hits": 0, "blacklist_hits": 0})

    def test_prefix_pattern_does_not_overmatch_deeper_path(self):
        # FV-create POST .../featureview must NOT match the deeper TD-create path
        # .../featureview/{name}/version/{v}/trainingdatasets (end-anchored).
        td_create = {"method": "POST",
                     "path": "/hopsworks-api/api/project/1/featurestores/9/featureview/fv/version/1/trainingdatasets"}
        log = _log(td_create)
        r = results.endpoint_hits(log, [FV_CREATE], [])
        self.assertEqual(r["whitelist_hits"], 0)  # the TD path is not an FV-create

    def test_fg_get_does_not_match_ingestion_subpath(self):
        # FG-get .../featuregroups/[^/]+ must not match .../featuregroups/5/ingestion
        log = _log({"method": "POST", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups/5/ingestion"})
        self.assertEqual(results.endpoint_hits(log, [FG_GET], [])["whitelist_hits"], 0)

    def test_trailing_slash_tolerated(self):
        log = _log({"method": "POST", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups/"})
        self.assertEqual(results.endpoint_hits(log, [FG_CREATE], [])["whitelist_hits"], 1)

    def test_real_sdk_get_paths_match(self):
        # The real SDK GET paths (verified against hopsworks-api source): FV get
        # is .../featureview/{name}/version/{v}; deployment get can be /serving/{id};
        # app get can be /apps/{name}. The optional-group patterns must match these.
        fv_get = "GET /hopsworks-api/api/project/[0-9]+/featurestores/[0-9]+/featureview/[^/]+(?:/version/[0-9]+)?"
        dep_get = "GET /hopsworks-api/api/project/[0-9]+/serving(?:/[0-9]+)?"
        app_get = "GET /hopsworks-api/api/project/[0-9]+/apps(?:/[^/]+)?"
        log = _log(
            {"method": "GET", "path": "/hopsworks-api/api/project/1/featurestores/9/featureview/fv/version/1"},
            {"method": "GET", "path": "/hopsworks-api/api/project/1/serving/5"},
            {"method": "GET", "path": "/hopsworks-api/api/project/1/apps/myapp"},
        )
        self.assertEqual(results.endpoint_hits(log, [fv_get, dep_get, app_get], [])["whitelist_hits"], 3)
        # FV-get must NOT also swallow the deeper TD path.
        td = _log({"method": "GET",
                   "path": "/hopsworks-api/api/project/1/featurestores/9/featureview/fv/version/1/trainingdatasets"})
        self.assertEqual(results.endpoint_hits(td, [fv_get], [])["whitelist_hits"], 0)


class EndpointCoverageTests(unittest.TestCase):
    def test_covered_and_missed_split(self):
        log = _log({"method": "POST", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups"})
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
        log = _log({"method": "POST", "path": "/hopsworks-api/api/project/1/featurestores/9/featuregroups"})
        cov = results.endpoint_coverage(log, [FG_CREATE, FG_GET], ["DELETE /x"])
        hits = results.endpoint_hits(log, [FG_CREATE, FG_GET], ["DELETE /x"])
        self.assertEqual(hits["whitelist_hits"], cov["whitelist_hits"])
        self.assertEqual(hits["blacklist_hits"], cov["blacklist_hits"])


class ResearcherPromptEndpointsTests(unittest.TestCase):
    def test_prompt_lists_targets_and_points_at_coverage_json(self):
        from banter import autoresearch as ar
        root = Path(__file__).resolve().parents[1]
        cfg = ar.load_config(root / "platforms/hopsworks/autoresearch/hopsworks-test-ar.yaml")
        prompt = ar.build_researcher_prompt(
            cfg, root, Path(tempfile.mkdtemp()), "testrun", root / ".venv/bin/banter")
        self.assertIn("Endpoint coverage targets", prompt)
        self.assertIn("endpoint_coverage.json", prompt)          # tells it where to look
        self.assertIn(f"max {len(cfg.endpoints['whitelist'])}", prompt)  # knows the target count
        self.assertNotIn("{endpoints_block}", prompt)            # placeholder consumed

    def test_no_endpoints_no_block(self):
        from banter import autoresearch as ar
        from dataclasses import replace
        root = Path(__file__).resolve().parents[1]
        cfg = ar.load_config(root / "platforms/hopsworks/autoresearch/hopsworks-test-ar.yaml")
        cfg = replace(cfg, endpoints={"whitelist": [], "blacklist": []})
        prompt = ar.build_researcher_prompt(
            cfg, root, Path(tempfile.mkdtemp()), "testrun", root / ".venv/bin/banter")
        self.assertNotIn("Endpoint coverage targets", prompt)
        self.assertNotIn("{endpoints_block}", prompt)


class EndpointSchemaTests(unittest.TestCase):
    def test_metrics_registered(self):
        for m in ("whitelist_hits", "blacklist_hits"):
            self.assertIn(m, results.FIELDS)
            self.assertIn(f"{m}_avg", results.FIELDS)
            self.assertIn(m, experiments.METRICS)
            self.assertIn(m, results.CALL_COUNT_DIRECTIONS)
        self.assertEqual(results.CALL_COUNT_DIRECTIONS["whitelist_hits"], "maximize")
        self.assertEqual(results.CALL_COUNT_DIRECTIONS["blacklist_hits"], "minimize")
        self.assertEqual(results.FIELDS[-1], "run_dir")  # invariant preserved

    def test_row_round_trips_endpoint_metrics(self):
        import csv
        out = Path(tempfile.mkdtemp()) / "results.csv"
        row = results.Row(
            started_at="2026-06-04T00:00:00+00:00", run="1", version="v1",
            platform="hopsworks", interface="cli", skills="none",
            prev_run="", prev_version="", task="t", challenge="c",
            whitelist_hits=5, blacklist_hits=0, run_dir=str(out.parent),
        )
        results.append(out, row)
        got = list(csv.DictReader(out.open()))[0]
        self.assertEqual(got["whitelist_hits"], "5")
        self.assertEqual(got["blacklist_hits"], "0")


class EndpointConfigTests(unittest.TestCase):
    def _write(self, body: str) -> Path:
        p = Path(tempfile.mkdtemp()) / "cfg.yaml"
        p.write_text(body)
        return p

    def test_endpoints_parsed(self):
        cfg = ar.load_config(self._write(
            """
improve: interface
skills: none
challenges: [aerial-cactus-identification]
interfaces: [{platform: hopsworks, interface: cli}]
goals: [score, {metric: whitelist_hits, direction: maximize}]
endpoints:
  whitelist: ['POST /a', 'GET /b']
  blacklist: ['DELETE /c']
budget: {max_increments: 1}
"""
        ))
        self.assertEqual(cfg.endpoints["whitelist"], ["POST /a", "GET /b"])
        self.assertEqual(cfg.endpoints["blacklist"], ["DELETE /c"])

    def test_scalar_whitelist_is_one_pattern_not_chars(self):
        cfg = ar.load_config(self._write(
            """
improve: interface
skills: none
challenges: [aerial-cactus-identification]
interfaces: [{platform: hopsworks, interface: cli}]
goals: [score]
endpoints:
  whitelist: 'POST /a'
budget: {max_increments: 1}
"""
        ))
        self.assertEqual(cfg.endpoints["whitelist"], ["POST /a"])  # NOT ['P','O','S',...]

    def test_endpoints_default_empty(self):
        cfg = ar.load_config(self._write(
            """
improve: interface
skills: none
challenges: [aerial-cactus-identification]
interfaces: [{platform: hopsworks, interface: cli}]
goals: [score]
budget: {max_increments: 1}
"""
        ))
        self.assertEqual(cfg.endpoints, {"whitelist": [], "blacklist": []})


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
        os.environ["BANTER_API_LOG"] = str(log)
        try:
            exec(claude_runner._API_LOG_SHIM, {})
            req = types.SimpleNamespace(
                method="post",
                url="https://h:443/hopsworks-api/api/project/1/featurestores/9/featuregroups?x=1",
            )
            self.assertEqual(fake.Session().send(req), "RESP")  # original still called
            rec = json.loads(log.read_text().splitlines()[0])
            self.assertEqual(rec["method"], "POST")
            self.assertEqual(rec["path"], "/hopsworks-api/api/project/1/featurestores/9/featuregroups")
        finally:
            if old is not None:
                sys.modules["requests"] = old
            else:
                sys.modules.pop("requests", None)
            os.environ.pop("BANTER_API_LOG", None)


if __name__ == "__main__":
    unittest.main()
