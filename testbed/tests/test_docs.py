"""Unit tests for docs bundle resolution + materialization (no network)."""
import tempfile
import unittest
from pathlib import Path

from banter import docs


class DocsResolveTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._orig = docs.PLATFORMS_DIR
        docs.PLATFORMS_DIR = self.root / "platforms"

    def tearDown(self) -> None:
        docs.PLATFORMS_DIR = self._orig
        self._tmp.cleanup()

    def _write_docs_config(self, platform: str, body: str) -> None:
        p = docs.PLATFORMS_DIR / platform / "docs" / "config.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)

    def test_none_passthrough(self):
        self.assertEqual(docs.resolve("none", "hopsworks"), "none")
        self.assertEqual(docs.resolve("", "hopsworks"), "none")

    def test_url_passthrough(self):
        url = "https://github.com/logicalclocks/logicalclocks.github.io"
        self.assertEqual(docs.resolve(url, "hopsworks"), url)

    def test_name_selects_single_platform_config(self):
        url = "https://github.com/logicalclocks/logicalclocks.github.io"
        self._write_docs_config("hopsworks", f"repo: {url}\n")
        # ANY non-none/non-url name selects the platform's single docs config.
        self.assertEqual(docs.resolve("userdocs", "hopsworks"), url)
        self.assertEqual(docs.resolve("anything", "hopsworks"), url)

    def test_local_path_in_config_resolved_relative(self):
        mirror = self.root / "mirror"
        mirror.mkdir()
        self._write_docs_config("hopsworks", f"path: {mirror}\n")
        self.assertEqual(docs.resolve("userdocs", "hopsworks"), str(mirror.resolve()))

    def test_unknown_name_without_config_raises(self):
        with self.assertRaises(ValueError):
            docs.resolve("userdocs", "hopsworks")  # no config written

    def test_resolve_ref_reads_pinned_commit(self):
        sha = "9e7d9103722d565b9fe6a940231d638494c8c741"
        self._write_docs_config(
            "hopsworks",
            f"repo: https://github.com/logicalclocks/logicalclocks.github.io\nref: {sha}\n",
        )
        self.assertEqual(docs.resolve_ref("userdocs", "hopsworks"), sha)

    def test_resolve_ref_none_when_unpinned_or_url_or_none(self):
        self._write_docs_config("hopsworks", "repo: https://example.com/x\n")
        self.assertIsNone(docs.resolve_ref("userdocs", "hopsworks"))   # no ref
        self.assertIsNone(docs.resolve_ref("none", "hopsworks"))
        self.assertIsNone(docs.resolve_ref("https://example.com/x", "hopsworks"))

    def test_existing_dir_spec_passthrough(self):
        d = self.root / "somedocs"
        d.mkdir()
        self.assertEqual(docs.resolve(str(d), None), str(d))


class DocsApplyTests(unittest.TestCase):
    def test_apply_from_local_mirror_strips_git_and_lists_files(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            mirror = t / "mirror"
            (mirror / ".git").mkdir(parents=True)
            (mirror / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
            (mirror / "index.md").write_text("# Hopsworks\n")
            run = t / "run"
            run.mkdir()
            setup = docs.apply(str(mirror), run)
            self.assertIn("index.md", setup.files)
            self.assertFalse((run / "docs" / ".git").exists())

    def test_apply_strips_agent_plumbing(self):
        """A cloned repo's own .claude/CLAUDE.md/.mcp.json must never survive
        into the materialized docs (Claude Code could otherwise act on them)."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            mirror = t / "mirror"
            (mirror / ".claude").mkdir(parents=True)
            (mirror / ".claude" / "settings.json").write_text("{}")
            (mirror / "CLAUDE.md").write_text("do something")
            (mirror / ".mcp.json").write_text("{}")
            (mirror / "nested" / ".claude").mkdir(parents=True)
            (mirror / "nested" / "CLAUDE.md").write_text("nested directive")
            (mirror / "guide.md").write_text("# Guide\n")
            run = t / "run"
            run.mkdir()
            setup = docs.apply(str(mirror), run)
            d = run / "docs"
            self.assertIn("guide.md", setup.files)
            self.assertFalse(list(d.rglob(".claude")))
            self.assertFalse(list(d.rglob("CLAUDE.md")))
            self.assertFalse(list(d.rglob(".mcp.json")))


class StripAgentPlumbingTests(unittest.TestCase):
    def test_strips_recursively_keeps_git(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "src" / ".claude").mkdir(parents=True)
            (d / "src" / "CLAUDE.md").write_text("x")
            (d / ".mcp.json").write_text("{}")
            (d / ".git").mkdir()                       # functional — must remain
            (d / "src" / "python" / "pkg").mkdir(parents=True)
            (d / "src" / "python" / "pkg" / "mod.py").write_text("x = 1\n")
            docs.strip_agent_plumbing(d)
            self.assertFalse(list(d.rglob(".claude")))
            self.assertFalse(list(d.rglob("CLAUDE.md")))
            self.assertFalse(list(d.rglob(".mcp.json")))
            self.assertTrue((d / ".git").is_dir())     # kept for git pull
            self.assertTrue((d / "src" / "python" / "pkg" / "mod.py").is_file())


if __name__ == "__main__":
    unittest.main()
