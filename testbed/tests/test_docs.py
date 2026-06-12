"""Unit tests for stripping vendored agent instructions from cloned repos."""
import tempfile
import unittest
from pathlib import Path

from banter import docs


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
            self.assertTrue((d / ".git").is_dir())     # kept for the pinned checkout
            self.assertTrue((d / "src" / "python" / "pkg" / "mod.py").is_file())

    def test_strips_nested_directives(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "nested" / ".claude").mkdir(parents=True)
            (d / "nested" / ".claude" / "settings.json").write_text("{}")
            (d / "nested" / "CLAUDE.md").write_text("nested directive")
            (d / "guide.md").write_text("# Guide\n")
            docs.strip_agent_plumbing(d)
            self.assertFalse(list(d.rglob(".claude")))
            self.assertFalse(list(d.rglob("CLAUDE.md")))
            self.assertTrue((d / "guide.md").is_file())


if __name__ == "__main__":
    unittest.main()
