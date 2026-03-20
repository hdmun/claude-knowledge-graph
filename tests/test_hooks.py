from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claude_knowledge_graph import hooks


class HooksTests(unittest.TestCase):
    def test_codex_notify_detection_distinguishes_current_and_legacy(self) -> None:
        current = 'notify = ["python3", "-m", "claude_knowledge_graph.qa_logger"]\n'
        legacy = 'notify = ["python3", "-m", "claude_knowledge_graph.codex_hook"]\n'

        self.assertTrue(hooks._is_ckg_codex_notify(current))
        self.assertFalse(hooks._is_ckg_codex_notify(legacy))
        self.assertTrue(hooks._is_legacy_ckg_codex_notify(legacy))

    def test_register_hooks_for_codex_migrates_legacy_notify(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                'notify = ["python3", "-m", "claude_knowledge_graph.codex_hook"]\n'
            )

            old_path = hooks.CODEX_CONFIG_PATH
            try:
                hooks.CODEX_CONFIG_PATH = config_path
                changed = hooks._register_hooks_for_codex()
            finally:
                hooks.CODEX_CONFIG_PATH = old_path

            self.assertTrue(changed)
            text = config_path.read_text()
            self.assertIn("claude_knowledge_graph.qa_logger", text)
            self.assertNotIn("claude_knowledge_graph.codex_hook", text)


if __name__ == "__main__":
    unittest.main()
