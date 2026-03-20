from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claude_knowledge_graph import qa_logger
from claude_knowledge_graph.qa_logger import handle_stop, normalize_hook_payload


class QaLoggerTests(unittest.TestCase):
    def test_normalize_codex_payload_with_string_input_messages(self) -> None:
        payload = {
            "type": "agent-turn-complete",
            "thread-id": "019d0a8b-f6b9-7f61-b6c9-2050a0dbfbcd",
            "turn-id": "019d0ac4-6465-7972-9167-93ab7e65c2ee",
            "cwd": "/Users/hdmun-macair/repo/fresco_dev_flutter/fresco_flutter",
            "client": "codex-tui",
            "input-messages": [
                "첫 번째 요청",
                "두 번째 요청",
                "Implement the plan.",
            ],
            "last-assistant-message": "리뷰 기반 수정 반영을 끝냈습니다.",
        }

        normalized = normalize_hook_payload(payload)

        self.assertEqual(normalized["source_platform"], "codex")
        self.assertEqual(normalized["normalized_event"], "turn_completed")
        self.assertEqual(normalized["session_id"], payload["thread-id"])
        self.assertEqual(normalized["prompt"], "Implement the plan.")
        self.assertEqual(normalized["response"], "리뷰 기반 수정 반영을 끝냈습니다.")

    def test_normalize_codex_payload_ignores_non_string_input_messages(self) -> None:
        payload = {
            "type": "agent-turn-complete",
            "thread-id": "thread-123",
            "input-messages": [
                {"role": "user", "content": "ignored structured message"},
                "",
                "final user prompt",
            ],
            "last-assistant-message": "assistant reply",
        }

        normalized = normalize_hook_payload(payload)

        self.assertEqual(normalized["prompt"], "final user prompt")
        self.assertEqual(normalized["response"], "assistant reply")

    def test_handle_stop_triggers_processor_for_direct_prompt(self) -> None:
        payload = {
            "session_id": "codex-session-1",
            "cwd": "/Users/hdmun-macair/repo/fresco_dev_flutter/fresco_flutter",
            "prompt": "Implement the plan.",
            "response": "done",
            "source_platform": "codex",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            old_queue_dir = qa_logger.QUEUE_DIR
            try:
                qa_logger.QUEUE_DIR = Path(tmpdir)
                with patch("claude_knowledge_graph.qa_logger.trigger_processor") as trigger:
                    handle_stop(payload)
                    trigger.assert_called_once()
            finally:
                qa_logger.QUEUE_DIR = old_queue_dir

            files = list(Path(tmpdir).rglob("*.json"))
            self.assertEqual(len(files), 1)
            written = json.loads(files[0].read_text())
            self.assertEqual(written["status"], "pending")
            self.assertEqual(written["prompt"], "Implement the plan.")
            self.assertEqual(written["response"], "done")
            self.assertEqual(written["source_platform"], "codex")

    def test_handle_stop_triggers_processor_for_response_only_direct_write(self) -> None:
        payload = {
            "session_id": "codex-session-2",
            "cwd": "/Users/hdmun-macair/repo/fresco_dev_flutter/fresco_flutter",
            "prompt": "",
            "response": "assistant only",
            "source_platform": "codex",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            old_queue_dir = qa_logger.QUEUE_DIR
            try:
                qa_logger.QUEUE_DIR = Path(tmpdir)
                with patch("claude_knowledge_graph.qa_logger.trigger_processor") as trigger:
                    handle_stop(payload)
                    trigger.assert_called_once()
            finally:
                qa_logger.QUEUE_DIR = old_queue_dir

            files = list(Path(tmpdir).rglob("*.json"))
            self.assertEqual(len(files), 1)
            written = json.loads(files[0].read_text())
            self.assertEqual(written["prompt"], "")
            self.assertEqual(written["response"], "assistant only")


if __name__ == "__main__":
    unittest.main()
