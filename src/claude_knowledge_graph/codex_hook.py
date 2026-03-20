from __future__ import annotations

"""Backward-compatible Codex hook entrypoint.

Older Codex installs may still point notify to ``claude_knowledge_graph.codex_hook``.
Delegate to qa_logger so those configs keep working until they are rewritten.
"""

from claude_knowledge_graph.qa_logger import main


if __name__ == "__main__":
    main()
