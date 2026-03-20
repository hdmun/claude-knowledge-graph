#!/usr/bin/env python3
from __future__ import annotations

"""Hook handler for Claude Code, Gemini CLI, and Codex CLI.

Reads hook JSON from stdin (Claude/Gemini) or argv (Codex) and captures
prompts and Q&A pairs to queue.
Designed to be fast (file I/O only, no blocking calls).

Usage as hook: python3 -m claude_knowledge_graph.qa_logger [json_payload]
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

from claude_knowledge_graph.config import DATA_DIR, QUEUE_DIR, LOGS_DIR
from claude_knowledge_graph.project_context import project_metadata, safe_session_token


def log(msg: str) -> None:
    """Append a log line with timestamp."""
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / "qa_logger.log"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def _prompt_file_path(data: dict) -> Path:
    meta = project_metadata(
        data.get("cwd", ""),
        data.get("source_platform", "unknown"),
    )
    project_dir = QUEUE_DIR / meta["project_slug"]
    project_dir.mkdir(parents=True, exist_ok=True)
    session_token = safe_session_token(data.get("session_id", "unknown"))
    return project_dir / f"{meta['source_platform']}_{session_token}_prompt.json"


def _qa_output_path(data: dict) -> Path:
    meta = project_metadata(
        data.get("cwd", ""),
        data.get("source_platform", "unknown"),
    )
    project_dir = QUEUE_DIR / meta["project_slug"]
    project_dir.mkdir(parents=True, exist_ok=True)
    session_token = safe_session_token(data.get("session_id", "unknown"))
    ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_dir / f"{ts_slug}_{meta['source_platform']}_{session_token}.json"


def handle_prompt_submit(data: dict) -> None:
    """Save prompt to a temporary file keyed by session_id."""
    meta = project_metadata(
        data.get("cwd", ""),
        data.get("source_platform", "unknown"),
    )
    data = {**data, **meta}
    session_id = data.get("session_id", "unknown")
    prompt = data.get("prompt", "")
    cwd = data.get("cwd", "")
    timestamp = datetime.now().isoformat()

    if not prompt.strip():
        log(f"Empty prompt from session {session_id}, skipping")
        return

    prompt_file = _prompt_file_path(data)
    entry = {
        "session_id": session_id,
        "timestamp": timestamp,
        "cwd": cwd,
        "prompt": prompt,
        "project_root": data.get("project_root", ""),
        "project_slug": data.get("project_slug", ""),
        "project_name": data.get("project_name", ""),
        "source_platform": data.get("source_platform", "unknown"),
    }

    # Append to list of prompts for this session
    existing = []
    if prompt_file.exists():
        try:
            existing = json.loads(prompt_file.read_text())
            if isinstance(existing, dict):
                existing = [existing]
        except (json.JSONDecodeError, Exception):
            existing = []

    existing.append(entry)
    prompt_file.write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    log(f"Saved prompt for session {session_id} (#{len(existing)})")


MAX_WRITE_CONTENT_CHARS = 2000  # Max chars of Write content to inline


def _is_user_prompt(msg: dict) -> bool:
    """Check if a message is a user prompt (not a tool_result)."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content", "")
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "text"
            for b in content
        ) and not any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False


def _extract_assistant_parts(msg: dict) -> list[str]:
    """Extract text and Write tool content from an assistant message."""
    parts: list[str] = []
    content = msg.get("content", "")
    if isinstance(content, str):
        if content.strip():
            parts.append(content.strip())
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                if isinstance(block, str) and block.strip():
                    parts.append(block.strip())
                continue
            if block.get("type") == "text":
                text = block.get("text", "").strip()
                if text:
                    parts.append(text)
            elif block.get("type") == "tool_use" and block.get("name") == "Write":
                inp = block.get("input", {})
                fpath = inp.get("file_path", "")
                code = inp.get("content", "")
                if fpath and code:
                    truncated = code[:MAX_WRITE_CONTENT_CHARS]
                    if len(code) > MAX_WRITE_CONTENT_CHARS:
                        truncated += "\n... (truncated)"
                    parts.append(f"[Created file: {fpath}]\n```\n{truncated}\n```")
    return parts


def extract_full_response(transcript_path: str) -> str:
    """Extract assistant response for the last turn from transcript JSONL.

    Finds the last user prompt message (not tool_result), then collects
    all assistant text blocks and Write tool content after it.
    """
    if not transcript_path:
        return ""
    try:
        # Read all entries
        entries: list[dict] = []
        with open(transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))

        # Find the index of the last user prompt
        last_prompt_idx = -1
        for i, entry in enumerate(entries):
            msg = entry.get("message", entry)
            if _is_user_prompt(msg):
                last_prompt_idx = i

        if last_prompt_idx < 0:
            return ""

        # Collect assistant parts after the last user prompt
        parts: list[str] = []
        for entry in entries[last_prompt_idx + 1:]:
            msg = entry.get("message", entry)
            if msg.get("role") == "assistant":
                parts.extend(_extract_assistant_parts(msg))

        return "\n\n".join(parts) if parts else ""
    except Exception as e:
        log(f"Failed to read transcript: {e}")
        return ""


def handle_stop(data: dict) -> None:
    """Merge assistant message with prompt, create Q&A pair."""
    meta = project_metadata(
        data.get("cwd", ""),
        data.get("source_platform", "unknown"),
    )
    data = {**data, **meta}
    session_id = data.get("session_id", "unknown")
    stop_hook_active = data.get("stop_hook_active", False)

    # Prevent infinite loops
    if stop_hook_active:
        log(f"stop_hook_active=True for session {session_id}, skipping")
        return

    transcript_path = data.get("transcript_path", "")
    response = data.get("response", "")
    if not response:
        response = data.get("prompt_response", "")
    if not response:
        response = extract_full_response(transcript_path)
    if not response:
        response = data.get("last_assistant_message", "")
    cwd = data.get("cwd", "")
    timestamp = datetime.now().isoformat()
    direct_prompt = data.get("prompt", "")

    prompt_file = _prompt_file_path(data)

    if not prompt_file.exists():
        if direct_prompt.strip():
            log(f"No prompt file for session {session_id}, using direct prompt from hook payload")
        else:
            log(f"No prompt file for session {session_id}, saving response-only")
        qa_entry = {
            "session_id": session_id,
            "timestamp": timestamp,
            "cwd": cwd,
            "prompt": direct_prompt,
            "response": response,
            "status": "pending",
            "transcript_path": transcript_path,
            "project_root": data.get("project_root", ""),
            "project_slug": data.get("project_slug", ""),
            "project_name": data.get("project_name", ""),
            "source_platform": data.get("source_platform", "unknown"),
        }
        out_file = _qa_output_path(data)
        out_file.write_text(json.dumps(qa_entry, ensure_ascii=False, indent=2))
        return

    # Read the most recent prompt
    try:
        prompts = json.loads(prompt_file.read_text())
        if isinstance(prompts, dict):
            prompts = [prompts]
    except (json.JSONDecodeError, Exception):
        log(f"Failed to read prompt file for session {session_id}")
        prompts = []

    if not prompts:
        log(f"Empty prompts list for session {session_id}")
        prompt_file.unlink(missing_ok=True)
        return

    # Take the last prompt entry
    last_prompt = prompts[-1]

    # Create the Q&A pair
    qa_entry = {
        "session_id": session_id,
        "timestamp": last_prompt.get("timestamp", timestamp),
        "cwd": last_prompt.get("cwd", cwd),
        "prompt": last_prompt.get("prompt", direct_prompt),
        "response": response,
        "status": "pending",
        "transcript_path": transcript_path,
        "project_root": last_prompt.get("project_root", data.get("project_root", "")),
        "project_slug": last_prompt.get("project_slug", data.get("project_slug", "")),
        "project_name": last_prompt.get("project_name", data.get("project_name", "")),
        "source_platform": last_prompt.get("source_platform", data.get("source_platform", "unknown")),
    }

    out_file = _qa_output_path(data)
    out_file.write_text(json.dumps(qa_entry, ensure_ascii=False, indent=2))
    log(f"Created Q&A pair: {out_file.name}")

    # Clean up prompt file
    prompt_file.unlink(missing_ok=True)

    # Trigger background processing
    trigger_processor()


def trigger_processor() -> None:
    """Launch qwen_processor in background if not already running.

    Uses a lock file to prevent duplicate runs.
    """
    import subprocess
    import fcntl

    lock_file = DATA_DIR / "processor.lock"
    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_WRONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            log("Processor already running, skipping trigger")
            return

        # Release the lock — the subprocess will acquire its own
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

        # Launch processor in background
        subprocess.Popen(
            [sys.executable, "-m", "claude_knowledge_graph.qwen_processor"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        log("Triggered background processor")
    except Exception as e:
        log(f"Failed to trigger processor: {e}")


def _extract_content_text(content) -> str:
    """Extract text from a message content field (string or structured blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "input_text":
                    parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content) if content else ""


def normalize_hook_payload(data: dict) -> dict:
    """Normalize Claude Code, Gemini CLI, and Codex CLI hook payloads."""
    event = data.get("hook_event_name", "")
    codex_type = data.get("type", "")
    normalized = {
        **data,
        "source_platform": "unknown",
        "normalized_event": "",
        "response": data.get("response", ""),
    }

    if event in {"UserPromptSubmit", "Stop"}:
        normalized["source_platform"] = "claude"
        normalized["normalized_event"] = (
            "prompt_submitted" if event == "UserPromptSubmit" else "turn_completed"
        )
    elif event in {"BeforeAgent", "AfterAgent"}:
        normalized["source_platform"] = "gemini"
        normalized["normalized_event"] = (
            "prompt_submitted" if event == "BeforeAgent" else "turn_completed"
        )
        if not normalized.get("response"):
            normalized["response"] = data.get("prompt_response", "")
    elif codex_type == "agent-turn-complete":
        normalized["source_platform"] = "codex"
        normalized["normalized_event"] = "turn_completed"
        normalized["session_id"] = data.get("thread-id", "unknown")
        # Extract prompt from last user message in input-messages
        input_messages = data.get("input-messages", [])
        last_user_prompt = ""
        for msg in input_messages:
            if msg.get("role") == "user":
                last_user_prompt = _extract_content_text(msg.get("content", ""))
        normalized["prompt"] = last_user_prompt
        # Extract response
        normalized["response"] = _extract_content_text(
            data.get("last-assistant-message", "")
        )

    return normalized


def exit_success(source_platform: str) -> None:
    """Exit successfully, emitting valid JSON for Gemini hooks."""
    if source_platform == "gemini":
        sys.stdout.write("{}")
    sys.exit(0)


def main() -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    source_platform = "unknown"

    try:
        if len(sys.argv) > 1:
            raw = sys.argv[1]       # Codex: JSON via argv
        else:
            raw = sys.stdin.read()  # Claude/Gemini: JSON via stdin
        data = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        log(f"Failed to parse input: {e}")
        exit_success(source_platform)

    data = normalize_hook_payload(data)
    source_platform = data.get("source_platform", "unknown")
    event = data.get("hook_event_name", "")
    log(f"Received event: {event} (session: {data.get('session_id', 'unknown')})")

    if data.get("normalized_event") == "prompt_submitted":
        handle_prompt_submit(data)
    elif data.get("normalized_event") == "turn_completed":
        handle_stop(data)
    else:
        log(f"Unknown event: {event}")

    exit_success(source_platform)


if __name__ == "__main__":
    main()
