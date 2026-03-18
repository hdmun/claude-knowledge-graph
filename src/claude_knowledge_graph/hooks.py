from __future__ import annotations

"""Manage Claude Code and Gemini CLI hook registration.

Reads/writes CLI settings files to add or remove ckg hooks while preserving
existing user hooks.
"""

import json
from pathlib import Path

CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
GEMINI_SETTINGS_PATH = Path.home() / ".gemini" / "settings.json"

# Marker to identify hooks managed by ckg
CKG_MARKER = "claude-knowledge-graph"

HOOK_COMMAND = "python3 -m claude_knowledge_graph.qa_logger"

CLAUDE_HOOKS_CONFIG = {
    "UserPromptSubmit": {
        "matcher": "",
        "hooks": [
            {
                "type": "command",
                "command": HOOK_COMMAND,
                "description": f"[{CKG_MARKER}] Capture user prompts",
            }
        ],
    },
    "Stop": {
        "matcher": "",
        "hooks": [
            {
                "type": "command",
                "command": HOOK_COMMAND,
                "description": f"[{CKG_MARKER}] Capture Q&A pairs",
            }
        ],
    },
}

GEMINI_HOOKS_CONFIG = {
    "BeforeAgent": {
        "hooks": [
            {
                "type": "command",
                "command": HOOK_COMMAND,
                "name": "ckg-capture-prompt",
                "timeout": 5000,
                "description": f"[{CKG_MARKER}] Capture user prompts",
            }
        ],
    },
    "AfterAgent": {
        "hooks": [
            {
                "type": "command",
                "command": HOOK_COMMAND,
                "name": "ckg-capture-response",
                "timeout": 5000,
                "description": f"[{CKG_MARKER}] Capture Q&A pairs",
            }
        ],
    },
}

PLATFORMS = ("claude", "gemini")


def _load_settings(path: Path) -> dict:
    """Load CLI settings, creating an empty structure if needed."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_settings(path: Path, settings: dict) -> None:
    """Save CLI settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")


def _platform_path(platform: str) -> Path:
    if platform == "claude":
        return CLAUDE_SETTINGS_PATH
    if platform == "gemini":
        return GEMINI_SETTINGS_PATH
    raise ValueError(f"Unsupported hooks platform: {platform}")


def _platform_hooks_config(platform: str) -> dict[str, dict]:
    if platform == "claude":
        return CLAUDE_HOOKS_CONFIG
    if platform == "gemini":
        return GEMINI_HOOKS_CONFIG
    raise ValueError(f"Unsupported hooks platform: {platform}")


def _normalize_platforms(platforms: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if platforms is None:
        return ("claude",)
    normalized = tuple(dict.fromkeys(platforms))
    invalid = [platform for platform in normalized if platform not in PLATFORMS]
    if invalid:
        raise ValueError(f"Unsupported hooks platform(s): {', '.join(invalid)}")
    return normalized


def _is_ckg_matcher_group(group: dict) -> bool:
    """Check if a matcher group or hook belongs to ckg.

    Also detects legacy flat-format hooks ({type, command, description})
    so they can be cleaned up during unregister.
    """
    # New format: matcher group with hooks array
    for hook in group.get("hooks", []):
        desc = hook.get("description", "")
        cmd = hook.get("command", "")
        if CKG_MARKER in desc or "claude_knowledge_graph" in cmd:
            return True
    # Legacy flat format: {type, command, description} without hooks array
    if "hooks" not in group:
        desc = group.get("description", "")
        cmd = group.get("command", "")
        if CKG_MARKER in desc or "claude_knowledge_graph" in cmd:
            return True
    return False


def _register_hooks_for_platform(platform: str) -> bool:
    """Register ckg hooks for one platform.

    Returns True if hooks were added, False if already present.
    """
    settings_path = _platform_path(platform)
    hooks_config = _platform_hooks_config(platform)
    settings = _load_settings(settings_path)
    hooks = settings.setdefault("hooks", {})
    changed = False

    for event_name, hook_config in hooks_config.items():
        event_hooks = hooks.setdefault(event_name, [])

        # Check if ckg hook already exists for this event
        already_registered = any(_is_ckg_matcher_group(h) for h in event_hooks)
        if not already_registered:
            event_hooks.append(hook_config)
            changed = True

    if changed:
        _save_settings(settings_path, settings)

    return changed


def _unregister_hooks_for_platform(platform: str) -> bool:
    """Remove ckg hooks from one platform settings file.

    Returns True if hooks were removed, False if none found.
    """
    settings_path = _platform_path(platform)
    settings = _load_settings(settings_path)
    hooks = settings.get("hooks", {})
    changed = False

    for event_name in list(hooks.keys()):
        original = hooks[event_name]
        filtered = [h for h in original if not _is_ckg_matcher_group(h)]
        if len(filtered) != len(original):
            hooks[event_name] = filtered
            changed = True
        # Clean up empty arrays
        if not hooks[event_name]:
            del hooks[event_name]

    if changed:
        if not hooks:
            settings.pop("hooks", None)
        _save_settings(settings_path, settings)

    return changed


def _check_hooks_for_platform(platform: str) -> dict[str, bool]:
    """Check which ckg hooks are registered for one platform.

    Returns dict mapping event name to registration status.
    """
    settings = _load_settings(_platform_path(platform))
    hooks_config = _platform_hooks_config(platform)
    hooks = settings.get("hooks", {})

    status = {}
    for event_name in hooks_config:
        event_hooks = hooks.get(event_name, [])
        status[event_name] = any(_is_ckg_matcher_group(h) for h in event_hooks)

    return status


def register_hooks(platforms: tuple[str, ...] | list[str] | None = None) -> bool:
    """Register ckg hooks for one or more platforms."""
    changed = False
    for platform in _normalize_platforms(platforms):
        if _register_hooks_for_platform(platform):
            changed = True
    return changed


def unregister_hooks(platforms: tuple[str, ...] | list[str] | None = None) -> bool:
    """Unregister ckg hooks for one or more platforms."""
    changed = False
    for platform in _normalize_platforms(platforms):
        if _unregister_hooks_for_platform(platform):
            changed = True
    return changed


def check_hooks(platforms: tuple[str, ...] | list[str] | None = None) -> dict[str, dict[str, bool]]:
    """Check ckg hook registration status by platform."""
    return {
        platform: _check_hooks_for_platform(platform)
        for platform in _normalize_platforms(platforms)
    }
