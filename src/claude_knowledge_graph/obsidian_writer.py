#!/usr/bin/env python3
from __future__ import annotations

"""Obsidian note writer: converts processed Q&A pairs to knowledge graph notes.

Writes markdown files directly to the Obsidian vault directory.
Creates session notes, daily indexes, concept notes, and a Map of Content (MOC).
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from claude_knowledge_graph.config import (
    CONCEPTS_DIR,
    DAILY_DIR,
    KNOWLEDGE_GRAPH_DIR,
    LOGS_DIR,
    MOC_PATH,
    PROJECTS_DIR,
    PROCESSED_DIR,
)
from claude_knowledge_graph.project_context import project_metadata

LOG_FILE = LOGS_DIR / "obsidian_writer.log"


def log(msg: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename."""
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    name = name.strip()
    return name[:100] if name else "untitled"


def truncate(text: str, max_len: int = 500) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def get_processed_files() -> list[Path]:
    """Get all processed Q&A pair files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(PROCESSED_DIR.rglob("*.json"))


def _qa_project_meta(qa: dict) -> dict[str, str]:
    meta = project_metadata(
        qa.get("project_root") or qa.get("cwd", ""),
        qa.get("source_platform", "unknown"),
    )
    meta["project_root"] = qa.get("project_root") or meta["project_root"]
    meta["project_slug"] = qa.get("project_slug") or meta["project_slug"]
    meta["project_name"] = qa.get("project_name") or meta["project_name"]
    meta["source_platform"] = qa.get("source_platform") or meta["source_platform"]
    return meta


def _qa_datetime(qa: dict) -> datetime | None:
    ts = qa.get("timestamp", "")
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _qa_date_str(qa: dict) -> str:
    dt = _qa_datetime(qa)
    if dt:
        return dt.strftime("%Y-%m-%d")
    return datetime.now().strftime("%Y-%m-%d")


def _qa_time_str(qa: dict) -> str:
    dt = _qa_datetime(qa)
    if dt:
        return dt.strftime("%H:%M")
    return "00:00"


def session_key(qa: dict) -> str:
    """Return a stable identifier for a Q&A item."""
    meta = _qa_project_meta(qa)
    return "::".join(
        [
            meta["project_slug"],
            meta["source_platform"],
            str(qa.get("session_id", "")),
            str(qa.get("timestamp", "")),
        ]
    )


def _session_identity_from_qa(qa: dict) -> tuple[str, str, str, str, str, str]:
    qwen = qa.get("qwen_result", {})
    return (
        _qa_project_meta(qa)["project_slug"],
        _qa_project_meta(qa)["source_platform"],
        str(qa.get("session_id", "")),
        _qa_date_str(qa),
        _qa_time_str(qa),
        qwen.get("title", "Untitled"),
    )


def _session_path(project_slug: str, stem: str) -> Path:
    return PROJECTS_DIR / project_slug / "sessions" / f"{stem}.md"


def _session_target(project_slug: str, stem: str) -> str:
    return f"projects/{project_slug}/sessions/{stem}"


def session_link_target(qa: dict, filename_map: dict[str, str] | None = None) -> str:
    """Return the Obsidian wikilink target for a session note."""
    meta = _qa_project_meta(qa)
    return _session_target(meta["project_slug"], session_filename(qa, filename_map))


def session_link(
    qa: dict,
    alias: str | None = None,
    filename_map: dict[str, str] | None = None,
) -> str:
    """Build a session note wikilink."""
    target = session_link_target(qa, filename_map)
    if alias:
        return f"[[{target}|{alias}]]"
    return f"[[{target}]]"


# ── Step 1: Session filename helper ──


def session_filename(qa: dict, filename_map: dict[str, str] | None = None) -> str:
    """Generate a session note filename from Q&A data."""
    key = session_key(qa)
    if filename_map and key in filename_map:
        return filename_map[key]

    qwen = qa.get("qwen_result", {})
    title = qwen.get("title", "Untitled")
    date_str = _qa_date_str(qa)
    return f"{date_str}_01_{sanitize_filename(title)}"


def _build_sequence_filename(date_str: str, sequence: int, title: str) -> str:
    return f"{date_str}_{sequence:02d}_{sanitize_filename(title)}"


def _extract_sequence_number(stem: str, date_str: str) -> int | None:
    match = re.match(rf"^{re.escape(date_str)}_(\d+)_", stem)
    if not match:
        return None
    return int(match.group(1))


def build_session_filename_map(qa_list: list[dict]) -> dict[str, str]:
    """Assign ordered session filenames per project and date."""
    grouped: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for idx, qa in enumerate(qa_list):
        project_slug = _qa_project_meta(qa)["project_slug"]
        grouped[(project_slug, _qa_date_str(qa))].append((idx, qa))

    filename_map: dict[str, str] = {}
    for (_project_slug, date_str), entries in grouped.items():
        sorted_entries = sorted(
            entries,
            key=lambda item: (
                _qa_datetime(item[1]) is None,
                _qa_datetime(item[1]) or datetime.max,
                item[0],
            ),
        )
        for sequence, (_idx, qa) in enumerate(sorted_entries, start=1):
            title = qa.get("qwen_result", {}).get("title", "Untitled")
            filename_map[session_key(qa)] = _build_sequence_filename(date_str, sequence, title)
    return filename_map


def _parse_frontmatter(text: str) -> dict[str, str]:
    if not text.startswith("---\n"):
        return {}
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if not match:
        return {}

    data: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data


def _scan_existing_session_records() -> list[dict]:
    records: list[dict] = []
    for path in PROJECTS_DIR.glob("*/sessions/*.md"):
        frontmatter = _parse_frontmatter(path.read_text())
        project_slug = frontmatter.get("project_slug") or path.parent.parent.name
        date_str = frontmatter.get("date", datetime.now().strftime("%Y-%m-%d"))
        time_str = frontmatter.get("time", "00:00")
        title = frontmatter.get("title", path.stem)
        source_platform = frontmatter.get("source_platform", "unknown")
        session_id = frontmatter.get("session_id", "")
        try:
            sort_dt = datetime.fromisoformat(f"{date_str}T{time_str}")
        except ValueError:
            sort_dt = None
        records.append(
            {
                "project_slug": project_slug,
                "date_str": date_str,
                "time_str": time_str,
                "title": title,
                "source_platform": source_platform,
                "session_id": session_id,
                "sort_dt": sort_dt,
                "old_stem": path.stem,
                "path": path,
                "identity": (
                    project_slug,
                    source_platform,
                    session_id,
                    date_str,
                    time_str,
                    title,
                ),
            }
        )
    return records


def build_existing_session_rename_map(qa_list: list[dict], filename_map: dict[str, str]) -> dict[str, str]:
    """Map existing session link targets to ordered filenames."""
    records = _scan_existing_session_records()
    current_by_identity = {
        _session_identity_from_qa(qa): session_filename(qa, filename_map) for qa in qa_list
    }
    current_group_sequences: dict[tuple[str, str], set[int]] = defaultdict(set)
    for qa in qa_list:
        project_slug = _qa_project_meta(qa)["project_slug"]
        date_str = _qa_date_str(qa)
        stem = session_filename(qa, filename_map)
        sequence = _extract_sequence_number(stem, date_str)
        if sequence is not None:
            current_group_sequences[(project_slug, date_str)].add(sequence)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["project_slug"], record["date_str"])].append(record)

    rename_map: dict[str, str] = {}
    for (project_slug, date_str), group in grouped.items():
        sorted_group = sorted(
            group,
            key=lambda record: (
                record["sort_dt"] is None,
                record["sort_dt"] or datetime.max,
                record["old_stem"],
            ),
        )
        used_sequences = set(current_group_sequences.get((project_slug, date_str), set()))
        next_sequence = 1
        for sequence, record in enumerate(sorted_group, start=1):
            desired_stem = current_by_identity.get(record["identity"])
            if not desired_stem:
                next_sequence = max(next_sequence, sequence)
                while next_sequence in used_sequences:
                    next_sequence += 1
                desired_stem = _build_sequence_filename(date_str, next_sequence, record["title"])
            desired_sequence = _extract_sequence_number(desired_stem, date_str)
            if desired_sequence is not None:
                used_sequences.add(desired_sequence)
            old_target = _session_target(project_slug, record["old_stem"])
            new_target = _session_target(project_slug, desired_stem)
            if old_target != new_target:
                rename_map[old_target] = new_target

    return rename_map


def rename_existing_session_files(rename_map: dict[str, str]) -> None:
    """Rename existing session files to their new ordered names."""
    if not rename_map:
        return

    temp_moves: list[tuple[Path, Path]] = []
    final_moves: list[tuple[Path, Path]] = []
    for old_target, new_target in rename_map.items():
        old_parts = old_target.split("/")
        new_parts = new_target.split("/")
        old_path = _session_path(old_parts[1], old_parts[-1])
        new_path = _session_path(new_parts[1], new_parts[-1])
        if not old_path.exists() or old_path == new_path:
            continue
        temp_path = old_path.with_name(f"{old_path.stem}.__renaming__.md")
        temp_moves.append((old_path, temp_path))
        final_moves.append((temp_path, new_path))

    for old_path, temp_path in temp_moves:
        old_path.rename(temp_path)
    for temp_path, new_path in final_moves:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.rename(new_path)


def rewrite_session_links(target_map: dict[str, str]) -> None:
    """Rewrite session wikilinks in generated knowledge-graph notes."""
    if not target_map:
        return

    def replace_link(match: re.Match[str]) -> str:
        target = match.group(1)
        alias = match.group(2) or ""
        return f"[[{target_map.get(target, target)}{alias}]]"

    files = list(DAILY_DIR.glob("*.md")) + list(CONCEPTS_DIR.glob("*.md"))
    if MOC_PATH.exists():
        files.append(MOC_PATH)

    pattern = re.compile(r"\[\[([^\]|]+)(\|[^\]]+)?\]\]")
    for path in files:
        text = path.read_text()
        updated = pattern.sub(replace_link, text)
        if updated != text:
            path.write_text(updated)


# ── Step 2: Weighted similarity ──


def compute_similarity(qa1: dict, qa2: dict) -> tuple[float, list[str]]:
    """Compute weighted similarity between two Q&A pairs.

    Returns (score, list of reason strings).
    """
    qwen1 = qa1.get("qwen_result", {})
    qwen2 = qa2.get("qwen_result", {})

    score = 0.0
    reasons: list[str] = []

    # Shared tags: 0.2 per tag, capped at 0.8
    tags1 = set(qwen1.get("tags", []))
    tags2 = set(qwen2.get("tags", []))
    shared_tags = tags1 & tags2
    if shared_tags:
        tag_score = min(len(shared_tags) * 0.2, 0.8)
        score += tag_score
        reasons.append(f"shared tags: {', '.join(sorted(shared_tags))}")

    # Shared concepts: 0.4 per concept, capped at 1.2
    concepts1 = set(qwen1.get("key_concepts", []))
    concepts2 = set(qwen2.get("key_concepts", []))
    shared_concepts = concepts1 & concepts2
    if shared_concepts:
        concept_score = min(len(shared_concepts) * 0.4, 1.2)
        score += concept_score
        reasons.append(f"shared concepts: {', '.join(sorted(shared_concepts))}")

    # Same category: 0.15
    cat1 = qwen1.get("category", "")
    cat2 = qwen2.get("category", "")
    if cat1 and cat2 and cat1 == cat2:
        score += 0.15
        reasons.append(f"same category: {cat1}")

    # Same cwd (project): 0.2
    meta1 = _qa_project_meta(qa1)
    meta2 = _qa_project_meta(qa2)
    if meta1["project_slug"] and meta1["project_slug"] == meta2["project_slug"]:
        score += 0.2
        reasons.append("same project")

    return score, reasons


# ── Step 3: Build session relations ──


def build_session_relations(
    qa_list: list[dict],
    filename_map: dict[str, str] | None = None,
) -> dict[str, list[tuple[dict, float, list[str]]]]:
    """Map each session filename to 'See Also' entries based on similarity.

    Returns {filename: [(other_filename, score, reasons), ...]}.
    Strong matches (>=0.8) are unlimited; moderate (0.6-0.8) capped at 5.
    """
    if len(qa_list) < 2:
        return {}

    # Pre-compute filenames
    targets = [session_link_target(qa, filename_map) for qa in qa_list]

    # Compute all pairwise similarities
    pairs: dict[str, list[tuple[dict, float, list[str]]]] = defaultdict(list)
    for i in range(len(qa_list)):
        for j in range(i + 1, len(qa_list)):
            score, reasons = compute_similarity(qa_list[i], qa_list[j])
            if score >= 0.6:
                pairs[targets[i]].append((qa_list[j], score, reasons))
                pairs[targets[j]].append((qa_list[i], score, reasons))

    # Apply tiered cap: strong (>=0.8) unlimited, moderate (0.6-0.8) max 5
    result: dict[str, list[tuple[dict, float, list[str]]]] = {}
    for target, matches in pairs.items():
        strong = [(m, s, r) for m, s, r in matches if s >= 0.8]
        moderate = sorted(
            [(m, s, r) for m, s, r in matches if s < 0.8],
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        result[target] = sorted(strong + moderate, key=lambda x: x[1], reverse=True)

    return result


# ── Step 4: Write session note ──


def write_session_note(
    qa: dict,
    see_also: list[tuple[dict, float, list[str]]],
    filename_map: dict[str, str] | None = None,
) -> Path:
    """Write an individual session note to sessions/ directory."""
    meta = _qa_project_meta(qa)
    sessions_dir = PROJECTS_DIR / meta["project_slug"] / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    fname = session_filename(qa, filename_map)
    session_file = sessions_dir / f"{fname}.md"

    qwen = qa.get("qwen_result", {})
    title = qwen.get("title", "Untitled")
    summary = qwen.get("summary", "")
    tags = qwen.get("tags", [])
    category = qwen.get("category", "other")
    key_concepts = qwen.get("key_concepts", [])
    prompt = qa.get("prompt", "")
    response = qa.get("response", "")
    session_id = qa.get("session_id", "")
    cwd = meta["cwd"].replace("\\", "/")
    project_root = meta["project_root"].replace("\\", "/")
    project_slug = meta["project_slug"]
    project_name = meta["project_name"]
    source_platform = meta["source_platform"]

    ts = qa.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts)
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M")
        timestamp_str = dt.isoformat()
    except (ValueError, TypeError):
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = "00:00"
        timestamp_str = ""

    tags_yaml = "\n".join(f"  - {t}" for t in tags) if tags else "  - untagged"
    tags_inline = " ".join(f"#{t}" for t in tags) if tags else ""
    concepts_str = ", ".join(f"[[{c}]]" for c in key_concepts) if key_concepts else ""

    # Tool summary data
    tool_summary = qa.get("tool_summary", {})
    files_modified = tool_summary.get("files_modified", [])
    commands_executed = tool_summary.get("commands_executed", [])
    tool_counts = tool_summary.get("tool_counts", {})
    total_tools = sum(tool_counts.values()) if tool_counts else 0

    # Frontmatter extras for tool summary
    tool_frontmatter = ""
    if tool_summary:
        tool_frontmatter = f"\nfiles_modified: {len(files_modified)}\ntools_used: {total_tools}"

    # See Also section
    see_also_lines = ""
    if see_also:
        lines = []
        for other_qa, _score, reasons in see_also:
            other_name = session_filename(other_qa, filename_map)
            reason_str = " — " + "; ".join(reasons) if reasons else ""
            lines.append(f"- {session_link(other_qa, other_name, filename_map)}{reason_str}")
        see_also_lines = "\n## See Also\n\n" + "\n".join(lines) + "\n"

    # Session Activity section
    activity_section = ""
    if tool_summary:
        activity_parts = []
        if files_modified:
            file_lines = "\n".join(f"  - `{f}`" for f in files_modified)
            activity_parts.append(f"### Files Modified ({len(files_modified)})\n{file_lines}")
        if commands_executed:
            cmd_lines = "\n".join(f"  - `{c}`" for c in commands_executed)
            activity_parts.append(f"### Commands Executed ({len(commands_executed)})\n{cmd_lines}")
        if tool_counts:
            counts_str = ", ".join(f"{k}: {v}" for k, v in sorted(tool_counts.items()))
            activity_parts.append(f"**Tool Usage** ({total_tools} total): {counts_str}")
        if activity_parts:
            activity_section = "\n## Session Activity\n\n" + "\n\n".join(activity_parts) + "\n"

    content = f"""---
title: "{title}"
date: {date_str}
time: "{time_str}"
timestamp: "{timestamp_str}"
session_id: {session_id}
source_platform: {source_platform}
category: {category}
tags:
{tags_yaml}
type: session
project_name: "{project_name}"
project_slug: "{project_slug}"
project_root: "{project_root}"
cwd: "{cwd}"{tool_frontmatter}
---

# {title}

**Summary**: {summary}

**Category**: {category}

**Tags**: {tags_inline}

**Key Concepts**: {concepts_str}

**Project**: `{project_name}` (`{project_root}`)
{activity_section}{see_also_lines}
## Conversation

> [!question] Prompt
> {prompt.replace(chr(10), chr(10) + '> ')}

> [!quote] Response
> {response.replace(chr(10), chr(10) + '> ')}
"""
    session_file.write_text(content)
    return session_file


# ── Step 5: Daily entry (index line) ──


def build_daily_entry(qa: dict, filename_map: dict[str, str] | None = None) -> str:
    """Build a one-line index entry linking to the session note."""
    qwen = qa.get("qwen_result", {})
    tags = qwen.get("tags", [])
    category = qwen.get("category", "other")
    meta = _qa_project_meta(qa)

    time_str = _qa_time_str(qa)
    tags_preview = " ".join(f"#{t}" for t in tags[:3]) if tags else ""
    alias = session_filename(qa, filename_map)

    return (
        f"- [{time_str}] {session_link(qa, alias, filename_map)} — "
        f"{meta['project_name']}, {category}, {tags_preview}"
    )


# ── Step 6: Daily note with "Today's Concepts" ──


def write_daily_note(
    date_str: str, entries: list[str], day_concepts: list[str] | None = None
) -> Path:
    """Write or append to a daily note (index format)."""
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    daily_file = DAILY_DIR / f"{date_str}.md"

    entries_block = "\n".join(entries)

    concepts_section = ""
    if day_concepts:
        unique = sorted(set(day_concepts))
        concepts_links = ", ".join(f"[[{c}]]" for c in unique)
        concepts_section = f"\n## Today's Concepts\n\n{concepts_links}\n"

    if daily_file.exists():
        existing = daily_file.read_text()
        # Append new entries before Today's Concepts or at the end
        if "## Today's Concepts" in existing:
            existing = re.sub(
                r"## Today's Concepts\n\n.*",
                "",
                existing,
                flags=re.DOTALL,
            )
        new_content = existing.rstrip() + "\n" + entries_block + "\n" + concepts_section
        daily_file.write_text(new_content)
    else:
        frontmatter = f"""---
title: {date_str} AI Conversation Log
date: {date_str}
tags:
  - ai-log
  - knowledge-graph
type: daily-ai-log
---

# {date_str} AI Conversation Log

"""
        daily_file.write_text(frontmatter + entries_block + "\n" + concepts_section)

    return daily_file


# ── Step 7: Concept relations with metadata ──


def build_concept_relations(
    concept_refs: dict[str, list[dict]],
) -> dict[str, dict[str, dict]]:
    """Build concept-to-concept relations with co-occurrence counts and shared tags.

    Returns {concept: {related_concept: {"co_occurred": int, "shared_tags": set}}}.
    """
    cooccurrence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Co-occurrence: concepts appearing in the same Q&A pair
    seen_pairs: dict[str, list[str]] = defaultdict(list)
    for concept, refs in concept_refs.items():
        for ref in refs:
            pair_id = (
                f"{ref.get('project_slug', '')}_"
                f"{ref.get('source_platform', '')}_"
                f"{ref.get('session_id', '')}_"
                f"{ref.get('timestamp', '')}"
            )
            seen_pairs[pair_id].append(concept)

    for pair_id, concepts in seen_pairs.items():
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1 :]:
                cooccurrence[c1][c2] += 1
                cooccurrence[c2][c1] += 1

    # Collect tags per concept
    concept_tags: dict[str, set[str]] = {}
    for concept, refs in concept_refs.items():
        tags: set[str] = set()
        for ref in refs:
            qwen = ref.get("qwen_result", {})
            tags.update(qwen.get("tags", []))
        concept_tags[concept] = tags

    # Build final relations dict with metadata
    all_concepts = list(concept_refs.keys())
    relations: dict[str, dict[str, dict]] = defaultdict(dict)

    for i, c1 in enumerate(all_concepts):
        for c2 in all_concepts[i + 1 :]:
            co_count = cooccurrence.get(c1, {}).get(c2, 0)
            shared = concept_tags.get(c1, set()) & concept_tags.get(c2, set())

            # Include if co-occurred or share 2+ tags
            if co_count > 0 or len(shared) >= 2:
                meta = {"co_occurred": co_count, "shared_tags": shared}
                relations[c1][c2] = meta
                relations[c2][c1] = meta

    return dict(relations)


# ── Step 8: Concept note with grouped refs and annotated relations ──


def write_concept_note(
    concept: str,
    references: list[dict],
    related_concepts: dict[str, dict] | None = None,
    filename_map: dict[str, str] | None = None,
) -> Path:
    """Create or update a concept note with category-grouped refs and annotated relations."""
    CONCEPTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(concept)
    concept_file = CONCEPTS_DIR / f"{safe_name}.md"

    # Group references by category, link to session notes
    by_category: dict[str, list[str]] = defaultdict(list)
    for ref in references:
        qwen = ref.get("qwen_result", {})
        cat = qwen.get("category", "other")
        fname = session_filename(ref, filename_map)
        link = f"- {session_link(ref, fname, filename_map)}"
        by_category[cat].append(link)

    ref_section_parts = []
    for cat in sorted(by_category.keys()):
        ref_section_parts.append(f"### {cat}\n\n" + "\n".join(by_category[cat]))
    ref_section = "\n\n".join(ref_section_parts)

    # Build annotated related concepts section
    related_lines = ""
    if related_concepts:
        sorted_related = sorted(
            ((c, meta) for c, meta in related_concepts.items() if c != concept),
            key=lambda x: x[0],
        )
        if sorted_related:
            lines = []
            for c, meta in sorted_related:
                parts = []
                co = meta.get("co_occurred", 0)
                if co > 0:
                    parts.append(f"co-occurred {co} time{'s' if co != 1 else ''}")
                shared = meta.get("shared_tags", set())
                if shared:
                    parts.append(f"shared tags: {', '.join(sorted(shared))}")
                annotation = " — " + ", ".join(parts) if parts else ""
                lines.append(f"- [[{c}]]{annotation}")
            related_lines = "\n## Related Concepts\n\n" + "\n".join(lines) + "\n"

    ref_count = sum(len(v) for v in by_category.values())

    if concept_file.exists():
        # Rewrite the file with updated content
        existing = concept_file.read_text()

        # Extract existing frontmatter created date
        created_match = re.search(r"created: (\S+)", existing)
        created = created_match.group(1) if created_match else datetime.now().strftime("%Y-%m-%d")
    else:
        created = datetime.now().strftime("%Y-%m-%d")

    all_tags: set[str] = set()
    for ref in references:
        qwen = ref.get("qwen_result", {})
        all_tags.update(qwen.get("tags", []))

    tags_yaml = "\n".join(f"  - {t}" for t in sorted(all_tags)) if all_tags else "  - concept"

    content = f"""---
title: {concept}
tags:
{tags_yaml}
  - concept
type: concept
created: {created}
references: {ref_count}
---

# {concept}

## Referenced Conversations

{ref_section}
{related_lines}"""
    concept_file.write_text(content)
    return concept_file


# ── Step 10: MOC with Recent Sessions ──


def update_moc(
    daily_files: list[str],
    concept_files: list[str],
    session_files: list[str] | None = None,
) -> None:
    """Update the Map of Content note with daily, concept, and recent session links."""
    KNOWLEDGE_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")

    existing_dailies: set[str] = set()
    existing_concepts: set[str] = set()
    existing_sessions: set[str] = set()

    if MOC_PATH.exists():
        existing = MOC_PATH.read_text()
        for match in re.finditer(r"\[\[daily/([^\]]+)\]\]", existing):
            existing_dailies.add(match.group(1))
        for match in re.finditer(r"\[\[concepts/([^\]]+)\]\]", existing):
            existing_concepts.add(match.group(1))
        for match in re.finditer(r"\[\[(projects/[^|\]]+/sessions/[^\]|]+)(?:\|[^\]]+)?\]\]", existing):
            existing_sessions.add(match.group(1))
        for match in re.finditer(r"\[\[(sessions/[^\]|]+)(?:\|[^\]]+)?\]\]", existing):
            existing_sessions.add(match.group(1))

    for d in daily_files:
        existing_dailies.add(d)
    for c in concept_files:
        existing_concepts.add(c)
    if session_files:
        for s in session_files:
            existing_sessions.add(s)

    sorted_dailies = sorted(existing_dailies, reverse=True)
    sorted_concepts = sorted(existing_concepts)
    sorted_sessions = sorted(existing_sessions, reverse=True)[:20]

    daily_links = "\n".join(f"- [[daily/{d}]]" for d in sorted_dailies)
    concept_links = "\n".join(f"- [[concepts/{c}]]" for c in sorted_concepts)
    session_links = "\n".join(f"- [[{s}]]" for s in sorted_sessions)

    content = f"""---
title: Knowledge Graph - Map of Content
updated: {today}
tags:
  - MOC
  - knowledge-graph
type: moc
---

# Knowledge Graph

Map of Content for the AI conversation knowledge graph.

## Recent Sessions

{session_links}

## Daily Logs

{daily_links}

## Concepts

{concept_links}
"""
    MOC_PATH.write_text(content)


# ── Step 9: Main orchestration ──


def main() -> None:
    if not KNOWLEDGE_GRAPH_DIR or KNOWLEDGE_GRAPH_DIR == Path(""):
        print("Error: Vault directory not configured. Run 'ckg init --vault-dir <path>' first.")
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 50)
    log("Obsidian Writer started")

    files = get_processed_files()
    if not files:
        log("No processed files to write; running session rename pass only")
    else:
        log(f"Found {len(files)} processed file(s)")

    written_dailies: list[str] = []
    written_concepts: list[str] = []
    written_sessions: list[str] = []
    all_qa: list[dict] = []

    for filepath in files:
        try:
            qa = json.loads(filepath.read_text())
        except (json.JSONDecodeError, Exception) as e:
            log(f"Failed to read {filepath.name}: {e}")
            continue

        if qa.get("status") != "processed":
            continue

        all_qa.append(qa)
        meta = _qa_project_meta(qa)
        qa["cwd"] = meta["cwd"]
        qa["project_root"] = meta["project_root"]
        qa["project_slug"] = meta["project_slug"]
        qa["project_name"] = meta["project_name"]
        qa["source_platform"] = meta["source_platform"]

        qa["status"] = "written"
        qa["written_at"] = datetime.now().isoformat()
        filepath.write_text(json.dumps(qa, ensure_ascii=False, indent=2))

    filename_map = build_session_filename_map(all_qa)
    rename_map = build_existing_session_rename_map(all_qa, filename_map)
    rename_existing_session_files(rename_map)
    rewrite_session_links(rename_map)
    if rename_map:
        log(f"Renamed {len(rename_map)} existing session note(s)")
    elif not files:
        log("No existing session note renames needed")

    if not files:
        log("Obsidian Writer finished")
        return

    daily_entries: dict[str, list[str]] = {}
    daily_concepts: dict[str, list[str]] = {}
    concept_refs: dict[str, list[dict]] = {}
    for qa in all_qa:
        date_str = _qa_date_str(qa)
        entry = build_daily_entry(qa, filename_map)
        daily_entries.setdefault(date_str, []).append(entry)

        qwen = qa.get("qwen_result", {})
        for concept in qwen.get("key_concepts", []):
            concept_refs.setdefault(concept, []).append(qa)
            daily_concepts.setdefault(date_str, []).append(concept)

    # Build session-to-session similarity relations
    session_rels = build_session_relations(all_qa, filename_map)
    total_see_also = sum(len(v) for v in session_rels.values())
    log(f"Built session relations: {total_see_also} 'See Also' links")

    # Write individual session notes
    for qa in all_qa:
        fname = session_link_target(qa, filename_map)
        see_also = session_rels.get(fname, [])
        session_path = write_session_note(qa, see_also, filename_map)
        written_sessions.append(fname)
        log(f"Wrote session note: {session_path.relative_to(KNOWLEDGE_GRAPH_DIR)}")

    # Write daily notes (index format)
    for date_str, entries in daily_entries.items():
        day_concepts = daily_concepts.get(date_str)
        daily_path = write_daily_note(date_str, entries, day_concepts)
        written_dailies.append(date_str)
        log(f"Wrote daily note: {daily_path.name}")

    # Build concept relations with metadata
    relations = build_concept_relations(concept_refs)
    total_relations = sum(len(v) for v in relations.values()) // 2
    log(f"Built concept relations: {total_relations} unique pairs")

    # Write concept notes
    for concept, refs in concept_refs.items():
        related = relations.get(concept)
        concept_path = write_concept_note(concept, refs, related, filename_map)
        written_concepts.append(sanitize_filename(concept))
        log(f"Wrote concept note: {concept_path.name}")

    # Update MOC
    if written_dailies or written_concepts or written_sessions:
        update_moc(written_dailies, written_concepts, written_sessions)
        log("Updated _MOC.md")

    log(
        f"Done: {len(written_sessions)} session notes, "
        f"{len(written_dailies)} daily notes, "
        f"{len(written_concepts)} concept notes"
    )
    log("Obsidian Writer finished")


if __name__ == "__main__":
    main()
