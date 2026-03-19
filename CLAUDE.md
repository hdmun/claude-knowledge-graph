# claude-knowledge-graph

Auto-capture Claude Code and Gemini CLI Q&A → Qwen 3.5 4B tagging/summarization → Obsidian knowledge graph.
Package: `claude-knowledge-graph`, CLI: `ckg`

## Project Structure

```
claude-knowledge-graph/
├── pyproject.toml              # Package meta + [project.scripts] ckg
├── README.md
├── LICENSE                     # MIT
├── src/
│   └── claude_knowledge_graph/
│       ├── __init__.py         # __version__
│       ├── config.py           # env vars → config.json → defaults
│       ├── project_context.py  # repo root detection + project slugging
│       ├── qa_logger.py        # Hook handler (Claude/Gemini stdin JSON → queue/)
│       ├── qwen_processor.py   # llama-server + OpenAI API tagging
│       ├── obsidian_writer.py  # Obsidian markdown generation
│       ├── cli.py              # Click CLI: init/run/status/uninstall
│       └── hooks.py            # ~/.claude/settings.json and ~/.gemini/settings.json hooks register/unregister
├── scripts/
│   └── gen_graph_image.py      # Graph visualization (optional, [graph] extra)
└── docs/
    └── knowledge-graph-example.png
```

## Pipeline Flow

```
Claude Code / Gemini CLI session
  ├─ UserPromptSubmit or BeforeAgent → qa_logger → queue/{project}/{platform}_{session}_prompt.json
  └─ Stop or AfterAgent → qa_logger → queue/{project}/{ts}_{platform}_{session}.json (Q&A pair)
       → trigger_processor() (background)
         → qwen_processor (lock → start llama-server → tagging → stop server)
           → processed/{project}/...
           → obsidian_writer (daily note, concept note, projects/{project}/sessions, _MOC.md generation)
             → release lock
```

## Configuration (config.py)

Priority: env vars → `~/.config/claude-knowledge-graph/config.json` → defaults

| Setting | Env var | config key | Default |
|---------|---------|------------|---------|
| Data dir | — | — | `~/.local/share/claude-knowledge-graph/` |
| Vault dir | `CKG_VAULT_DIR` | `vault_dir` | Set via `ckg init` |
| llama-server | `CKG_LLAMA_SERVER` | `llama_server` | PATH lookup |
| Model path | `CKG_MODEL_PATH` | `model_path` | Scan `DATA_DIR/models/` |
| Port | `CKG_LLAMA_PORT` | `llama_port` | `8199` |

## File Details

### config.py
- `_get(env, key, default)`: Config value resolution helper
- `_find_llama_server()`: env var → config → `shutil.which` → default path
- `_find_gguf_model()`: env var → config → `DATA_DIR/models/` rglob

### qa_logger.py (Hook Handler)
- Runs via `python3 -m claude_knowledge_graph.qa_logger`
- Receives Claude Code or Gemini CLI hook JSON from stdin
- Resolves `project_root` from Git repo root if available, else `cwd`
- **UserPromptSubmit / BeforeAgent**: appends prompt to `queue/{project_slug}/{platform}_{session_id}_prompt.json`
- **Stop / AfterAgent**: merges final assistant response + last prompt → generates Q&A pair JSON
- `stop_hook_active` check to prevent infinite loops
- Always exit 0 (prevents blocking Claude Code)
- Emits `{}` on stdout for Gemini hooks to satisfy strict JSON output requirements
- `trigger_processor()`: checks fcntl lock, runs qwen_processor in background

### qwen_processor.py
- Collects `status == "pending"` files from `queue/` recursively
- Starts `llama-server` → waits for health check (up to 60s)
- Calls `http://127.0.0.1:{port}/v1/chat/completions` via OpenAI client
- Uses `--chat-template-kwargs '{"enable_thinking": false}'` for non-thinking mode
- Output: `{title, summary, tags, category, key_concepts}`
- On completion → moves to `processed/{project_slug}/`, stops server (frees VRAM)
- `cleanup_orphan_prompts()`: removes orphan prompt files older than 1 hour recursively
- On success, auto-calls `obsidian_writer.main()`

### obsidian_writer.py
- Reads `status == "processed"` files from `processed/` recursively
- **Daily note** (`daily/YYYY-MM-DD.md`): frontmatter + callout format conversation log
- **Concept note** (`concepts/{name}.md`): per key_concept note, wikilinks to daily notes
- **Session note** (`projects/{project_slug}/sessions/{name}.md`): project-scoped conversation note
- **Related concept linking**: co-occurrence (same Q&A pair) + shared tags (2+ shared)
- **_MOC.md**: full daily/concept table of contents (Map of Content)
- Updates status to `written` after processing

### cli.py
- `ckg init --vault-dir <path> [--hooks ...]`: creates config.json + directories + registers hooks + checks dependencies
- `ckg run`: calls qwen_processor.main() (tagging + note generation)
- `ckg status`: pending/processed/written counts + per-project totals + per-platform hooks status
- `ckg uninstall [--hooks ...]`: unregisters hooks + optional config deletion

### hooks.py
- `register_hooks()`: adds ckg hooks to `~/.claude/settings.json` and/or `~/.gemini/settings.json` (preserves existing hooks)
- `unregister_hooks()`: removes only ckg hooks
- `check_hooks()`: checks registration status per platform
- Identifies ckg hooks by `[claude-knowledge-graph]` description

## Design Decisions

- **fcntl locking**: Mac + Linux only (no Windows support)
- **Hook handler speed**: qa_logger uses minimal imports, always exit 0
- **On-demand server**: llama-server starts/stops on `ckg run` (saves VRAM)
- **Obsidian-native**: wikilinks, callouts, frontmatter, graph view compatible

## Development

```bash
pip install -e .
ckg --help
ckg init --vault-dir /tmp/test-vault
ckg status
```

## Q&A Pair JSON

```json
{
  "session_id": "abc123",
  "timestamp": "2026-03-10T14:30:00",
  "cwd": "/path/to/project",
  "project_root": "/path/to/project",
  "project_slug": "project-ab12cd34",
  "project_name": "project",
  "source_platform": "claude | gemini",
  "prompt": "User question",
  "response": "AI response",
  "status": "pending | processed | written",
  "qwen_result": {
    "title": "Short descriptive title",
    "summary": "2-3 sentence summary",
    "tags": ["python", "debugging"],
    "category": "development | debugging | architecture | devops | data | testing | tooling | other",
    "key_concepts": ["Key Concept"]
  }
}
```
