# local-demo — cold-start scaffolding for `.local/`

This directory contains sample files that demonstrate the expected structure of
`coach/.local/`. Use it when setting up the coach agent on a new device or
when onboarding a new developer.

## Quick start

From the repo root:

```bash
bash coach/bootstrap.sh            # relocate any existing runtime state
mkdir -p coach/.local/profile coach/.local/projects
cp -n coach/local-demo/profile/learner.json.sample coach/.local/profile/learner.json
cp -n coach/local-demo/projects/sample-project.md.sample coach/.local/projects/sample-project.md
cp -n coach/local-demo/projects/coding-project.md.sample coach/.local/projects/coding-project.md
```

## Structure copied into `.local/`

```
.local/
├── profile/
│   └── learner.json        # Learner mastery & interest profile (skills read/write this)
└── projects/
    ├── sample-project.md   # Generic PBL project tracker
    └── coding-project.md   # Example coding project for /codex delegation (Codex CWD, stack, etc.)
```

## What lives in `.local/` at runtime

Everything the platform and skills generate at runtime ends up here. The
`coach/bootstrap.sh` script wires transparent symlinks from `profile/` into
`.local/` so the platform's `workspace_dir/...` paths keep working unchanged:

- `credentials.yaml` — secrets (never commit)
- `agent.json` — live agent, channel, provider, and ACP configuration
- `drivers/` — MCP driver cards and migration reports
- `history.db` — conversation history
- `chats.json`, `jobs.json`
- `sessions/`, `memory/`, `mem_metadata/`, `mem_session/`
- `dialog/`, `media/`, `embedding_cache/`, `tool_results/`, `resource/`, `digest/`
- `.mcp`, `.reme_store_v1`, `.skill.json.lock`, `.bootstrap_completed`
- `profile/learner.json`, `logs/`, `exam/`, `projects/`, `evolution/` (skill-authored)

## What does NOT go here

Anything in `coach/profile/` that isn't a runtime symlink — that's the synced
public template (persona files, `skills/`, and `knowledge/`). Never copy user
identity, private project context, or credentials into those files.
