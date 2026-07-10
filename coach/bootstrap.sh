#!/usr/bin/env bash
# coach/bootstrap.sh — relocate platform runtime state into .local/ and
# create transparent symlinks back into profile/ so the platform continues
# to read/write the same logical paths.
#
# Why:
#   The platform (src/qwenpaw) writes runtime artifacts (credentials.yaml,
#   drivers/, history.db, sessions/, memory/, ...) directly under the agent's
#   workspace_dir (coach/profile/). Those artifacts must NOT be committed.
#   Rather than maintaining a long gitignore allowlist, we keep the real data
#   in coach/.local/ (gitignored) and symlink from profile/.
#
# This script is:
#   - Idempotent (safe to run repeatedly)
#   - Non-destructive (existing files are moved, not deleted)
#   - Cross-platform (macOS/Linux; on Windows use WSL or Git Bash)
#
# Usage:
#   cd <repo-root>
#   bash coach/bootstrap.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COACH_DIR="$REPO_ROOT/coach"
PROFILE_DIR="$COACH_DIR/profile"
LOCAL_DIR="$COACH_DIR/.local"

# Everything under profile/ that is runtime-generated state (not the template).
# Update this list if the platform introduces new runtime artifacts.
RUNTIME_ITEMS=(
  agent.json
  credentials.yaml
  drivers
  history.db
  chats.json
  jobs.json
  sessions
  memory
  mem_metadata
  mem_session
  memory_file_metadata.json
  skill.json
  dialog
  media
  embedding_cache
  tool_results
  resource
  digest
  file_store
  .mcp
  .reme_store_v1
  .skill.json.lock
  .bootstrap_completed
)

log() { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bootstrap]\033[0m %s\n' "$*"; }

mkdir -p "$LOCAL_DIR"

for item in "${RUNTIME_ITEMS[@]}"; do
  src="$PROFILE_DIR/$item"
  dst="$LOCAL_DIR/$item"

  # Skip if already a symlink (from a previous run).
  if [ -L "$src" ]; then
    continue
  fi

  # Skip if neither file nor directory exists.
  if [ ! -e "$src" ]; then
    continue
  fi

  if [ -e "$dst" ]; then
    warn "$item exists in both profile/ and .local/ — keeping .local/ copy; skipping move. Review manually if needed."
  else
    log "moving profile/$item → .local/$item"
    mv -- "$src" "$dst"
  fi
done

# Create symlinks profile/<item> → ../.local/<item>
for item in "${RUNTIME_ITEMS[@]}"; do
  src="$PROFILE_DIR/$item"
  dst="../.local/$item"

  # If the symlink already exists and points to the right target, skip.
  if [ -L "$src" ]; then
    current_target="$(readlink "$src")"
    if [ "$current_target" = "$dst" ]; then
      continue
    fi
    warn "profile/$item is a symlink to $current_target; re-pointing to $dst"
    rm -- "$src"
  fi

  # Don't overwrite an existing regular file/dir that wasn't moved above.
  if [ -e "$src" ]; then
    warn "profile/$item still exists; refusing to replace. Resolve manually."
    continue
  fi

  log "linking profile/$item → $dst"
  ln -s -- "$dst" "$src"
done

log ""
log "Bootstrap complete."
log "  - Runtime state lives in:     coach/.local/"
log "  - Template (synced) lives in: coach/profile/"
log "  - profile/* paths resolve transparently via symlinks."
log ""
log "Next steps:"
log "  1. Run 'git status' — you should see no runtime files under profile/."
log "  2. Launch the coach agent normally and verify it works."
