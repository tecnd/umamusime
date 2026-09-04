#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the umamusime project.
# Installs the uv toolchain (if missing) and syncs the pinned Python
# interpreter and locked dependencies from uv.lock.
set -euo pipefail

# Install uv into ~/.local/bin when it is not already available.
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Expose uv/uvx on a system PATH location so every future agent shell finds it,
# even non-login shells. Best-effort: skip silently if sudo is unavailable.
if command -v sudo >/dev/null 2>&1; then
  sudo ln -sf "$(command -v uv)" /usr/local/bin/uv 2>/dev/null || true
  sudo ln -sf "$(command -v uvx)" /usr/local/bin/uvx 2>/dev/null || true
fi

# Install the pinned Python (3.11) and all locked project + dev dependencies.
uv sync

echo "install.sh complete: $(uv run python --version)"
