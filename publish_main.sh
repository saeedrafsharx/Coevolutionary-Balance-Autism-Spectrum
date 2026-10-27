#!/usr/bin/env bash
# Replaces remote main with the prepared local commit, as requested by the owner.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$repo_root/.publication-git" ]]; then
  git --git-dir="$repo_root/.publication-git" --work-tree="$repo_root" push --force origin main:main
else
  git -C "$repo_root" push --force origin main:main
fi
