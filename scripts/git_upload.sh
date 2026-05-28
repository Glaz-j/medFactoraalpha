#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REMOTE_URL="${REMOTE_URL:-git@github.com:Glaz-j/medFactoraalpha.git}"
BRANCH="${BRANCH:-main}"
MESSAGE="${1:-update medFactoraalpha project}"

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REMOTE_URL"
else
  git remote set-url origin "$REMOTE_URL"
fi

git symbolic-ref HEAD "refs/heads/$BRANCH" 2>/dev/null || true

echo "== Git status before add =="
git status --short --branch

echo "== Adding tracked project files =="
git add .

echo "== Git status after add =="
git status --short --branch

if git diff --cached --quiet; then
  echo "No staged changes to commit."
else
  git commit -m "$MESSAGE"
fi

echo "== Pushing to $REMOTE_URL ($BRANCH) =="
git push -u origin "$BRANCH"
