#!/usr/bin/env bash
# Launch the MailMind REST API with hot-reload on port 8000.
set -e

# Resolve the repository root from this script's location so it can be run
# from anywhere, then expose the src layout to Python.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

exec uvicorn mailmind.api.main:app --reload --port 8000
