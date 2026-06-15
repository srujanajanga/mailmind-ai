#!/usr/bin/env bash
# Launch the MailMind Streamlit UI.
set -e

# Resolve the repository root from this script's location so it can be run
# from anywhere, then expose the src layout to Python.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

# --server.headless true skips Streamlit's first-run "Email:" onboarding prompt
# (which otherwise pauses startup) and stops it auto-opening a browser tab.
exec streamlit run app/streamlit_app.py \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.port 8501
