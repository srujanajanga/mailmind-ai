#!/usr/bin/env python3
"""Run the MailMind command-line demo.

Thin runnable wrapper around :func:`mailmind.cli.main`. Adds the repository's
``src`` directory to ``sys.path`` so the script works from the repo root without
installing the package::

    python scripts/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    """Invoke the CLI demo and return its exit status."""
    from mailmind.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
