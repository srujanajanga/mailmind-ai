#!/usr/bin/env python3
"""Capture dashboard screenshots for the report / slide deck.

Requires the Streamlit app to be running (``bash scripts/run_ui.sh``) and
Playwright with Chromium installed::

    pip install playwright && python -m playwright install chromium
    PYTHONPATH=src python3 scripts/capture_screenshots.py [--url http://localhost:8501]

Writes ``dashboard_inbox.png``, ``dashboard_detail.png`` and
``dashboard_analytics.png`` into ``docs/screenshots/``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mailmind import config  # noqa: E402

OUT = config.FIGURES_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8501")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1024},
                                device_scale_factor=2)
        page.goto(args.url, wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(2500)

        # Load the demo inbox.
        page.get_by_role("button", name="Load demo inbox").click()
        page.wait_for_timeout(3500)
        page.wait_for_selector('[data-testid="stDataFrame"]', timeout=30_000)
        page.wait_for_timeout(1000)

        # 1) Inbox (scrolled to top).
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(600)
        page.screenshot(path=str(OUT / "dashboard_inbox.png"))
        print("Wrote", OUT / "dashboard_inbox.png")

        # 2) Email-detail panel (scroll it into view, full page).
        try:
            page.get_by_text("Email detail", exact=False).first.scroll_into_view_if_needed()
            page.wait_for_timeout(800)
            page.screenshot(path=str(OUT / "dashboard_detail.png"), full_page=True)
            print("Wrote", OUT / "dashboard_detail.png")
        except Exception as exc:  # pragma: no cover - best effort
            print("detail screenshot skipped:", exc)

        # 3) Analytics tab.
        try:
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(400)
            page.get_by_role("tab", name="Analytics").click()
            page.wait_for_timeout(3000)
            page.screenshot(path=str(OUT / "dashboard_analytics.png"), full_page=True)
            print("Wrote", OUT / "dashboard_analytics.png")
        except Exception as exc:  # pragma: no cover - best effort
            print("analytics screenshot skipped:", exc)

        browser.close()


if __name__ == "__main__":
    main()
