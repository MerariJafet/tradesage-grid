#!/usr/bin/env python3
"""CLI shim to expose the aggregated signals backtest runner under scripts/."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

runpy.run_module("backend.scripts.backtest_aggregated_signals", run_name="__main__")
