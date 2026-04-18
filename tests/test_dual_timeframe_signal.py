from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.generate_dual_timeframe_signal import compute_direction_signal, merge_dual_signals


def test_compute_direction_signal_long_short_flat():
    assert compute_direction_signal(last_close=100.0, next_close=100.2, threshold_bps=5.0) == "long"
    assert compute_direction_signal(last_close=100.0, next_close=99.7, threshold_bps=5.0) == "short"
    assert compute_direction_signal(last_close=100.0, next_close=100.01, threshold_bps=5.0) == "flat"


def test_merge_dual_signals_priority_rules():
    assert merge_dual_signals("long", "long") == "long"
    assert merge_dual_signals("short", "short") == "short"
    assert merge_dual_signals("flat", "long") == "long"
    assert merge_dual_signals("short", "flat") == "short"
    assert merge_dual_signals("long", "short") == "flat"
