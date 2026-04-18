from __future__ import annotations

import json
from pathlib import Path

import pytest

from finetune_csv.prepare_bybit_datasets import (
    build_output_csv_path,
    compute_time_range_ms,
    interval_label,
    parse_iso_to_utc_ms,
    sanitize_symbol,
    split_intervals,
)


def test_parse_iso_to_utc_ms_handles_zulu_time():
    ts = parse_iso_to_utc_ms("2024-01-01T00:00:00Z")
    assert ts == 1704067200000


def test_compute_time_range_ms_orders_start_end():
    start_ms, end_ms = compute_time_range_ms("2024-01-01T00:00:00Z", "2024-02-01T00:00:00Z", history_days=10)
    assert start_ms < end_ms


def test_compute_time_range_ms_rejects_invalid_order():
    with pytest.raises(ValueError):
        compute_time_range_ms("2024-02-01T00:00:00Z", "2024-01-01T00:00:00Z", history_days=10)


def test_split_intervals_parses_csv_values():
    assert split_intervals("60, 5, D") == ["60", "5", "D"]


def test_symbol_and_interval_labels():
    assert sanitize_symbol("btc/usdt") == "BTCUSDT"
    assert interval_label("5") == "5m"
    assert interval_label("D") == "d"


def test_build_output_csv_path_uses_normalized_name():
    path = build_output_csv_path(Path("finetune_csv/data/bybit"), "btc/usdt", "60")
    assert path.as_posix().endswith("finetune_csv/data/bybit/bybit_BTCUSDT_60m.csv")
