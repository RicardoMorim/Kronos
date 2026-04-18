from __future__ import annotations

import pandas as pd
import pytest

from finetune_csv.bybit_data import BybitKlineRequest, build_bybit_kline_url, normalize_bybit_kline_rows


def test_normalize_bybit_kline_rows_sorts_and_maps_columns():
    rows = [
        ["1711929600000", "65000", "65500", "64500", "65200", "12.5", "815000"],
        ["1711926000000", "64000", "65100", "63900", "65000", "10.0", "650000"],
    ]

    df = normalize_bybit_kline_rows(rows)

    assert list(df.columns) == ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    assert len(df) == 2
    assert pd.api.types.is_datetime64_any_dtype(df["timestamps"])
    assert df["timestamps"].is_monotonic_increasing
    assert df.iloc[0]["open"] == pytest.approx(64000.0)
    assert df.iloc[1]["close"] == pytest.approx(65200.0)
    assert df.iloc[0]["amount"] == pytest.approx(650000.0)


def test_build_bybit_kline_url_includes_required_params():
    request = BybitKlineRequest(
        category="linear",
        symbol="BTCUSDT",
        interval="60",
        start_ms=1711926000000,
        end_ms=1711929600000,
        limit=200,
    )

    url = build_bybit_kline_url(request)

    assert "https://api.bybit.com/v5/market/kline" in url
    assert "category=linear" in url
    assert "symbol=BTCUSDT" in url
    assert "interval=60" in url
    assert "start=1711926000000" in url
    assert "end=1711929600000" in url
    assert "limit=200" in url
