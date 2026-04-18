from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError

import pandas as pd

BYBIT_V5_BASE_URL = "https://api.bybit.com/v5/market/kline"


@dataclass(frozen=True)
class BybitKlineRequest:
    category: str = "linear"
    symbol: str = "BTCUSDT"
    interval: str = "60"
    start_ms: int | None = None
    end_ms: int | None = None
    limit: int = 1000
    timeout: int = 15
    retries: int = 3
    retry_backoff_seconds: float = 1.5
    pause_between_pages_seconds: float = 0.2


def build_bybit_kline_url(request: BybitKlineRequest) -> str:
    params: dict[str, Any] = {
        "category": request.category,
        "symbol": request.symbol,
        "interval": request.interval,
        "limit": request.limit,
    }
    if request.start_ms is not None:
        params["start"] = request.start_ms
    if request.end_ms is not None:
        params["end"] = request.end_ms
    return f"{BYBIT_V5_BASE_URL}?{urlencode(params)}"


def _request_json(url: str, timeout: int = 15, retries: int = 3, backoff_seconds: float = 1.5) -> dict[str, Any]:
    last_error: Exception | None = None
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KronosBybitAdapter/1.0",
    }

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                payload = resp.read().decode("utf-8")
            return json.loads(payload)
        except (URLError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds * attempt)

    if last_error is None:
        raise RuntimeError(f"Failed to fetch JSON from {url}")
    raise RuntimeError(f"Failed to fetch JSON from {url}: {last_error}") from last_error


def _to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _to_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def normalize_bybit_kline_rows(rows: Iterable[Sequence[Any]]) -> pd.DataFrame:
    """Normalize raw Bybit kline rows to the Kronos CSV schema.

    The expected Bybit row layout is:
    [startTime, open, high, low, close, volume, turnover]
    """

    normalized_rows: list[dict[str, Any]] = []

    for row in rows:
        if len(row) < 6:
            continue

        start_ms = _to_int(row[0])
        open_price = _to_float(row[1])
        high_price = _to_float(row[2])
        low_price = _to_float(row[3])
        close_price = _to_float(row[4])
        volume = _to_float(row[5])
        turnover = _to_float(row[6]) if len(row) > 6 else close_price * volume

        normalized_rows.append(
            {
                "timestamps": pd.to_datetime(start_ms, unit="ms"),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "amount": turnover,
            }
        )

    if not normalized_rows:
        return pd.DataFrame(columns=["timestamps", "open", "high", "low", "close", "volume", "amount"])

    df = pd.DataFrame(normalized_rows)
    df = df.sort_values("timestamps").drop_duplicates(subset=["timestamps"], keep="last").reset_index(drop=True)
    return df


def _extract_result_rows(payload: dict[str, Any]) -> list[list[Any]]:
    if payload.get("retCode") not in (0, "0", None):
        raise RuntimeError(f"Bybit API error: retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")

    result = payload.get("result") or {}
    rows = result.get("list") or []
    if not isinstance(rows, list):
        raise RuntimeError("Bybit API response did not include a list of kline rows")
    return rows


def fetch_bybit_kline_page(request: BybitKlineRequest) -> pd.DataFrame:
    url = build_bybit_kline_url(request)
    payload = _request_json(
        url,
        timeout=request.timeout,
        retries=request.retries,
        backoff_seconds=request.retry_backoff_seconds,
    )
    rows = _extract_result_rows(payload)
    return normalize_bybit_kline_rows(rows)


def _empty_kline_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamps", "open", "high", "low", "close", "volume", "amount"])


def _filter_kline_frame_by_range(
    page_df: pd.DataFrame,
    start_ms: int | None,
    end_ms: int | None,
) -> pd.DataFrame:
    filtered = page_df
    if end_ms is not None:
        filtered = filtered[filtered["timestamps"] <= pd.to_datetime(end_ms, unit="ms")]
    if start_ms is not None:
        filtered = filtered[filtered["timestamps"] >= pd.to_datetime(start_ms, unit="ms")]
    return filtered


def _next_page_cursor_ms(page_df: pd.DataFrame) -> int:
    oldest_ts = page_df["timestamps"].min()
    return int(oldest_ts.timestamp() * 1000) - 1


def _combine_kline_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return _empty_kline_frame()
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values("timestamps").drop_duplicates(subset=["timestamps"], keep="last").reset_index(drop=True)


def _fetch_single_page_frame(
    category: str,
    symbol: str,
    interval: str,
    start_ms: int | None,
    end_ms: int | None,
    limit: int,
    timeout: int,
    retries: int,
    retry_backoff_seconds: float,
) -> pd.DataFrame:
    request = BybitKlineRequest(
        category=category,
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=limit,
        timeout=timeout,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    return fetch_bybit_kline_page(request)


def fetch_bybit_klines(
    category: str = "linear",
    symbol: str = "BTCUSDT",
    interval: str = "60",
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = 1000,
    max_pages: int = 20,
    timeout: int = 15,
    retries: int = 3,
    retry_backoff_seconds: float = 1.5,
    pause_between_pages_seconds: float = 0.2,
) -> pd.DataFrame:
    """Fetch multiple Bybit kline pages and return them normalized.

    The Bybit endpoint returns rows in reverse chronological order. This helper
    pages backward in time until it reaches `start_ms` or exhausts the result set.
    """

    if limit <= 0:
        raise ValueError("limit must be a positive integer")
    if max_pages <= 0:
        raise ValueError("max_pages must be a positive integer")

    frames: list[pd.DataFrame] = []
    current_end = end_ms

    for _ in range(max_pages):
        page_df = _fetch_single_page_frame(
            category=category,
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=current_end,
            limit=limit,
            timeout=timeout,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        if page_df.empty:
            break

        filtered_page_df = _filter_kline_frame_by_range(page_df, start_ms=start_ms, end_ms=end_ms)
        if filtered_page_df.empty:
            break

        frames.append(filtered_page_df)

        if len(filtered_page_df) < limit:
            break

        current_end = _next_page_cursor_ms(filtered_page_df)
        if pause_between_pages_seconds > 0:
            time.sleep(pause_between_pages_seconds)

    return _combine_kline_frames(frames)


def export_bybit_klines_to_csv(
    output_path: str,
    category: str = "linear",
    symbol: str = "BTCUSDT",
    interval: str = "60",
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = 1000,
    max_pages: int = 20,
    timeout: int = 15,
    retries: int = 3,
    retry_backoff_seconds: float = 1.5,
    pause_between_pages_seconds: float = 0.2,
) -> pd.DataFrame:
    df = fetch_bybit_klines(
        category=category,
        symbol=symbol,
        interval=interval,
        start_ms=start_ms,
        end_ms=end_ms,
        limit=limit,
        max_pages=max_pages,
        timeout=timeout,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        pause_between_pages_seconds=pause_between_pages_seconds,
    )
    if df.empty:
        raise RuntimeError("No Bybit kline data was returned; nothing to export")
    df.to_csv(output_path, index=False)
    return df
