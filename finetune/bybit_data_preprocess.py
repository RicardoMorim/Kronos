from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ccxt
import numpy as np
import pandas as pd
import yfinance as yf


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_BYBIT_DIR = REPO_ROOT / "finetune_csv" / "data" / "bybit"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "bybit_multi"

TRAIN_END = pd.Timestamp("2025-09-30 23:59:59")
VAL_START = pd.Timestamp("2025-07-01 00:00:00")
VAL_END = pd.Timestamp("2026-03-31 23:59:59")


@dataclass(frozen=True)
class SourceConfig:
    name: str
    path: Path | None = None
    ticker: str | None = None
    ccxt_symbol: str | None = None
    timeframe: str | None = None


def as_naive_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=False)
        frame = frame.set_index("datetime")
    elif "timestamps" in frame.columns:
        frame["timestamps"] = pd.to_datetime(frame["timestamps"], utc=False)
        frame = frame.set_index("timestamps")
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Frame must contain a datetime/timestamps column or a DatetimeIndex")

    frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_convert(None)
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.index.name = "datetime"
    return frame


def normalize_ohlcv_frame(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    frame = _to_datetime_index(df)

    rename_map: dict[str, str] = {}
    if "volume" in frame.columns and "vol" not in frame.columns:
        rename_map["volume"] = "vol"
    if "quote_volume" in frame.columns and "amt" not in frame.columns:
        rename_map["quote_volume"] = "amt"
    if "amount" in frame.columns and "amt" not in frame.columns:
        rename_map["amount"] = "amt"
    if rename_map:
        frame = frame.rename(columns=rename_map)

    required_price_columns = ["open", "high", "low", "close"]
    missing = [col for col in required_price_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"{source_name}: missing required columns {missing}")

    if "vol" not in frame.columns:
        if "volume" in frame.columns:
            frame["vol"] = frame["volume"]
        else:
            raise ValueError(f"{source_name}: missing volume/vol column")

    if "amt" not in frame.columns:
        frame["amt"] = frame["close"].astype(float) * frame["vol"].astype(float)

    clean = frame[["open", "high", "low", "close", "vol", "amt"]].copy()
    clean = clean.apply(pd.to_numeric, errors="coerce")
    clean = clean.dropna(how="any")
    clean.index.name = "datetime"
    return clean


def load_existing_bybit_csv(csv_path: Path, source_name: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing source CSV: {csv_path}")
    raw = pd.read_csv(csv_path)
    return normalize_ohlcv_frame(raw, source_name=source_name)


def download_yfinance_daily(ticker_candidates: Iterable[str], start: pd.Timestamp, end: pd.Timestamp, source_name: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for ticker in ticker_candidates:
        try:
            start_ts = as_naive_timestamp(start)
            end_ts = as_naive_timestamp(end) + pd.Timedelta(days=1)
            raw = yf.download(
                ticker,
                start=start_ts.to_pydatetime(),
                end=end_ts.to_pydatetime(),
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if raw is None or raw.empty:
                continue

            rename_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "vol",
            }
            raw = raw.rename(columns=rename_map)
            if "vol" not in raw.columns:
                raw["vol"] = 0.0
            if "close" not in raw.columns:
                raise ValueError(f"{source_name}: Yahoo Finance data missing close column for {ticker}")
            raw["amt"] = raw["close"].astype(float) * raw["vol"].fillna(0.0).astype(float)
            return normalize_ohlcv_frame(raw, source_name=f"{source_name}:{ticker}")
        except Exception as exc:  # pragma: no cover - fallback path exercised only on provider failure
            last_error = exc
            continue

    raise RuntimeError(f"Unable to download {source_name} from Yahoo Finance") from last_error


def resolve_bybit_symbol(exchange: ccxt.Exchange, base: str) -> str:
    candidates = [f"{base}/USDT:USDT", f"{base}/USDT", base]
    markets = getattr(exchange, "markets", {}) or {}
    for candidate in candidates:
        if candidate in markets:
            return candidate
    return candidates[0]


def fetch_ccxt_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    timeframe_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    since = int(as_naive_timestamp(start).timestamp() * 1000)
    until = int((as_naive_timestamp(end) + pd.Timedelta(milliseconds=timeframe_ms)).timestamp() * 1000)
    rows: list[list[float]] = []

    while since <= until:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break

        rows.extend(batch)
        last_ts = int(batch[-1][0])
        next_since = last_ts + timeframe_ms
        if next_since <= since:
            break
        since = next_since

        if last_ts >= until:
            break

    if not rows:
        raise RuntimeError(f"No OHLCV data returned for {symbol} @ {timeframe}")

    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "vol"])
    frame["datetime"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    frame = frame.drop(columns=["timestamp"])
    frame["amt"] = frame["close"].astype(float) * frame["vol"].astype(float)
    return normalize_ohlcv_frame(frame, source_name=f"ccxt:{symbol}:{timeframe}")


def load_bybit_sources(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    exchange.load_markets()

    sources: dict[str, pd.DataFrame] = {}

    btc_60m = load_existing_bybit_csv(SOURCE_BYBIT_DIR / "bybit_BTCUSDT_60m.csv", "BTCUSDT_60m")
    btc_5m = load_existing_bybit_csv(SOURCE_BYBIT_DIR / "bybit_BTCUSDT_5m.csv", "BTCUSDT_5m")
    sources["BTCUSDT_60m"] = btc_60m
    sources["BTCUSDT_5m"] = btc_5m

    sources["SPY_1d"] = download_yfinance_daily(["SPY"], start, end, "SPY_1d")
    sources["GLD_1d"] = download_yfinance_daily(["GLD"], start, end, "GLD_1d")
    sources["DXY_1d"] = download_yfinance_daily(["DX-Y.NYB", "^DXY", "UUP"], start, end, "DXY_1d")

    eth_symbol = resolve_bybit_symbol(exchange, "ETH")
    sol_symbol = resolve_bybit_symbol(exchange, "SOL")
    sources["ETHUSDT_60m"] = fetch_ccxt_ohlcv(exchange, eth_symbol, "1h", start, end)
    sources["SOLUSDT_60m"] = fetch_ccxt_ohlcv(exchange, sol_symbol, "1h", start, end)

    return sources


def split_symbol_frames(
    sources: dict[str, pd.DataFrame],
    train_end: pd.Timestamp = TRAIN_END,
    val_start: pd.Timestamp = VAL_START,
    val_end: pd.Timestamp = VAL_END,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    train_data: dict[str, pd.DataFrame] = {}
    val_data: dict[str, pd.DataFrame] = {}

    for symbol, frame in sources.items():
        clean = normalize_ohlcv_frame(frame, source_name=symbol)
        train_mask = clean.index <= train_end
        val_mask = (clean.index >= val_start) & (clean.index <= val_end)
        train_data[symbol] = clean.loc[train_mask].copy()
        val_data[symbol] = clean.loc[val_mask].copy()

    return train_data, val_data


def write_pickles(train_data: dict[str, pd.DataFrame], val_data: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_data.pkl", "wb") as f:
        pickle.dump(train_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(output_dir / "val_data.pkl", "wb") as f:
        pickle.dump(val_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "output_dir": output_dir.as_posix(),
        "train_symbols": list(train_data.keys()),
        "val_symbols": list(val_data.keys()),
        "train_ranges": {
            symbol: {
                "rows": len(frame),
                "start": str(frame.index.min()) if not frame.empty else None,
                "end": str(frame.index.max()) if not frame.empty else None,
            }
            for symbol, frame in train_data.items()
        },
        "val_ranges": {
            symbol: {
                "rows": len(frame),
                "start": str(frame.index.min()) if not frame.empty else None,
                "end": str(frame.index.max()) if not frame.empty else None,
            }
            for symbol, frame in val_data.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def infer_source_span() -> tuple[pd.Timestamp, pd.Timestamp]:
    frames = [
        load_existing_bybit_csv(SOURCE_BYBIT_DIR / "bybit_BTCUSDT_60m.csv", "BTCUSDT_60m"),
        load_existing_bybit_csv(SOURCE_BYBIT_DIR / "bybit_BTCUSDT_5m.csv", "BTCUSDT_5m"),
    ]
    start = min(frame.index.min() for frame in frames if not frame.empty)
    end = max(frame.index.max() for frame in frames if not frame.empty)
    return pd.Timestamp(start), pd.Timestamp(end)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Bybit multi-asset train/val pickles for Kronos fine-tuning.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write train_data.pkl and val_data.pkl")
    parser.add_argument("--start", type=str, default=None, help="Optional override for the download start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Optional override for the download end date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inferred_start, inferred_end = infer_source_span()

    start = pd.Timestamp(args.start) if args.start else inferred_start
    current_utc = pd.Timestamp.now(tz="UTC").tz_convert(None)
    end = pd.Timestamp(args.end) if args.end else min(inferred_end, current_utc)

    print(f"Using download window: {start} -> {end}")
    sources = load_bybit_sources(start=start, end=end)
    train_data, val_data = split_symbol_frames(sources)
    summary = write_pickles(train_data, val_data, args.output_dir)

    print(json.dumps(summary, indent=2))
    print(f"Saved train_data.pkl and val_data.pkl to {args.output_dir.as_posix()}")


if __name__ == "__main__":
    main()
