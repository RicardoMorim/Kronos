from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_csv.bybit_data import fetch_bybit_klines
from model import Kronos, KronosPredictor, KronosTokenizer


REQUIRED_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


@dataclass(frozen=True)
class HorizonRuntime:
    name: str
    interval: str
    csv_path: Path
    lookback: int
    pred_len: int
    history_days: int
    model_path: str
    tokenizer_path: str


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_direction_signal(last_close: float, next_close: float, threshold_bps: float) -> str:
    change = (next_close - last_close) / max(abs(last_close), 1e-12)
    threshold = threshold_bps / 10_000.0
    if change > threshold:
        return "long"
    if change < -threshold:
        return "short"
    return "flat"


def merge_dual_signals(signal_1h: str, signal_5m: str) -> str:
    if signal_1h == signal_5m:
        return signal_1h
    if signal_1h == "flat":
        return signal_5m
    if signal_5m == "flat":
        return signal_1h
    return "flat"


def utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def fetch_horizon_csv(
    symbol: str,
    category: str,
    horizon: HorizonRuntime,
    max_pages: int,
    limit: int,
) -> pd.DataFrame:
    end_ms = utc_now_ms()
    start_ms = int((datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc) - timedelta(days=horizon.history_days)).timestamp() * 1000)

    df = fetch_bybit_klines(
        category=category,
        symbol=symbol,
        interval=horizon.interval,
        start_ms=start_ms,
        end_ms=end_ms,
        max_pages=max_pages,
        limit=limit,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {horizon.name} ({horizon.interval})")

    horizon.csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(horizon.csv_path, index=False)
    return df


def load_predictor(model_name: str, tokenizer_name: str, max_context: int, device: str | None) -> KronosPredictor:
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    return KronosPredictor(model=model, tokenizer=tokenizer, device=device, max_context=max_context)


def run_horizon_prediction(
    predictor: KronosPredictor,
    horizon: HorizonRuntime,
    temperature: float,
    top_p: float,
    top_k: int,
    sample_count: int,
    threshold_bps: float,
) -> dict[str, Any]:
    df = pd.read_csv(horizon.csv_path)
    ensure_columns(df, REQUIRED_COLUMNS)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    if len(df) < horizon.lookback + horizon.pred_len:
        raise ValueError(
            f"Not enough rows for {horizon.name}: need at least {horizon.lookback + horizon.pred_len}, got {len(df)}"
        )

    x_df = df.iloc[-horizon.lookback:][["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.iloc[-horizon.lookback:]["timestamps"]

    last_ts = x_timestamp.iloc[-1]
    freq = x_timestamp.diff().mode().iloc[0]
    y_timestamp = pd.Series(pd.date_range(start=last_ts + freq, periods=horizon.pred_len, freq=freq))

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=horizon.pred_len,
        T=temperature,
        top_p=top_p,
        top_k=top_k,
        sample_count=sample_count,
        verbose=False,
    )

    last_close = float(x_df["close"].iloc[-1])
    next_close = float(pred_df["close"].iloc[0])
    signal = compute_direction_signal(last_close=last_close, next_close=next_close, threshold_bps=threshold_bps)

    return {
        "horizon": horizon.name,
        "interval": horizon.interval,
        "last_timestamp": str(last_ts),
        "forecast_timestamp": str(pred_df.index[0]),
        "last_close": last_close,
        "forecast_close": next_close,
        "signal": signal,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch latest Bybit data and generate dual-timeframe signal in one command.")

    parser.add_argument("--symbol", default="BTCUSDT", help="Bybit symbol")
    parser.add_argument("--category", default="linear", choices=["linear", "inverse", "spot", "option"])
    parser.add_argument("--max-pages", type=int, default=200, help="Max pages per horizon fetch")
    parser.add_argument("--limit", type=int, default=1000, help="Rows per page for Bybit API")

    parser.add_argument("--interval-1h", default="60")
    parser.add_argument("--interval-5m", default="5")
    parser.add_argument("--history-days-1h", type=int, default=730)
    parser.add_argument("--history-days-5m", type=int, default=60)
    parser.add_argument("--lookback-1h", type=int, default=512)
    parser.add_argument("--lookback-5m", type=int, default=512)
    parser.add_argument("--pred-len-1h", type=int, default=1)
    parser.add_argument("--pred-len-5m", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=512)

    parser.add_argument("--csv-1h", default="finetune_csv/data/bybit/bybit_BTCUSDT_60m.csv")
    parser.add_argument("--csv-5m", default="finetune_csv/data/bybit/bybit_BTCUSDT_5m.csv")

    parser.add_argument("--model-1h", default="finetune/outputs/bybit_multi/bybit_predictor/checkpoints/best_model")
    parser.add_argument("--tokenizer-1h", default="finetune/outputs/bybit_multi/bybit_tokenizer/checkpoints/best_model")
    parser.add_argument("--model-5m", default="finetune/outputs/bybit_multi/bybit_predictor/checkpoints/best_model")
    parser.add_argument("--tokenizer-5m", default="finetune/outputs/bybit_multi/bybit_tokenizer/checkpoints/best_model")

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--threshold-bps", type=float, default=5.0)

    parser.add_argument("--output", default="finetune_csv/data/bybit/latest_signal.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    one_h = HorizonRuntime(
        name="1h",
        interval=args.interval_1h,
        csv_path=Path(args.csv_1h),
        lookback=args.lookback_1h,
        pred_len=args.pred_len_1h,
        history_days=args.history_days_1h,
        model_path=args.model_1h,
        tokenizer_path=args.tokenizer_1h,
    )
    five_m = HorizonRuntime(
        name="5m",
        interval=args.interval_5m,
        csv_path=Path(args.csv_5m),
        lookback=args.lookback_5m,
        pred_len=args.pred_len_5m,
        history_days=args.history_days_5m,
        model_path=args.model_5m,
        tokenizer_path=args.tokenizer_5m,
    )

    print("[1/3] Fetching Bybit data...")
    fetch_horizon_csv(args.symbol, args.category, one_h, max_pages=args.max_pages, limit=args.limit)
    fetch_horizon_csv(args.symbol, args.category, five_m, max_pages=args.max_pages, limit=args.limit)

    print("[2/3] Loading models...")
    predictor_1h = load_predictor(one_h.model_path, one_h.tokenizer_path, max_context=args.max_context, device=args.device)
    predictor_5m = load_predictor(five_m.model_path, five_m.tokenizer_path, max_context=args.max_context, device=args.device)

    print("[3/3] Generating signals...")
    signal_1h = run_horizon_prediction(
        predictor=predictor_1h,
        horizon=one_h,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        sample_count=args.sample_count,
        threshold_bps=args.threshold_bps,
    )
    signal_5m = run_horizon_prediction(
        predictor=predictor_5m,
        horizon=five_m,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        sample_count=args.sample_count,
        threshold_bps=args.threshold_bps,
    )

    final_signal = merge_dual_signals(signal_1h["signal"], signal_5m["signal"])

    payload = {
        "symbol": args.symbol,
        "category": args.category,
        "models": {
            "1h": one_h.model_path,
            "5m": five_m.model_path,
        },
        "tokenizers": {
            "1h": one_h.tokenizer_path,
            "5m": five_m.tokenizer_path,
        },
        "threshold_bps": args.threshold_bps,
        "signals": {
            "1h": signal_1h,
            "5m": signal_5m,
        },
        "final_signal": final_signal,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Saved signal file to: {output_path.as_posix()}")


if __name__ == "__main__":
    main()
