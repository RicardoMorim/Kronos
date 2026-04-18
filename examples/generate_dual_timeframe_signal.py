from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer


REQUIRED_COLUMNS = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


@dataclass(frozen=True)
class HorizonConfig:
    name: str
    csv_path: Path
    lookback: int
    pred_len: int
    max_context: int


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


def load_predictor(model_name: str, tokenizer_name: str, max_context: int, device: str | None) -> KronosPredictor:
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    return KronosPredictor(model=model, tokenizer=tokenizer, device=device, max_context=max_context)


def run_horizon_prediction(
    predictor: KronosPredictor,
    horizon: HorizonConfig,
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
    y_timestamp = pd.date_range(start=last_ts + freq, periods=horizon.pred_len, freq=freq)

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
    horizon_signal = compute_direction_signal(last_close=last_close, next_close=next_close, threshold_bps=threshold_bps)

    return {
        "horizon": horizon.name,
        "last_timestamp": str(last_ts),
        "forecast_timestamp": str(pred_df.index[0]),
        "last_close": last_close,
        "forecast_close": next_close,
        "signal": horizon_signal,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dual-timeframe signal from Bybit datasets with Kronos.")
    parser.add_argument("--csv-1h", default="finetune_csv/data/bybit/bybit_BTCUSDT_60m.csv")
    parser.add_argument("--csv-5m", default="finetune_csv/data/bybit/bybit_BTCUSDT_5m.csv")
    parser.add_argument("--model", default="NeoQuasar/Kronos-small", help="Model name or local model path")
    parser.add_argument("--tokenizer", default="NeoQuasar/Kronos-Tokenizer-base", help="Tokenizer name or local path")
    parser.add_argument("--device", default=None, help="Force device, e.g. cpu or cuda:0")
    parser.add_argument("--lookback-1h", type=int, default=512)
    parser.add_argument("--lookback-5m", type=int, default=512)
    parser.add_argument("--pred-len-1h", type=int, default=1)
    parser.add_argument("--pred-len-5m", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--threshold-bps", type=float, default=5.0, help="Signal threshold in basis points")
    parser.add_argument("--output", default="finetune_csv/data/bybit/latest_signal.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictor = load_predictor(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        max_context=args.max_context,
        device=args.device,
    )

    one_h = HorizonConfig(
        name="1h",
        csv_path=Path(args.csv_1h),
        lookback=args.lookback_1h,
        pred_len=args.pred_len_1h,
        max_context=args.max_context,
    )
    five_m = HorizonConfig(
        name="5m",
        csv_path=Path(args.csv_5m),
        lookback=args.lookback_5m,
        pred_len=args.pred_len_5m,
        max_context=args.max_context,
    )

    signal_1h = run_horizon_prediction(
        predictor=predictor,
        horizon=one_h,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        sample_count=args.sample_count,
        threshold_bps=args.threshold_bps,
    )
    signal_5m = run_horizon_prediction(
        predictor=predictor,
        horizon=five_m,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        sample_count=args.sample_count,
        threshold_bps=args.threshold_bps,
    )

    final_signal = merge_dual_signals(signal_1h=signal_1h["signal"], signal_5m=signal_5m["signal"])

    payload = {
        "model": args.model,
        "tokenizer": args.tokenizer,
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
