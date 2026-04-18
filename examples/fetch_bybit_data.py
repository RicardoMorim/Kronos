from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune_csv.bybit_data import export_bybit_klines_to_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public Bybit candles and export them to CSV.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Bybit symbol, e.g. BTCUSDT")
    parser.add_argument("--category", default="linear", choices=["linear", "inverse", "spot", "option"], help="Market category")
    parser.add_argument("--interval", default="60", help="Bybit interval, e.g. 1, 5, 15, 60, 240, D")
    parser.add_argument("--start-ms", type=int, default=None, help="Start timestamp in milliseconds")
    parser.add_argument("--end-ms", type=int, default=None, help="End timestamp in milliseconds")
    parser.add_argument("--limit", type=int, default=1000, help="Rows per API page")
    parser.add_argument("--max-pages", type=int, default=20, help="Maximum number of pages to fetch")
    parser.add_argument("--output", default="data/bybit_klines.csv", help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = export_bybit_klines_to_csv(
        output_path=str(output_path),
        category=args.category,
        symbol=args.symbol,
        interval=args.interval,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        limit=args.limit,
        max_pages=args.max_pages,
    )

    print(f"Saved {len(df)} rows to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
