from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML config at {path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def apply_smoke_overrides(data: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(data)
    training = dict(cfg.get("training") or {})
    training["tokenizer_epochs"] = 1
    training["basemodel_epochs"] = 1
    training["batch_size"] = min(int(training.get("batch_size", 16)), 8)
    training["num_workers"] = 0
    cfg["training"] = training

    experiment = dict(cfg.get("experiment") or {})
    experiment["use_comet"] = False
    cfg["experiment"] = experiment
    return cfg


def build_train_command(config_path: Path, skip_tokenizer: bool, skip_basemodel: bool) -> list[str]:
    command = [
        sys.executable,
        "finetune_csv/train_sequential.py",
        "--config",
        str(config_path.as_posix()),
    ]
    if skip_tokenizer:
        command.append("--skip-tokenizer")
    if skip_basemodel:
        command.append("--skip-basemodel")
    return command


def run_training(
    config_path: Path,
    smoke_mode: bool,
    skip_tokenizer: bool,
    skip_basemodel: bool,
) -> None:
    effective_config_path = config_path
    temp_file: tempfile.NamedTemporaryFile[str] | None = None

    try:
        if smoke_mode:
            original = load_yaml(config_path)
            modified = apply_smoke_overrides(original)
            temp_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
            tmp_path = Path(temp_file.name)
            temp_file.close()
            dump_yaml(tmp_path, modified)
            effective_config_path = tmp_path

        cmd = build_train_command(
            config_path=effective_config_path,
            skip_tokenizer=skip_tokenizer,
            skip_basemodel=skip_basemodel,
        )
        print(f"[run] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    finally:
        if temp_file is not None:
            try:
                Path(temp_file.name).unlink(missing_ok=True)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Kronos sequentially for dual Bybit timeframes (1H + 5m).")
    parser.add_argument(
        "--config-1h",
        default="finetune_csv/configs/config_bybit_btcusdt_60m.yaml",
        help="Config path for 1H training",
    )
    parser.add_argument(
        "--config-5m",
        default="finetune_csv/configs/config_bybit_btcusdt_5m.yaml",
        help="Config path for 5m training",
    )
    parser.add_argument("--smoke", action="store_true", help="Run with 1 epoch and smaller batch size")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer phase")
    parser.add_argument("--skip-basemodel", action="store_true", help="Skip basemodel phase")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_1h = Path(args.config_1h)
    config_5m = Path(args.config_5m)

    if not config_1h.exists():
        raise FileNotFoundError(f"1H config not found: {config_1h}")
    if not config_5m.exists():
        raise FileNotFoundError(f"5m config not found: {config_5m}")

    print("=== Dual-timeframe training: 1H ===")
    run_training(
        config_path=config_1h,
        smoke_mode=args.smoke,
        skip_tokenizer=args.skip_tokenizer,
        skip_basemodel=args.skip_basemodel,
    )

    print("=== Dual-timeframe training: 5m ===")
    run_training(
        config_path=config_5m,
        smoke_mode=args.smoke,
        skip_tokenizer=args.skip_tokenizer,
        skip_basemodel=args.skip_basemodel,
    )

    print("Dual-timeframe training completed.")


if __name__ == "__main__":
    main()
