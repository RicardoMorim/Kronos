from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FINETUNE_CSV_DIR = REPO_ROOT / "finetune_csv"


def resolve_python_executable(explicit_python: str | None = None) -> str:
    if explicit_python:
        return explicit_python

    project_venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if project_venv_python.exists():
        return str(project_venv_python)

    return sys.executable


def assert_required_runtime_deps() -> None:
    required_modules = ["yaml", "torch", "pandas", "numpy", "einops", "tqdm", "huggingface_hub", "safetensors"]
    missing: list[str] = []
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Missing required runtime dependencies: "
            f"{joined}.\n"
            "Install project dependencies in this interpreter with:\n"
            "  python -m pip install -r requirements.txt"
        )


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


def build_train_command(
    config_path: Path,
    skip_tokenizer: bool,
    skip_basemodel: bool,
    python_executable: str,
) -> list[str]:
    command = [
        python_executable,
        "train_sequential.py",
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
    python_executable: str,
) -> None:
    effective_config_path = config_path
    temp_file_path: Path | None = None

    try:
        if smoke_mode:
            original = load_yaml(config_path)
            modified = apply_smoke_overrides(original)
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as temp_file:
                temp_file_path = Path(temp_file.name)
            dump_yaml(temp_file_path, modified)
            effective_config_path = temp_file_path

        cmd = build_train_command(
            config_path=effective_config_path,
            skip_tokenizer=skip_tokenizer,
            skip_basemodel=skip_basemodel,
            python_executable=python_executable,
        )
        print(f"[run] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=FINETUNE_CSV_DIR, check=True)
    finally:
        if temp_file_path is not None:
            try:
                temp_file_path.unlink(missing_ok=True)
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
    parser.add_argument("--python", default=None, help="Python executable path for the child training process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_required_runtime_deps()

    config_1h = Path(args.config_1h)
    config_5m = Path(args.config_5m)
    python_executable = resolve_python_executable(args.python)

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
        python_executable=python_executable,
    )

    print("=== Dual-timeframe training: 5m ===")
    run_training(
        config_path=config_5m,
        smoke_mode=args.smoke,
        skip_tokenizer=args.skip_tokenizer,
        skip_basemodel=args.skip_basemodel,
        python_executable=python_executable,
    )

    print("Dual-timeframe training completed.")


if __name__ == "__main__":
    main()
