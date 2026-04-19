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


def _normalize_path_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((REPO_ROOT / candidate).resolve())


def normalize_config_paths(data: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(data)

    data_section = dict(cfg.get("data") or {})
    if "data_path" in data_section:
        data_section["data_path"] = _normalize_path_like(data_section["data_path"])
    cfg["data"] = data_section

    model_paths = dict(cfg.get("model_paths") or {})
    for key in ("pretrained_tokenizer", "pretrained_predictor", "base_path", "finetuned_tokenizer", "base_save_path"):
        if key in model_paths and model_paths[key]:
            current = str(model_paths[key])
            # Keep Hugging Face IDs untouched (e.g., NeoQuasar/Kronos-small)
            if "/" in current and not current.startswith(".") and not current.startswith("..") and not current.startswith("~") and not current.startswith("C:") and not current.startswith("/"):
                continue
            model_paths[key] = _normalize_path_like(current)
    cfg["model_paths"] = model_paths

    return cfg


def _count_csv_rows(file_path: Path) -> int:
    with file_path.open("r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f)
    return max(0, row_count - 1)  # minus header


def _compute_split_lengths(total_rows: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = int(total_rows * train_ratio)
    val_end = int(total_rows * (train_ratio + val_ratio))
    train_len = train_end
    val_len = max(0, val_end - train_end)
    return train_len, val_len


def adapt_config_for_dataset(cfg: dict[str, Any], min_train_ratio: float = 0.60) -> tuple[dict[str, Any], list[str]]:
    adapted = dict(cfg)
    data_section = dict(adapted.get("data") or {})

    data_path_raw = data_section.get("data_path")
    if not data_path_raw:
        return adapted, []

    data_path = Path(str(data_path_raw))
    if not data_path.exists():
        return adapted, []

    total_rows = _count_csv_rows(data_path)
    if total_rows <= 0:
        return adapted, []

    lookback = int(data_section.get("lookback_window", 512))
    predict_window = int(data_section.get("predict_window", 48))
    max_context = int(data_section.get("max_context", lookback))

    train_ratio = float(data_section.get("train_ratio", 0.9))
    val_ratio = float(data_section.get("val_ratio", 0.1))
    test_ratio = float(data_section.get("test_ratio", 0.0))

    window = lookback + predict_window + 1
    messages: list[str] = []

    train_len, val_len = _compute_split_lengths(total_rows, train_ratio, val_ratio)

    # Step 1: try to preserve window by increasing validation ratio when possible
    if val_len < window:
        required_val_ratio = (window + 16) / max(total_rows, 1)
        target_val_ratio = max(val_ratio, min(0.35, required_val_ratio))
        target_train_ratio = 1.0 - test_ratio - target_val_ratio

        if target_train_ratio >= min_train_ratio and target_val_ratio > val_ratio:
            train_ratio = target_train_ratio
            val_ratio = target_val_ratio
            data_section["train_ratio"] = round(train_ratio, 4)
            data_section["val_ratio"] = round(val_ratio, 4)
            train_len, val_len = _compute_split_lengths(total_rows, train_ratio, val_ratio)
            messages.append(
                "Adjusted split ratios for dataset size: "
                f"train_ratio={data_section['train_ratio']}, val_ratio={data_section['val_ratio']}, test_ratio={test_ratio}"
            )

    # Step 2: if still insufficient, reduce window settings conservatively
    if val_len < window:
        max_window_for_val = max(2, val_len - 1)
        if max_window_for_val < 2:
            raise ValueError(
                "Validation split is too small even after adaptation. "
                f"total_rows={total_rows}, val_len={val_len}. Increase dataset size."
            )

        new_predict = min(predict_window, max(8, max_window_for_val // 6))
        if new_predict >= max_window_for_val:
            new_predict = max(1, max_window_for_val - 1)
        new_lookback = max(8, max_window_for_val - new_predict)

        if new_lookback < lookback or new_predict < predict_window:
            data_section["lookback_window"] = new_lookback
            data_section["predict_window"] = new_predict
            data_section["max_context"] = max(max_context, new_lookback)
            lookback = new_lookback
            predict_window = new_predict
            window = lookback + predict_window + 1
            messages.append(
                "Reduced windows for dataset size: "
                f"lookback_window={lookback}, predict_window={predict_window}, max_context={data_section['max_context']}"
            )

    # Final guard
    train_len, val_len = _compute_split_lengths(total_rows, train_ratio, val_ratio)
    train_samples = train_len - (lookback + predict_window + 1) + 1
    val_samples = val_len - (lookback + predict_window + 1) + 1
    if train_samples <= 0 or val_samples <= 0:
        raise ValueError(
            "Auto-adaptation could not produce valid sample counts. "
            f"train_samples={train_samples}, val_samples={val_samples}, total_rows={total_rows}, "
            f"lookback={lookback}, predict={predict_window}, train_ratio={train_ratio}, val_ratio={val_ratio}."
        )

    adapted["data"] = data_section
    return adapted, messages


def apply_smoke_overrides(data: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(data)

    data_section = dict(cfg.get("data") or {})
    # Make smoke mode resilient on small datasets (e.g., ~2000 rows)
    data_section["lookback_window"] = min(int(data_section.get("lookback_window", 512)), 128)
    data_section["predict_window"] = min(int(data_section.get("predict_window", 48)), 16)
    data_section["max_context"] = max(int(data_section.get("max_context", 512)), data_section["lookback_window"])
    data_section["train_ratio"] = 0.75
    data_section["val_ratio"] = 0.20
    data_section["test_ratio"] = 0.05
    cfg["data"] = data_section

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
    python_executable: str,
    auto_adapt_data: bool,
) -> None:
    effective_config_path = config_path
    temp_file_path: Path | None = None

    try:
        original = load_yaml(config_path)
        normalized = normalize_config_paths(original)
        effective_cfg = apply_smoke_overrides(normalized) if smoke_mode else normalized

        if auto_adapt_data:
            effective_cfg, adaptation_messages = adapt_config_for_dataset(effective_cfg)
            for msg in adaptation_messages:
                print(f"[auto-adapt] {msg}")

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as temp_file:
            temp_file_path = Path(temp_file.name)
        dump_yaml(temp_file_path, effective_cfg)
        effective_config_path = temp_file_path

        cmd = build_train_command(
            config_path=effective_config_path,
            skip_tokenizer=skip_tokenizer,
            skip_basemodel=skip_basemodel,
            python_executable=python_executable,
        )
        print(f"[run] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
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
    parser.add_argument(
        "--auto-adapt-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically adapt split/window settings to dataset size before launching training",
    )
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
        auto_adapt_data=args.auto_adapt_data,
    )

    print("=== Dual-timeframe training: 5m ===")
    run_training(
        config_path=config_5m,
        smoke_mode=args.smoke,
        skip_tokenizer=args.skip_tokenizer,
        skip_basemodel=args.skip_basemodel,
        python_executable=python_executable,
        auto_adapt_data=args.auto_adapt_data,
    )

    print("Dual-timeframe training completed.")


if __name__ == "__main__":
    main()
