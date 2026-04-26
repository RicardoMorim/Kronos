from __future__ import annotations
import os

from config import Config


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def get_bybit_config_overrides() -> dict[str, object]:
    """Return the Bybit fine-tuning overrides without instantiating Config again."""

    # Safer defaults for single-GPU consumer setups (Windows-friendly).
    predictor_batch_size = _env_int("KRONOS_PREDICTOR_BATCH_SIZE", 4)
    predictor_accumulation_steps = _env_int("KRONOS_PREDICTOR_ACC_STEPS", 16)

    return {
        "dataset_path": "./data/bybit_multi",
        "lookback_window": _env_int("KRONOS_LOOKBACK_WINDOW", 384),
        "predict_window": _env_int("KRONOS_PREDICT_WINDOW", 24),
        "max_context": 512,
        "pretrained_tokenizer_path": "NeoQuasar/Kronos-Tokenizer-base",
        "pretrained_predictor_path": _env_str("KRONOS_PRETRAINED_PREDICTOR", "NeoQuasar/Kronos-base"),
        "finetuned_tokenizer_path": "./outputs/bybit_multi/bybit_tokenizer/checkpoints/best_model",
        "finetuned_predictor_path": "./outputs/bybit_multi/bybit_predictor/checkpoints/best_model",
        "epochs": 30,
        "batch_size": _env_int("KRONOS_TOKENIZER_BATCH_SIZE", 16),
        "predictor_batch_size": predictor_batch_size,
        "predictor_accumulation_steps": predictor_accumulation_steps,
        "accumulation_steps": 1,
        "num_workers": 0,
        "predictor_num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "n_train_iter": 4000 * predictor_batch_size,
        "n_val_iter": 600 * predictor_batch_size,
        "predictor_learning_rate": 1e-4,
        "adam_weight_decay": 0.25,
        "use_comet": False,
        "use_amp": True,
        "predictor_tokenizer_device": _env_str("KRONOS_PREDICTOR_TOKENIZER_DEVICE", "cuda"),
        "empty_cuda_cache_each_epoch": True,
        "save_path": "./outputs/bybit_multi",
        "tokenizer_save_folder_name": "bybit_tokenizer",
        "predictor_save_folder_name": "bybit_predictor",
        "backtest_save_folder_name": "bybit_backtest",
        "train_time_range": ["2011-01-01", "2025-09-30"],
        "val_time_range": ["2025-07-01", "2026-03-31"],
        # Early stopping: stop after N epochs without val improvement.
        "early_stopping_patience": 5,
        # Freeze backbone during predictor fine-tuning (set True to save memory/compute).
        "predictor_freeze_backbone": False,
        "predictor_unfreeze_last_n_blocks": 2,
    }


class BybitConfig(Config):
    def __init__(self):
        super().__init__()
        for key, value in get_bybit_config_overrides().items():
            setattr(self, key, value)
