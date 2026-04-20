from __future__ import annotations

from config import Config


def get_bybit_config_overrides() -> dict[str, object]:
    """Return the Bybit fine-tuning overrides without instantiating Config again."""

    return {
        "dataset_path": "./data/bybit_multi",
        "lookback_window": 512,
        "predict_window": 48,
        "pretrained_tokenizer_path": "NeoQuasar/Kronos-Tokenizer-base",
        "pretrained_predictor_path": "NeoQuasar/Kronos-base",
        "epochs": 30,
        "batch_size": 64,
        "n_train_iter": 4000 * 64,
        "n_val_iter": 600 * 64,
        "adam_weight_decay": 0.15,
        "use_comet": False,
        "save_path": "./outputs/bybit_multi",
        "tokenizer_save_folder_name": "bybit_tokenizer",
        "predictor_save_folder_name": "bybit_predictor",
        "backtest_save_folder_name": "bybit_backtest",
        "train_time_range": ["2011-01-01", "2025-09-30"],
        "val_time_range": ["2025-07-01", "2026-03-31"],
    }


class BybitConfig(Config):
    def __init__(self):
        super().__init__()
        for key, value in get_bybit_config_overrides().items():
            setattr(self, key, value)
