from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

FINETUNE_DIR = Path(__file__).resolve().parents[1] / "finetune"
if str(FINETUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNE_DIR))

from bybit_data_preprocess import normalize_ohlcv_frame, split_symbol_frames
from config_bybit import BybitConfig, get_bybit_config_overrides


def test_normalize_ohlcv_frame_creates_vol_and_amt():
    index = pd.date_range("2025-01-01", periods=3, freq="D")
    frame = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5],
            "close": [1.2, 2.2, 3.2],
            "volume": [10, 20, 30],
        },
        index=index,
    )

    normalized = normalize_ohlcv_frame(frame, "sample")

    assert list(normalized.columns) == ["open", "high", "low", "close", "vol", "amt"]
    assert normalized.index.name == "datetime"
    assert normalized.loc[pd.Timestamp("2025-01-01"), "vol"] == 10
    assert normalized.loc[pd.Timestamp("2025-01-01"), "amt"] == pytest.approx(12.0)


def test_split_symbol_frames_uses_explicit_dates_with_overlap():
    index = pd.date_range("2025-06-01", "2026-04-15", freq="D")
    frame = pd.DataFrame(
        {
            "open": range(len(index)),
            "high": range(len(index)),
            "low": range(len(index)),
            "close": range(len(index)),
            "vol": range(len(index)),
            "amt": range(len(index)),
        },
        index=index,
    )

    train_data, val_data = split_symbol_frames({"BTCUSDT_60m": frame})

    assert train_data["BTCUSDT_60m"].index.max() <= pd.Timestamp("2025-09-30 23:59:59")
    assert val_data["BTCUSDT_60m"].index.min() >= pd.Timestamp("2025-07-01 00:00:00")
    assert val_data["BTCUSDT_60m"].index.max() <= pd.Timestamp("2026-03-31 23:59:59")
    assert not train_data["BTCUSDT_60m"].empty
    assert not val_data["BTCUSDT_60m"].empty


def test_bybit_config_overrides_match_requested_profile():
    overrides = get_bybit_config_overrides()

    assert overrides["dataset_path"] == "./data/bybit_multi"
    assert overrides["pretrained_predictor_path"] == "NeoQuasar/Kronos-base"
    assert overrides["batch_size"] == 64
    assert overrides["n_train_iter"] == 4000 * 64


def test_bybit_config_direct_instantiation_sets_bybit_paths():
    cfg = BybitConfig()

    assert cfg.dataset_path == "./data/bybit_multi"
    assert cfg.save_path == "./outputs/bybit_multi"
    assert cfg.pretrained_predictor_path == "NeoQuasar/Kronos-base"
    assert cfg.tokenizer_save_folder_name == "bybit_tokenizer"
