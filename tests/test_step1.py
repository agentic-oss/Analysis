import json
import yaml
from pathlib import Path
import pytest
from src.ingestion.yfinance_provider import YFinanceProvider

def test_config_loading():
    inst_path = Path("config/instruments.json")
    cfg_path = Path("config/config.yaml")

    assert inst_path.exists()
    assert cfg_path.exists()

    with open(inst_path) as f:
        inst = json.load(f)
        assert "instruments" in inst
        assert len(inst["instruments"]) > 0

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
        assert "system" in cfg
        assert "ingestion" in cfg

def test_provider_initialization():
    provider = YFinanceProvider(max_retries=1)
    assert provider.max_retries == 1
