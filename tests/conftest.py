from __future__ import annotations

import pytest

from watchinu.config import AppConfig

from .helpers import make_config


@pytest.fixture
def cfg() -> AppConfig:
    return make_config()
