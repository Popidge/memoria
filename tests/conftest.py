from __future__ import annotations

from pathlib import Path

import pytest

from memoria.config import EngineConfig
from memoria.engine import MemoryEngine


@pytest.fixture
def engine(tmp_path: Path) -> MemoryEngine:
    db_path = tmp_path / "memoria.db"
    config = EngineConfig(database_url=f"sqlite:///{db_path}", use_sentence_transformers=False)
    return MemoryEngine(config=config)
