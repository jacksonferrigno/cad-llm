"""Smoke tests for project bootstrap."""

from cad_llm import __version__
from cad_llm.config import settings


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_settings_defaults() -> None:
    assert settings.mlx_model_id.endswith("4bit")
    assert settings.project_root.exists()
