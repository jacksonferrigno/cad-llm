"""Smoke tests for project bootstrap."""

from cad_llm import __version__
from cad_llm.config import settings


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_settings_defaults() -> None:
    assert settings.vertex_base_model == "Qwen/Qwen3.5-4B"
    assert settings.hf_model_id == "Qwen/Qwen3.5-4B"
    assert settings.mlx_model_id == "Qwen/Qwen3.5-4B"
    assert "Qwen3.5-4B" in settings.mlx_model_id
    assert "Instruct" not in settings.hf_model_id
    assert settings.project_root.exists()
