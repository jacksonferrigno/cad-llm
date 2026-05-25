"""Smoke tests for project bootstrap."""

from cad_llm import __version__
from cad_llm.config import settings


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_settings_defaults() -> None:
    assert settings.vertex_base_model == "qwen/qwen3@qwen3-4b"
    assert settings.hf_model_id == "Qwen/Qwen3-4B"
    assert settings.mlx_model_id == "mlx-community/Qwen3-4B-4bit"
    assert settings.mlx_model_id.endswith("4bit")
    assert "Qwen3-4B" in settings.mlx_model_id
    assert "Instruct" not in settings.hf_model_id
    assert settings.project_root.exists()
