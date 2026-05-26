from pathlib import Path

from cad_llm.inference.generate import VERTEX_SFT_MLX_MODEL, resolve_model_id


def test_resolve_model_id_keeps_hf_repo() -> None:
    assert resolve_model_id("mlx-community/Qwen3-4B-4bit") == "mlx-community/Qwen3-4B-4bit"


def test_resolve_model_id_from_project_root() -> None:
    if not VERTEX_SFT_MLX_MODEL.exists():
        return

    resolved = resolve_model_id("artifacts/models/qwen3-4b-vertex-sft-4bit")
    assert Path(resolved).is_absolute()
    assert Path(resolved).exists()
