from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("artifacts/models")
    checkpoints_dir: Path = Path("artifacts/checkpoints")
    text2cadquery_dir: Path = Path("data/text2cadquery")
    text2cad_bench_dir: Path = Path("data/text2cad_bench")

    # Vertex managed tuning: qwen/qwen3@qwen3-4b → Qwen/Qwen3-4B on Hugging Face.
    vertex_base_model: str = "qwen/qwen3@qwen3-4b"
    hf_model_id: str = "Qwen/Qwen3-4B"
    mlx_model_id: str = "mlx-community/Qwen3-4B-4bit"
    mlx_server_host: str = "127.0.0.1"
    mlx_server_port: int = 8080

    def resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path


settings = Settings()
