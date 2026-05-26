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
    workspace_dir: Path = Path("workspace")
    projects_dir: Path = Path("workspace/projects")
    text2cadquery_dir: Path = Path("data/text2cadquery")
    text2cad_bench_dir: Path = Path("data/text2cad_bench")
    cadquery_docs_dir: Path = Path("data/cadquery-latest")
    docs_db_url: str = "postgresql://cadllm:cadllm@127.0.0.1:5433/cadllm"
    docs_collection_name: str = "cadquery_docs"
    docs_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    docs_chunks_cache: Path = Path("artifacts/docs/cadquery_chunks.pkl")

    # Local default: Qwen3.5-4B from Hugging Face.
    vertex_base_model: str = "Qwen/Qwen3.5-4B"
    hf_model_id: str = "Qwen/Qwen3.5-4B"
    mlx_model_id: str = "Qwen/Qwen3.5-4B"
    mlx_server_host: str = "127.0.0.1"
    mlx_server_port: int = 8080

    def resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path


settings = Settings()
