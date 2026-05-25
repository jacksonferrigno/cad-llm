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

    mlx_model_id: str = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    mlx_server_host: str = "127.0.0.1"
    mlx_server_port: int = 8080

    def resolve(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path


settings = Settings()
