from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CORPUS_", env_file=".env", extra="ignore")

    data_dir: Path = Path(".corpus_data")
    db_path: Path | None = None
    max_concurrent_pages: int = 4
    chunk_max_chars: int = 900
    chunk_overlap: int = 120

    @property
    def sqlite_path(self) -> Path:
        if self.db_path is not None:
            return self.db_path
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / "catalog.sqlite3"

    @property
    def blob_dir(self) -> Path:
        d = self.data_dir / "blobs"
        d.mkdir(parents=True, exist_ok=True)
        return d


def get_settings() -> Settings:
    return Settings()
