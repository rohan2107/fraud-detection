"""Centralized settings for the Fraud Detection project."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    model_path: Path = Field(default=Path("fraud_model.pkl"), alias="FRAUD_MODEL_PATH")
    data_path: Path = Field(default=Path("data/creditcard.csv"), alias="DATA_PATH")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


settings = Settings()
