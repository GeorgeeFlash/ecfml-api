from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ALLOWED_MODELS = {
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-4-6",
}


class Settings(BaseSettings):
    CLERK_JWKS_URL: str = ""
    ALLOWED_ORIGINS: Annotated[list[str], NoDecode] = ["http://localhost:3000"]
    DATABASE_URL: str = "sqlite:///./ecfml.db"
    DATA_DIR: str = "./data"
    MODELS_DIR: str = "./models"
    ACTIVE_LLM_MODEL: str = "openai/gpt-5.4"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: str = "true"
    CLERK_ISSUER: str = ""
    CLERK_AUDIENCE: str = ""

    def model_post_init(self, __context):
        if self.ACTIVE_LLM_MODEL not in ALLOWED_MODELS:
            raise ValueError(
                "ACTIVE_LLM_MODEL must be one of "
                f"{ALLOWED_MODELS}. Got: {self.ACTIVE_LLM_MODEL}"
            )

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def _parse_origins(cls, value):
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
