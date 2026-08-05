"""Settings loaded from the environment (and `.env`).

Anything that varies between the laptop, CI and the box in the closet lives here,
so no module has to reach for `os.environ` itself.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration. Field names map case-insensitively to env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Aviation Weather Center -------------------------------------------
    # AWC needs no API key but requires a custom, identifying User-Agent and
    # rate-limits to roughly 100 requests/minute.
    awc_base_url: str = "https://aviationweather.gov/api/data"
    awc_user_agent: str = "turbulence-agent/0.1"
    awc_timeout_seconds: float = 30.0

    # --- FlightAware AeroAPI -----------------------------------------------
    aeroapi_key: str | None = None

    # --- Anthropic ----------------------------------------------------------
    anthropic_api_key: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings singleton. Pass an explicit `Settings` in tests."""
    return Settings()
