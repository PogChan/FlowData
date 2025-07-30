"""
Configuration management for the options flow classifier system.
Handles API keys, database connections, and application settings.
"""
import os
from typing import Optional
import streamlit as st
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str
    key: str


@dataclass
class APIConfig:
    """External API configuration settings."""
    polygon_api_key: str


@dataclass
class AppConfig:
    """Application configuration settings."""
    cache_ttl: int = 300  # 5 minutes default
    max_file_size_mb: int = 50
    delta_threshold: float = 0.18
    api_rate_limit_delay: float = 0.1


class ConfigManager:
    """
    Centralized configuration management for the application.
    Handles loading from environment variables and Streamlit secrets.
    """

    def __init__(self):
        self._db_config: Optional[DatabaseConfig] = None
        self._api_config: Optional[APIConfig] = None
        self._app_config: AppConfig = AppConfig()

    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        if self._db_config is None:
            self._db_config = self._load_database_config()
        return self._db_config

    @property
    def api(self) -> APIConfig:
        """Get API configuration."""
        if self._api_config is None:
            self._api_config = self._load_api_config()
        return self._api_config

    @property
    def app(self) -> AppConfig:
        """Get application configuration."""
        return self._app_config

    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration from Streamlit secrets or environment."""
        try:
            # Try Streamlit secrets first
            url = st.secrets["supabase"]["url"]
            key = st.secrets["supabase"]["key"]
        except (KeyError, FileNotFoundError):
            # Fallback to environment variables
            url = os.getenv("SUPA_URL")
            key = os.getenv("SUPA_KEY")

            if not url or not key:
                raise ValueError(
                    "Database configuration not found. Please set SUPA_URL and SUPA_KEY "
                    "in environment variables or Streamlit secrets."
                )

        return DatabaseConfig(url=url, key=key)

    def _load_api_config(self) -> APIConfig:
        """Load API configuration from Streamlit secrets or environment."""
        try:
            # Try Streamlit secrets first
            polygon_key = st.secrets["polygon"]["api_key"]
        except (KeyError, FileNotFoundError):
            # Fallback to environment variables
            polygon_key = os.getenv("POLYGON")

            if not polygon_key:
                raise ValueError(
                    "Polygon API key not found. Please set POLYGON in environment "
                    "variables or add polygon.api_key to Streamlit secrets."
                )

        return APIConfig(polygon_api_key=polygon_key)

    def update_app_config(self, **kwargs) -> None:
        """Update application configuration settings."""
        for key, value in kwargs.items():
            if hasattr(self._app_config, key):
                setattr(self._app_config, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")


# Global configuration instance
config = ConfigManager()
