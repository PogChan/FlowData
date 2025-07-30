"""
Base service classes and interfaces for the options flow classifier system.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from models.data_models import OptionsFlow, OptionsChainData, ClassificationRule


class BaseService(ABC):
    """Base class for all services in the system."""

    def __init__(self, config: Any):
        self.config = config


class DatabaseServiceInterface(ABC):
    """Interface for database operations."""

    @abstractmethod
    def save_options_flow(self, flow: OptionsFlow) -> bool:
        """Save options flow data to database."""
        pass

    @abstractmethod
    def get_options_flows(self, filters: Optional[Dict[str, Any]] = None) -> List[OptionsFlow]:
        """Retrieve options flow data from database."""
        pass

    @abstractmethod
    def save_classification_rule(self, rule: ClassificationRule) -> bool:
        """Save classification rule to database."""
        pass

    @abstractmethod
    def get_classification_rules(self, active_only: bool = True) -> List[ClassificationRule]:
        """Retrieve classification rules from database."""
        pass

    @abstractmethod
    def save_options_chain_data(self, chain_data: OptionsChainData) -> bool:
        """Save options chain data to cache table."""
        pass

    @abstractmethod
    def get_cached_options_data(self, symbol: str, expiration_date: str,
                               strike: float, contract_type: str) -> Optional[OptionsChainData]:
        """Retrieve cached options chain data if not expired."""
        pass

    @abstractmethod
    def clean_expired_cache(self) -> int:
        """Clean expired options chain cache entries."""
        pass

    @abstractmethod
    def update_outcome(self, flow_id: str, actual_outcome: str) -> bool:
        """Update the actual outcome for a specific options flow."""
        pass

    @abstractmethod
    def get_classification_accuracy(self, classification: str) -> Dict[str, float]:
        """Get accuracy metrics for a specific classification."""
        pass


class APIServiceInterface(ABC):
    """Interface for external API services."""

    @abstractmethod
    def fetch_options_chain(self, symbol: str) -> Dict[str, OptionsChainData]:
        """Fetch options chain data from external API."""
        pass


class ClassificationServiceInterface(ABC):
    """Interface for trade classification services."""

    @abstractmethod
    def classify_trade(self, trades: List[OptionsFlow]) -> str:
        """Classify a trade or set of trades."""
        pass
