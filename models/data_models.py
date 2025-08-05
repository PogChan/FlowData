"""
Data models for the options flow classifier system.
"""
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List, Dict, Any


@dataclass
class OptionsFlow:
    """
    Enhanced options flow data model with classification and outcome tracking.
    Maps to the enhanced options_flow table schema.
    """
    id: str
    created_datetime: datetime
    symbol: str
    buy_sell: str
    call_put: str
    strike: float
    spot: float
    expiration_date: date
    premium: float
    volume: int
    open_interest: int
    price: float
    side: str
    color: str
    set_count: int
    implied_volatility: float
    dte: int
    er_flag: bool
    # Enhanced classification and outcome fields
    classification: Optional[str] = None
    expected_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None  # Values: "FOREVER DISCOUNTED", "DISCOUNT THEN PUMP", "FOREVER PUMPED", "PUMP THEN DISCOUNT", "MANUAL REVIEW"
    trade_value: float = 0.0
    confidence_score: float = 0.0
    # Multi-leg trade identification
    trade_signature: Optional[str] = None
    trade_group_id: Optional[str] = None
    # Volatility analysis
    historical_volatility: Optional[float] = None
    implied_volatility_atm: Optional[float] = None
    volatility_flag: Optional[str] = None
    volatility_premium: Optional[float] = None
    # Additional flow screener fields
    direction: Optional[str] = None
    moneiness: Optional[str] = None
    pc_ratio: Optional[float] = None
    earnings_date: Optional[date] = None
    sector: Optional[str] = None
    market_cap: Optional[str] = None
    stock_etf: Optional[str] = None
    uoa: Optional[str] = None
    weekly: Optional[bool] = None
    type: Optional[str] = None
    # Stock movement tracking
    stock_movement_1d: Optional[float] = None
    stock_movement_3d: Optional[float] = None
    stock_movement_7d: Optional[float] = None
    stock_movement_30d: Optional[float] = None
    movement_direction: Optional[str] = None
    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class OptionsChainData:
    """
    Options chain data model for real-time market data from Polygon API.
    Maps to the options_chain_cache table schema.
    """
    symbol: str
    expiration_date: date  # Changed from 'expiration' to match DB schema
    strike: float
    contract_type: str  # 'call' or 'put'
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    last_price: Optional[float] = None  # Changed from 'close' to match DB schema
    bid: Optional[float] = None
    ask: Optional[float] = None
    cached_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


@dataclass
class ClassificationRule:
    """
    Dynamic classification rule model for trade analysis.
    Maps to the classification_rules table schema.
    """
    rule_id: str
    name: str
    description: Optional[str]
    classification_logic: Dict[str, Any]
    expected_outcome: str
    result_keywords: List[str]
    is_active: bool = True
    success_rate: Optional[float] = None
    created_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
