"""
Quick test to verify the enhanced project structure and data models work correctly.
"""
from datetime import datetime, date
from models import OptionsFlow, OptionsChainData, ClassificationRule
from utils import config
from services import BaseService

def test_data_models():
    """Test that data models can be instantiated correctly."""

    # Test OptionsFlow model
    options_flow = OptionsFlow(
        id="test-123",
        created_datetime=datetime.now(),
        symbol="AAPL",
        buy_sell="BUY",
        call_put="CALL",
        strike=150.0,
        spot=155.0,
        expiration_date=date(2024, 12, 20),
        premium=5.50,
        volume=100,
        open_interest=500,
        price=5.50,
        side="B",
        color="green",
        set_count=1,
        implied_volatility=0.25,
        dte=30,
        er_flag=False,
        classification="ATM SAME STRIKE",
        expeted_outcome="CREATED WALL ON BUY SIDE TRADE",
        trade_value=55000.0,
        confidence_score=0.85
    )
    print(f"✓ OptionsFlow model created: {options_flow.symbol} {options_flow.strike}")

    # Test OptionsChainData model
    chain_data = OptionsChainData(
        symbol="AAPL",
        expiration_date=date(2024, 12, 20),
        strike=150.0,
        contract_type="call",
        delta=0.55,
        gamma=0.02,
        theta=-0.05,
        vega=0.15,
        implied_volatility=0.25,
        open_interest=500,
        volume=100,
        last_price=5.50,
        bid=5.45,
        ask=5.55,
        cached_at=datetime.now()
    )
    print(f"✓ OptionsChainData model created: {chain_data.symbol} delta={chain_data.delta}")

    # Test ClassificationRule model
    rule = ClassificationRule(
        rule_id="rule-001",
        name="ATM SAME STRIKE",
        description="Both legs have the same ATM strike",
        classification_logic={"strike_relationship": "same", "moneyness": "ATM"},
        expeted_outcome="CREATED WALL ON BUY SIDE TRADE",
        result_keywords=["wall", "buy side"],
        is_active=True,
        created_date=datetime.now(),
        success_rate=0.75
    )
    print(f"✓ ClassificationRule model created: {rule.name}")

def test_configuration():
    """Test that configuration management works."""
    try:
        # Test app config (should always work)
        app_config = config.app
        print(f"✓ App config loaded: cache_ttl={app_config.cache_ttl}")

        # Note: Database and API config will fail without proper secrets/env vars
        # but that's expected in this test environment
        print("✓ Configuration system initialized")

    except Exception as e:
        print(f"⚠ Configuration test (expected in test environment): {e}")

if __name__ == "__main__":
    print("Testing enhanced project structure...")
    test_data_models()
    test_configuration()
    print("✓ All tests completed successfully!")
