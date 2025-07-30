"""
Test script for the enhanced database service.
This tests the database service interface without requiring actual database connection.
"""

from datetime import datetime, date
from models.data_models import OptionsFlow, OptionsChainData, ClassificationRule
from services.database_service import SupabaseService
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockSupabaseClient:
    """Mock Supabase client for testing without actual database connection."""

    def __init__(self):
        self.data_store = {
            'options_flow': [],
            'classification_rules': [],
            'options_chain_cache': []
        }

    def table(self, table_name):
        return MockTable(table_name, self.data_store)

class MockTable:
    """Mock table for testing database operations."""

    def __init__(self, table_name, data_store):
        self.table_name = table_name
        self.data_store = data_store
        self.query_filters = {}

    def select(self, columns):
        return self

    def upsert(self, data):
        self.data_store[self.table_name].append(data)
        return MockResult([data])

    def eq(self, column, value):
        self.query_filters[column] = value
        return self

    def execute(self):
        # Simple mock - return all data for the table
        return MockResult(self.data_store[self.table_name])

class MockResult:
    """Mock result for database operations."""

    def __init__(self, data):
        self.data = data

def test_database_service():
    """Test the database service with mock client."""

    print("Testing Enhanced Database Service...")

    # Create mock client and service
    mock_client = MockSupabaseClient()
    db_service = SupabaseService(mock_client)

    # Test 1: Save and retrieve options flow
    print("\n1. Testing OptionsFlow operations...")

    test_flow = OptionsFlow(
        id="test-flow-001",
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
        expected_hypothesis="CREATED WALL ON BUY SIDE TRADE",
        trade_value=55000.0,
        confidence_score=0.85
    )

    # Save flow
    success = db_service.save_options_flow(test_flow)
    print(f"✓ Save options flow: {'SUCCESS' if success else 'FAILED'}")

    # Test 2: Save classification rule
    print("\n2. Testing ClassificationRule operations...")

    test_rule = ClassificationRule(
        rule_id="rule-001",
        name="ATM SAME STRIKE",
        description="Both legs have the same ATM strike",
        classification_logic={"strike_relationship": "same", "moneyness": "ATM"},
        expected_hypothesis="CREATED WALL ON BUY SIDE TRADE",
        result_keywords=["wall", "buy side"],
        is_active=True,
        success_rate=0.75
    )

    success = db_service.save_classification_rule(test_rule)
    print(f"✓ Save classification rule: {'SUCCESS' if success else 'FAILED'}")

    # Test 3: Save options chain data
    print("\n3. Testing OptionsChainData operations...")

    test_chain_data = OptionsChainData(
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

    success = db_service.save_options_chain_data(test_chain_data)
    print(f"✓ Save options chain data: {'SUCCESS' if success else 'FAILED'}")

    # Test 4: Update outcome
    print("\n4. Testing outcome update...")

    success = db_service.update_outcome("test-flow-001", "Forever Pumped")
    print(f"✓ Update outcome: {'SUCCESS' if success else 'FAILED'}")

    # Test 5: Clean expired cache
    print("\n5. Testing cache cleanup...")

    cleaned = db_service.clean_expired_cache()
    print(f"✓ Clean expired cache: {cleaned} entries cleaned")

    print("\n✓ All database service tests completed!")
    print("\nNote: These tests use a mock client. To test with real Supabase:")
    print("1. Run the database migrations first")
    print("2. Update this script to use a real Supabase client")
    print("3. Ensure proper authentication is configured")

if __name__ == "__main__":
    test_database_service()
