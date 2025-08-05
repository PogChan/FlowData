"""
Initialize default classification rules in the database.
Run this script once to set up the basic classification rules.
"""

import uuid
from datetime import datetime
from supabase import create_client
from utils.config import config
from models.data_models import ClassificationRule
from services.rules_engine import RulesEngine

def initialize_default_rules():
    """Initialize default classification rules."""

    # Initialize services
    supabase_client = create_client(config.database.url, config.database.key)
    rules_engine = RulesEngine(supabase_client)

    # Define default rules
    default_rules = [
        {
            'name': 'ATM SAME STRIKE',
            'description': 'Both legs have identical at-the-money strikes',
            'classification_logic': {'strike_relationship': {'==': 'atm_same'}},
            'expected_outcome': 'High volatility expected - direction uncertain',
            'result_keywords': ['volatility', 'uncertain', 'gamma']
        },
        {
            'name': 'ITM SAME STRIKE',
            'description': 'Both legs have identical in-the-money strikes',
            'classification_logic': {'strike_relationship': {'==': 'itm_same'}},
            'expected_outcome': 'Conservative play - likely to hold value',
            'result_keywords': ['conservative', 'hold', 'value']
        },
        {
            'name': 'OTM SAME STRIKE',
            'description': 'Both legs have identical out-of-the-money strikes',
            'classification_logic': {'strike_relationship': {'==': 'otm_same'}},
            'expected_outcome': 'High risk/reward - needs significant move',
            'result_keywords': ['risk', 'reward', 'move']
        },
        {
            'name': 'WITHIN RANGE OTMS',
            'description': 'Both legs within 0.18 delta range of buy side direction',
            'classification_logic': {'delta_range': {'==': 'within_otm'}},
            'expected_outcome': 'Moderate risk - needs directional move',
            'result_keywords': ['moderate', 'directional', 'range']
        },
        {
            'name': 'OUTSIDE RANGE OTMS',
            'description': 'Either leg outside 0.18 delta range',
            'classification_logic': {'delta_range': {'==': 'outside_otm'}},
            'expected_outcome': 'High risk - needs large directional move',
            'result_keywords': ['high risk', 'large move', 'outside']
        },
        {
            'name': 'BLANK SIDE',
            'description': 'Missing or null side values requiring manual review',
            'classification_logic': {'side': {'==': 'blank'}},
            'expected_outcome': 'Manual review required',
            'result_keywords': ['manual', 'review', 'blank']
        },
        {
            'name': 'WITHIN RANGE ITMS',
            'description': 'ITM strikes on both sides within 0.18 delta range',
            'classification_logic': {'delta_range': {'==': 'within_itm'}},
            'expected_outcome': 'Conservative - likely profitable',
            'result_keywords': ['conservative', 'profitable', 'itm']
        },
        {
            'name': 'STRADDLE',
            'description': 'Simultaneous buy call and buy put positions',
            'classification_logic': {'pattern': {'==': 'straddle'}},
            'expected_outcome': 'Volatility play - expects big move either direction',
            'result_keywords': ['volatility', 'big move', 'either direction']
        },
        {
            'name': 'NEGATIVE ITM',
            'description': 'Sell side aggregate value exceeds buy side value',
            'classification_logic': {'value_relationship': {'==': 'negative_itm'}},
            'expected_outcome': 'Bearish bias - expects downward pressure',
            'result_keywords': ['bearish', 'downward', 'pressure']
        },
        {
            'name': 'DEBIT AND SELL',
            'description': 'Debit spread combined with opposite sell leg',
            'classification_logic': {'pattern': {'==': 'debit_and_sell'}},
            'expected_outcome': 'Complex strategy - mixed directional bias',
            'result_keywords': ['complex', 'mixed', 'directional']
        },
        {
            'name': 'UNCLASSIFIED',
            'description': 'Fallback for trades not matching any pattern',
            'classification_logic': {'pattern': {'==': 'unclassified'}},
            'expected_outcome': 'Pattern unclear - manual analysis needed',
            'result_keywords': ['unclear', 'manual', 'analysis']
        }
    ]

    # Add rules to database
    added_count = 0
    for rule_data in default_rules:
        rule = ClassificationRule(
            rule_id=str(uuid.uuid4()),
            name=rule_data['name'],
            description=rule_data['description'],
            classification_logic=rule_data['classification_logic'],
            expected_outcome=rule_data['expected_outcome'],
            result_keywords=rule_data['result_keywords'],
            is_active=True,
            created_date=datetime.now(),
            updated_date=datetime.now()
        )

        if rules_engine.add_rule(rule):
            print(f"✓ Added rule: {rule.name}")
            added_count += 1
        else:
            print(f"✗ Failed to add rule: {rule.name}")

    print(f"\nInitialization complete! Added {added_count}/{len(default_rules)} rules.")

if __name__ == "__main__":
    initialize_default_rules()
