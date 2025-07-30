-- Migration: Create classification_rules table for dynamic rule management
-- Requirements: 6.1, 6.2, 6.3, 6.4, 6.5

-- Create classification_rules table
CREATE TABLE classification_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    classification_logic JSONB NOT NULL,
    expected_hypothesis TEXT NOT NULL,
    result_keywords TEXT[] NOT NULL,
    is_active BOOLEAN DEFAULT true,
    success_rate DECIMAL(5,4),
    created_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_date TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_classification_rules_name ON classification_rules(name);
CREATE INDEX idx_classification_rules_active ON classification_rules(is_active);
CREATE INDEX idx_classification_rules_success_rate ON classification_rules(success_rate);

-- Create trigger for updated_date timestamp
CREATE TRIGGER update_classification_rules_updated_date
    BEFORE UPDATE ON classification_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE classification_rules IS 'Dynamic classification rules for trade analysis';
COMMENT ON COLUMN classification_rules.classification_logic IS 'JSON configuration defining rule evaluation logic';
COMMENT ON COLUMN classification_rules.result_keywords IS 'Array of keywords associated with this rule for search and analysis';
COMMENT ON COLUMN classification_rules.success_rate IS 'Historical success rate of this rule (0.0 to 1.0)';

-- Insert default classification rules
INSERT INTO classification_rules (name, description, classification_logic, expected_hypothesis, result_keywords) VALUES
('ATM SAME STRIKE', 'Both legs have identical at-the-money strikes',
 '{"type": "same_strike", "moneyness": "ATM", "delta_threshold": 0.18}',
 'CREATED WALL ON BUY SIDE TRADE',
 ARRAY['wall', 'buy side']),

('ITM SAME STRIKE', 'Both legs have identical in-the-money strikes',
 '{"type": "same_strike", "moneyness": "ITM", "delta_threshold": 0.18}',
 'GO TO BUY STRIKE, PUMP BUY SIDE',
 ARRAY['pump', 'buy strike']),

('OTM SAME STRIKE', 'Both legs have identical out-of-the-money strikes',
 '{"type": "same_strike", "moneyness": "OTM", "delta_threshold": 0.18}',
 'GO TO BUY STRIKE',
 ARRAY['buy strike']),

('WITHIN RANGE OTMS', 'Both legs within 0.18 delta range of buy side direction',
 '{"type": "delta_range", "range": "within", "moneyness": "OTM", "delta_threshold": 0.18}',
 'DISCOUNT SELL SIDE, RUN BUY SIDE',
 ARRAY['discount', 'sell side', 'run buy side']),

('OUTSIDE RANGE OTMS', 'Either leg outside 0.18 delta range',
 '{"type": "delta_range", "range": "outside", "moneyness": "OTM", "delta_threshold": 0.18}',
 'GO TO BUY SIDE IMMEDIATELY',
 ARRAY['buy side immediately']),

('BLANK SIDE', 'Missing or null side values requiring manual review',
 '{"type": "validation", "check": "blank_side"}',
 'FOLLOW BUY SIDE',
 ARRAY['follow buy side']),

('WITHIN RANGE ITMS', 'ITM strikes on both sides within 0.18 delta range',
 '{"type": "delta_range", "range": "within", "moneyness": "ITM", "delta_threshold": 0.18}',
 'TAG BUY STRIKE',
 ARRAY['tag buy strike']),

('STRADDLE', 'Simultaneous buy call and buy put positions',
 '{"type": "position_structure", "structure": "straddle"}',
 'RUN CHEAPER SIDE FIRST, THEN OTHER',
 ARRAY['cheaper side', 'then other']),

('NEGATIVE ITM', 'Sell side aggregate value exceeds buy side value',
 '{"type": "value_comparison", "comparison": "sell_exceeds_buy"}',
 'DROP TO ITM STRIKE',
 ARRAY['drop', 'itm strike']),

('DEBIT AND SELL', 'Debit spread combined with opposite sell leg',
 '{"type": "position_structure", "structure": "debit_and_sell"}',
 'DEBIT WORKED OUT',
 ARRAY['debit worked']),

('UNCLASSIFIED', 'Fallback for trades not matching any pattern',
 '{"type": "fallback"}',
 'MANUAL REVIEW REQUIRED',
 ARRAY['manual review', 'unclassified']);
