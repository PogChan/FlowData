-- Migration: Create classification_rules table for dynamic rule management
-- Requirements: 6.1, 6.2, 6.3, 6.4, 6.5

-- Create classification_rules table
CREATE TABLE classification_rules (
    rule_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    classification_logic JSONB NOT NULL,
    expected_outcome VARCHAR(50) NOT NULL CHECK (
        expected_outcome IN (
            'FOREVER DISCOUNTED',
            'DISCOUNT THEN PUMP',
            'FOREVER PUMPED',
            'PUMP THEN DISCOUNT',
            'MANUAL REVIEW'
        )
    ),
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
INSERT INTO classification_rules (name, description, classification_logic, expected_outcome, result_keywords) VALUES
('ATM SAME STRIKE', 'Both legs have identical at-the-money strikes',
 '{"type": "same_strike", "moneyness": "ATM", "delta_threshold": 0.18}',
 'FOREVER DISCOUNTED',
 ARRAY['forever', 'discount']),

('ITM SAME STRIKE', 'Both legs have identical in-the-money strikes',
 '{"type": "same_strike", "moneyness": "ITM", "delta_threshold": 0.18}',
 'DISCOUNT THEN PUMP',
 ARRAY['discount', 'then pump', 'pump']),

('OTM SAME STRIKE', 'Both legs have identical out-of-the-money strikes',
 '{"type": "same_strike", "moneyness": "OTM", "delta_threshold": 0.18}',
 'FOREVER PUMPED',
 ARRAY['forever', 'pump']),

('WITHIN RANGE OTMS', 'Both legs within 0.18 delta range of buy side direction',
 '{"type": "delta_range", "range": "within", "moneyness": "OTM", "delta_threshold": 0.18}',
 'DISCOUNT THEN PUMP',
 ARRAY['discount', 'then pump', 'pump']),

('OUTSIDE RANGE OTMS', 'Either leg outside 0.18 delta range',
 '{"type": "delta_range", "range": "outside", "moneyness": "OTM", "delta_threshold": 0.18}',
 'FOREVER PUMPED',
 ARRAY['forever', 'pump']),

('BLANK SIDE', 'Missing or null side values requiring manual review',
 '{"type": "validation", "check": "blank_side"}',
 'FOREVER PUMPED',
 ARRAY['forever', 'pump']),

('WITHIN RANGE ITMS', 'ITM strikes on both sides within 0.18 delta range',
 '{"type": "delta_range", "range": "within", "moneyness": "ITM", "delta_threshold": 0.18}',
 'FOREVER DISCOUNTED',
 ARRAY['forever', 'discount']),

('STRADDLE', 'Simultaneous buy call and buy put positions',
 '{"type": "position_structure", "structure": "straddle"}',
 'DISCOUNT THEN PUMP',
 ARRAY['discount', 'then pump', 'pump']),

('NEGATIVE ITM', 'Sell side aggregate value exceeds buy side value',
 '{"type": "value_comparison", "comparison": "sell_exceeds_buy"}',
 'FOREVER PUMPED',
 ARRAY['forever', 'pump']),

('DEBIT AND SELL', 'Debit spread combined with opposite sell leg',
 '{"type": "position_structure", "structure": "debit_and_sell"}',
 'FOREVER PUMPED',
 ARRAY['forever', 'pump']),

('UNCLASSIFIED', 'Fallback for trades not matching any pattern',
 '{"type": "fallback"}',
 'MANUAL REVIEW',
 ARRAY['manual review', 'unclassified']);
