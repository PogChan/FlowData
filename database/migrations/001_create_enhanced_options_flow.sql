-- Migration: Enhanced options_flow table with classification and outcome fields
-- Requirements: 6.1, 6.2, 4.1, 4.2

-- Drop existing table if it exists (for development purposes)
-- In production, this would be an ALTER TABLE statement #PROD
DROP TABLE IF EXISTS options_flow CASCADE;

-- Create enhanced options_flow table
CREATE TABLE options_flow (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_datetime TIMESTAMP WITH TIME ZONE,
    symbol VARCHAR(10) NOT NULL,
    buy_sell VARCHAR(4) NOT NULL,
    call_put VARCHAR(4) NOT NULL,
    strike DECIMAL(10,2) NOT NULL,
    spot DECIMAL(10,2) NOT NULL,
    expiration_date DATE NOT NULL,
    premium DECIMAL(12,2),
    volume INTEGER,
    open_interest INTEGER,
    price DECIMAL(10,2),
    side VARCHAR(1),
    color VARCHAR(20),
    set_count INTEGER,
    implied_volatility DECIMAL(6,4),
    dte INTEGER,
    er_flag BOOLEAN,
    -- Multi-leg trade identification
    trade_signature VARCHAR(500),
    trade_group_id UUID,
    -- Classification and analysis
    classification VARCHAR(50),
    expected_outcome VARCHAR(50) CHECK (
        expected_outcome IN (
            'FOREVER DISCOUNTED',
            'DISCOUNT THEN PUMP',
            'FOREVER PUMPED',
            'PUMP THEN DISCOUNT',
            'MANUAL REVIEW'
        )
    ),
    actual_outcome VARCHAR(50) CHECK (
        actual_outcome IN (
            'FOREVER DISCOUNTED',
            'DISCOUNT THEN PUMP',
            'FOREVER PUMPED',
            'PUMP THEN DISCOUNT',
            'MANUAL REVIEW'
        )
    ),
    trade_value DECIMAL(15,2),
    confidence_score DECIMAL(4,3),
    -- Volatility analysis
    historical_volatility DECIMAL(6,4),
    implied_volatility_atm DECIMAL(6,4),
    volatility_flag VARCHAR(20),
    volatility_premium DECIMAL(6,4),
    -- Additional fields from flow screener
    direction VARCHAR(10),
    moneiness VARCHAR(10),
    pc_ratio DECIMAL(6,4),
    earnings_date DATE,
    sector VARCHAR(50),
    market_cap VARCHAR(20),
    stock_etf VARCHAR(10),
    uoa VARCHAR(20),
    weekly BOOLEAN,
    type VARCHAR(20),
    -- Stock movement tracking for predictive model
    stock_movement_1d DECIMAL(6,4),
    stock_movement_3d DECIMAL(6,4),
    stock_movement_7d DECIMAL(6,4),
    stock_movement_30d DECIMAL(6,4),
    movement_direction VARCHAR(20),
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);


-- Create indexes for performance
CREATE INDEX idx_options_flow_symbol ON options_flow(symbol);
CREATE INDEX idx_options_flow_expiration ON options_flow(expiration_date);
CREATE INDEX idx_options_flow_classification ON options_flow(classification);
CREATE INDEX idx_options_flow_created_datetime ON options_flow(created_datetime);
CREATE INDEX idx_options_flow_er_flag ON options_flow(er_flag);
CREATE INDEX idx_options_flow_trade_group ON options_flow(trade_group_id);
CREATE INDEX idx_options_flow_actual_outcome ON options_flow(actual_outcome);
CREATE INDEX idx_options_flow_volatility_flag ON options_flow(volatility_flag);
CREATE INDEX idx_options_flow_direction ON options_flow(direction);

-- Create trigger for updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_options_flow_updated_at
    BEFORE UPDATE ON options_flow
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE options_flow IS 'Enhanced options flow data with classification and outcome tracking';
COMMENT ON COLUMN options_flow.classification IS 'Trade classification result (ATM SAME STRIKE, ITM SAME STRIKE, etc.)';
COMMENT ON COLUMN options_flow.expected_outcome IS 'Expected outcome hypothesis based on classification';
COMMENT ON COLUMN options_flow.actual_outcome IS 'Actual trade outcome: Forever Discounted, Discount then pump, Forever Pumped, Pump then discount';
COMMENT ON COLUMN options_flow.trade_value IS 'Total value of the trade in dollars';
COMMENT ON COLUMN options_flow.confidence_score IS 'Classification confidence score (0.0 to 1.0)';
