-- Migration: Create options_chain_cache table for API data caching
-- Requirements: 2.1, 2.2, 2.5, 2.6

-- Create options_chain_cache table
CREATE TABLE options_chain_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(10) NOT NULL,
    expiration_date DATE NOT NULL,
    strike DECIMAL(10,2) NOT NULL,
    contract_type VARCHAR(4) NOT NULL CHECK (contract_type IN ('call', 'put')),
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    implied_volatility DECIMAL(6,4),
    open_interest INTEGER,
    volume INTEGER,
    last_price DECIMAL(10,2),
    bid DECIMAL(10,2),
    ask DECIMAL(10,2),
    cached_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '1 hour'),
    UNIQUE(symbol, expiration_date, strike, contract_type)
);

-- Create indexes for performance
CREATE INDEX idx_options_chain_symbol ON options_chain_cache(symbol);
CREATE INDEX idx_options_chain_expiration ON options_chain_cache(expiration_date);
CREATE INDEX idx_options_chain_strike ON options_chain_cache(strike);
CREATE INDEX idx_options_chain_type ON options_chain_cache(contract_type);
CREATE INDEX idx_options_chain_cached_at ON options_chain_cache(cached_at);
CREATE INDEX idx_options_chain_expires_at ON options_chain_cache(expires_at);
CREATE INDEX idx_options_chain_delta ON options_chain_cache(delta);

-- Composite index for common queries
CREATE INDEX idx_options_chain_lookup ON options_chain_cache(symbol, expiration_date, contract_type);

-- Add comments for documentation
COMMENT ON TABLE options_chain_cache IS 'Cached options chain data from Polygon API to reduce API calls';
COMMENT ON COLUMN options_chain_cache.contract_type IS 'Option type: call or put';
COMMENT ON COLUMN options_chain_cache.delta IS 'Option delta (price sensitivity to underlying)';
COMMENT ON COLUMN options_chain_cache.gamma IS 'Option gamma (delta sensitivity)';
COMMENT ON COLUMN options_chain_cache.theta IS 'Option theta (time decay)';
COMMENT ON COLUMN options_chain_cache.vega IS 'Option vega (volatility sensitivity)';
COMMENT ON COLUMN options_chain_cache.cached_at IS 'When this data was cached';
COMMENT ON COLUMN options_chain_cache.expires_at IS 'When this cached data expires';

-- Create function to clean expired cache entries
CREATE OR REPLACE FUNCTION clean_expired_options_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM options_chain_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get fresh cache data
CREATE OR REPLACE FUNCTION get_fresh_options_data(
    p_symbol VARCHAR(10),
    p_expiration DATE,
    p_strike DECIMAL(10,2),
    p_contract_type VARCHAR(4)
)
RETURNS TABLE (
    delta DECIMAL(8,6),
    gamma DECIMAL(8,6),
    theta DECIMAL(8,6),
    vega DECIMAL(8,6),
    implied_volatility DECIMAL(6,4),
    last_price DECIMAL(10,2),
    cached_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        oc.delta,
        oc.gamma,
        oc.theta,
        oc.vega,
        oc.implied_volatility,
        oc.last_price,
        oc.cached_at
    FROM options_chain_cache oc
    WHERE oc.symbol = p_symbol
      AND oc.expiration_date = p_expiration
      AND oc.strike = p_strike
      AND oc.contract_type = p_contract_type
      AND oc.expires_at > NOW()
    ORDER BY oc.cached_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;
