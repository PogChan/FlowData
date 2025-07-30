# Database Schema and Migrations

This directory contains the enhanced database schema and migration scripts for the Options Flow Classifier system.

## Overview

The enhanced database schema supports:
- Multi-leg options trade classification
- Dynamic rule management
- Options chain data caching
- Outcome tracking and analysis
- Predictive modeling capabilities

## Database Tables

### 1. Enhanced options_flow Table
**File**: `migrations/001_create_enhanced_options_flow.sql`

Extended from the original table with new fields for classification and outcome tracking:

**New Fields Added**:
- `classification` - Trade classification result (ATM SAME STRIKE, ITM SAME STRIKE, etc.)
- `expected_hypothesis` - Expected outcome hypothesis based on classification
- `actual_outcome` - Actual trade outcome (Forever Discounted, Discount then pump, etc.)
- `trade_value` - Total value of the trade in dollars
- `confidence_score` - Classification confidence score (0.0 to 1.0)
- `created_at` / `updated_at` - Timestamp tracking

**Requirements Addressed**: 6.1, 6.2, 4.1, 4.2

### 2. classification_rules Table
**File**: `migrations/002_create_classification_rules.sql`

Stores dynamic classification rules for trade analysis:

**Key Fields**:
- `rule_id` - Unique identifier for each rule
- `name` - Rule name (ATM SAME STRIKE, WITHIN RANGE OTMS, etc.)
- `classification_logic` - JSON configuration defining rule evaluation logic
- `expected_hypothesis` - Expected outcome for trades matching this rule
- `result_keywords` - Array of keywords for search and analysis
- `success_rate` - Historical success rate of the rule

**Pre-loaded Rules**:
- ATM SAME STRIKE
- ITM SAME STRIKE
- OTM SAME STRIKE
- WITHIN RANGE OTMS
- OUTSIDE RANGE OTMS
- BLANK SIDE
- WITHIN RANGE ITMS
- STRADDLE
- NEGATIVE ITM
- DEBIT AND SELL
- UNCLASSIFIED (fallback)

**Requirements Addressed**: 6.1, 6.2, 6.3, 6.4, 6.5

### 3. options_chain_cache Table
**File**: `migrations/003_create_options_chain_cache.sql`

Caches options chain data from Polygon API to reduce API calls:

**Key Fields**:
- `symbol`, `expiration_date`, `strike`, `contract_type` - Option identifier
- `delta`, `gamma`, `theta`, `vega` - Greeks for analysis
- `implied_volatility` - IV for volatility analysis
- `cached_at` / `expires_at` - Cache management timestamps

**Features**:
- Automatic cache expiration (1 hour default)
- Helper functions for cache cleanup and data retrieval
- Optimized indexes for fast lookups

**Requirements Addressed**: 2.1, 2.2, 2.5, 2.6

## Migration Execution

### Option 1: Manual Execution (Recommended)

1. **Run the migration tool to see instructions**:
   ```bash
   python database/migrate.py
   ```

2. **Execute in Supabase SQL Editor**:
   - Go to your Supabase project dashboard
   - Navigate to SQL Editor
   - Copy and paste each migration file content in order:
     1. `001_create_enhanced_options_flow.sql`
     2. `002_create_classification_rules.sql`
     3. `003_create_options_chain_cache.sql`
   - Execute each migration

### Option 2: Check Migration Status

```bash
python database/migrate.py --status
```

This will show:
- Available migration files
- Schema validation results (if connected to Supabase)
- Migration descriptions

## Schema Features

### Indexes
All tables include optimized indexes for:
- Primary key lookups
- Common query patterns
- Date range queries
- Classification filtering

### Triggers
- Automatic `updated_at` timestamp updates
- Data consistency enforcement

### Functions
- `clean_expired_options_cache()` - Removes expired cache entries
- `get_fresh_options_data()` - Retrieves non-expired cached data
- `update_updated_at_column()` - Timestamp trigger function

### Constraints
- Data type validation
- Enum constraints for contract types
- Unique constraints for cache entries

## Data Models

The corresponding Python data models are defined in `models/data_models.py`:

- `OptionsFlow` - Enhanced options flow with classification fields
- `OptionsChainData` - Options chain data from Polygon API
- `ClassificationRule` - Dynamic classification rule definition

## Performance Considerations

### Indexing Strategy
- Composite indexes for multi-column queries
- Separate indexes for filtering and sorting
- Delta-based indexes for range queries

### Caching Strategy
- 1-hour default cache expiration for options data
- Automatic cleanup of expired entries
- Unique constraints prevent duplicate cache entries

### Query Optimization
- Prepared statements for common queries
- Efficient JOIN patterns for multi-table queries
- Pagination support for large result sets

## Security

### Access Control
- Row-level security policies (to be implemented)
- Encrypted connections required
- API key management through Streamlit secrets

### Data Validation
- Input sanitization at application layer
- Database constraints for data integrity
- Audit logging for sensitive operations

## Troubleshooting

### Common Issues

1. **Migration fails with permission error**:
   - Ensure your Supabase user has CREATE TABLE permissions
   - Check that you're using the correct API key

2. **Table already exists error**:
   - The migrations include DROP TABLE IF EXISTS for development
   - In production, comment out the DROP statements

3. **Function creation fails**:
   - Ensure your Supabase plan supports custom functions
   - Some functions may require elevated privileges

### Validation

After running migrations, validate the schema:

```python
from database.migrate import DatabaseMigrator, init_supabase_client

supabase = init_supabase_client()
migrator = DatabaseMigrator(supabase)
migrator.validate_schema()
```

## Next Steps

After completing the database migrations:

1. **Update application code** to use the new schema
2. **Implement data migration** from old to new format (if needed)
3. **Test classification rules** with sample data
4. **Configure Polygon API integration** for options chain caching
5. **Set up monitoring** for cache hit rates and performance

## Requirements Mapping

| Requirement | Tables | Description |
|-------------|--------|-------------|
| 6.1, 6.2 | options_flow, classification_rules | Dynamic rule management |
| 4.1, 4.2 | options_flow | Outcome tracking |
| 2.1, 2.2 | options_chain_cache | API data caching |
| 2.5, 2.6 | options_chain_cache | Rate limiting and retry logic |
