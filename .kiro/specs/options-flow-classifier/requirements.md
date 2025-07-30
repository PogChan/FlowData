# Requirements Document

## Introduction

This feature enhances the existing options flow classification system to provide systematic multi-leg trade classification, earnings detection integration, outcome tracking, and predictive modeling capabilities. The system will analyze uploaded Excel data containing options flow information, classify trades according to predefined rules using real-time options chain data from Polygon API, track trade outcomes, and provide predictive insights for trading decisions.

## Requirements

### Requirement 1

**User Story:** As a trader, I want the system to automatically classify multi-leg options trades based on strike relationships and market conditions, so that I can quickly understand the trade structure and expected behavior.

#### Acceptance Criteria

1. WHEN an options flow is uploaded THEN the system SHALL determine if strikes are ATM, ITM, or OTM using current spot price
2. WHEN both legs have the same ATM strike THEN the system SHALL classify as "ATM SAME STRIKE"
3. WHEN both legs have the same ITM strike THEN the system SHALL classify as "ITM SAME STRIKE"
4. WHEN both legs have the same OTM strike THEN the system SHALL classify as "OTM SAME STRIKE"
5. WHEN both legs strikes are within 0.18 delta range of buy side direction THEN the system SHALL classify as "WITHIN RANGE OTMS"
6. WHEN either legs strikes are outside 0.18 delta range THEN the system SHALL classify as "OUTSIDE RANGE OTMS"
7. WHEN side values are blank or none THEN the system SHALL classify as "BLANK SIDE"
8. WHEN ITM strikes exist on both sides within 0.18 delta range THEN the system SHALL classify as "WITHIN RANGE ITMS"
9. WHEN there is both a buy call and buy put THEN the system SHALL classify as "STRADDLE"
10. WHEN sell side aggregate value exceeds buy side THEN the system SHALL classify as "NEGATIVE ITM"
11. WHEN there is a debit spread with opposite sell leg THEN the system SHALL classify as "DEBIT AND SELL"

### Requirement 2

**User Story:** As a trader, I want the system to integrate with Polygon API to fetch real-time options chain data including Greeks, so that I can make accurate ATM/ITM/OTM determinations and delta calculations.

#### Acceptance Criteria

1. WHEN classifying a trade THEN the system SHALL fetch current options chain data from Polygon API or streamlit cache
2. WHEN options data is retrieved THEN the system SHALL extract delta values for strike price analysis
3. WHEN determining ATM strikes THEN the system SHALL use current spot price from yfinance and available strike prices - ATM would be the strike that is closest to the close stock spot price
4. WHEN calculating delta ranges THEN the system SHALL use 0.18 delta as the threshold for range determination
5. IF Polygon API is unavailable THEN the system SHALL use cached data with timestamp warning
6. WHEN API rate limits are reached THEN the system SHALL implement exponential backoff retry logic

### Requirement 3

**User Story:** As a trader, I want the system to validate and enhance earnings detection beyond the Excel ER flag, so that I can have more accurate earnings-related trade classification.

#### Acceptance Criteria

1. WHEN processing a trade with ER flag "T" THEN the system SHALL classify as earnings trade
2. WHEN ER flag is "F" THEN the system SHALL not classify as earnings trade and use normal trade structure classification
3. WHEN earnings classification is applied THEN the system SHALL add earnings-specific expected outcomes

### Requirement 4

**User Story:** As a trader, I want to track actual trade outcomes against expected hypotheses, so that I can measure the accuracy of classification rules and improve predictions.

#### Acceptance Criteria

1. WHEN a trade is classified THEN the system SHALL store the expected hypothesis
2. WHEN trade outcomes are recorded THEN the system SHALL accept values:'FOREVER DISCOUNTED', 'DISCOUNT THEN PUMP',           'FOREVER PUMPED', 'PUMP THEN DISCOUNT'
3. WHEN comparing outcomes THEN the system SHALL calculate accuracy metrics for each classification type
4. WHEN displaying results THEN the system SHALL show expected vs actual outcome comparison
5. WHEN generating reports THEN the system SHALL provide classification accuracy statistics

### Requirement 5

**User Story:** As a trader, I want to query predictive insights about trade outcomes, so that I can make informed decisions about similar trade setups.

#### Acceptance Criteria

1. WHEN querying an archetype of trades (for example "NEGATIVE ITM") trades THEN the system SHALL provide probability of expected outcome
2. WHEN trade value is under $100k THEN the system SHALL show how probability changes for low-value trades
3. WHEN querying earnings trades THEN the system SHALL show success rate statistics
4. WHEN asking about classification patterns THEN the system SHALL provide historical performance data
5. WHEN generating predictions THEN the system SHALL include confidence intervals and sample sizes

### Requirement 6

**User Story:** As a trader, I want to add and modify classification rules dynamically, so that I can adapt the system to new trading patterns and strategies.

#### Acceptance Criteria

1. WHEN adding new rules THEN the system SHALL accept rule name, classification logic, and expected hypothesis
2. WHEN modifying existing rules THEN the system SHALL preserve historical classifications with version tracking
3. WHEN rules are updated THEN the system SHALL re-classify existing trades with new rules
4. WHEN displaying rules THEN the system SHALL show rule effectiveness metrics
5. IF rule conflicts exist THEN the system SHALL provide conflict resolution interface

### Requirement 7

**User Story:** As a trader, I want the system to handle Excel file uploads with proper data validation, so that I can reliably process options flow data.

#### Acceptance Criteria

1. WHEN Excel file is uploaded THEN the system SHALL validate required columns exist
2. WHEN processing rows THEN the system SHALL handle missing or invalid data gracefully
3. WHEN data validation fails THEN the system SHALL provide specific error messages
4. WHEN upload is successful THEN the system SHALL show processing progress and results summary
5. IF duplicate trades exist THEN the system SHALL provide deduplication options
