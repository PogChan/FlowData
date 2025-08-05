# 📊 Enhanced Options Flow Classifier

A comprehensive Streamlit application for analyzing multi-leg options trades with real-time classification, outcome tracking, and predictive modeling capabilities. Built with Supabase backend and Polygon API integration.

## 🚀 Features

### Core Functionality
- **📁 Excel Upload & Processing**: Upload and validate Excel files with comprehensive error handling
- **🎯 Multi-Leg Trade Classification**: 11 sophisticated classification rules for options strategies
- **🔍 Enhanced Flow Screener**: Advanced multi-leg synthetic options trade detection
- **📊 Volatility Analysis**: Yang-Zhang Historical Volatility vs Implied Volatility comparison
- **📊 Real-Time Market Data**: Integration with Polygon API for current options chain data
- **🔮 Predictive Analytics**: ML-powered outcome predictions with confidence intervals
- **📈 Outcome Tracking**: Record and analyze actual trade results vs predictions
- **⚙️ Dynamic Rules Engine**: Add, modify, and manage classification rules on the fly

### Advanced Analytics
- **📊 Interactive Dashboard**: Comprehensive analytics with Plotly visualizations
- **🎯 Classification Accuracy**: Track performance of each classification type
- **💰 Value-Based Analysis**: Separate analytics for high/low value trades
- **📅 Earnings Trade Analysis**: Specialized handling for earnings-related trades
- **🔍 Query-Based Insights**: Natural language queries for trade pattern analysis
- **📈 Multi-Leg Flow Screening**: Sophisticated filtering for synthetic options strategies
- **📊 Volatility Premium Analysis**: HV vs IV comparison with Yang-Zhang methodology

### Technical Features
- **🏗️ Modular Architecture**: Clean separation of concerns with service-oriented design
- **💾 Intelligent Caching**: Options chain data caching to minimize API calls
- **🔄 Rate Limiting**: Built-in API rate limiting with exponential backoff
- **🛡️ Error Handling**: Comprehensive error handling and user feedback
- **📱 Responsive UI**: Mobile-friendly Streamlit interface

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- Supabase account and project
- Polygon.io API key (for real-time data)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create `.streamlit/secrets.toml`:
```toml
[supabase]
url = "your_supabase_project_url"
key = "your_supabase_anon_key"

[polygon]
api_key = "your_polygon_api_key"
```

### 3. Database Setup
Run the database migrations to create the required tables:
```bash
python database/migrate.py
```

### 4. Initialize Classification Rules
Set up the default classification rules:
```bash
python initialize_rules.py
```

### 5. Launch Application
```bash
streamlit run app.py
```

## 📋 Database Schema

### Enhanced Tables
- **`options_flow`**: Main table with classification and outcome fields
- **`classification_rules`**: Dynamic rules for trade classification
- **`options_chain_cache`**: Cached options data from Polygon API

### Key Fields
- **Classification Types**: 11 distinct trade pattern classifications
- **Outcome Tracking**: Expected vs actual outcomes with confidence scores
- **Market Data**: Real-time spot prices, deltas, and Greeks
- **Trade Metadata**: Volume, premium, DTE, earnings flags

## 🎯 Classification System

### 11 Classification Rules
1. **ATM SAME STRIKE**: Both legs at-the-money
2. **ITM SAME STRIKE**: Both legs in-the-money
3. **OTM SAME STRIKE**: Both legs out-of-the-money
4. **WITHIN RANGE OTMS**: OTM strikes within 0.18 delta range
5. **OUTSIDE RANGE OTMS**: OTM strikes outside delta range
6. **BLANK SIDE**: Missing side information
7. **WITHIN RANGE ITMS**: ITM strikes within delta range
8. **STRADDLE**: Buy call + buy put combination
9. **NEGATIVE ITM**: Sell side value exceeds buy side
10. **DEBIT AND SELL**: Debit spread with opposite sell leg
11. **UNCLASSIFIED**: Fallback for unmatched patterns

### Delta Threshold
- Uses 0.18 delta as the standard threshold for range classifications
- Configurable via application settings
- Based on real-time options chain data from Polygon API

## 📊 Usage Guide

### 1. Dashboard Overview
- View key metrics and recent activity
- Monitor classification rates and accuracy
- Track system performance over time

### 2. Upload & Classify
- Upload Excel files with options flow data
- Automatic data validation and cleaning
- Real-time classification with confidence scores
- Batch processing with progress indicators

### 3. Outcome Tracking
- Record actual trade outcomes
- Compare against expected hypotheses
- Calculate accuracy metrics by classification
- Track performance over time

### 4. Predictive Insights
- Query trade patterns with natural language
- Get probability distributions for outcomes
- Analyze earnings vs regular trade performance
- Compare high vs low value trade success rates

### 5. Rule Management
- View and manage classification rules
- Add custom rules with logic builder
- Track rule effectiveness metrics
- Enable/disable rules dynamically

### 6. Multi-Leg Flow Screener
- Upload daily CSV flow data
- Advanced multi-leg trade detection
- Synthetic options strategy identification
- Directional bias analysis with conviction filtering

### 7. Volatility Analysis
- Yang-Zhang Historical Volatility calculation
- Implied Volatility comparison from ATM options
- Expensive vs cheap options identification
- Batch analysis for multiple symbols

### 8. Analytics Dashboard
- Comprehensive performance analytics
- Classification-specific analysis
- Earnings trade analysis
- Value-based performance metrics

## 📁 Excel File Format

### Required Columns
- `symbol`: Stock ticker (e.g., AAPL)
- `buy_sell`: BUY or SELL
- `call_put`: CALL or PUT
- `strike`: Strike price
- `spot`: Current stock price
- `expiration_date`: Option expiration
- `premium`: Option premium
- `volume`: Trade volume
- `side`: Trade side indicator

### Optional Columns
- `open_interest`: Open interest
- `price`: Trade price
- `color`: Trade color coding
- `set_count`: Set count
- `implied_volatility`: IV
- `dte`: Days to expiration
- `er_flag`: Earnings flag (T/F)

### Column Mapping
The system automatically maps common alternative column names:
- `ticker` → `symbol`
- `type` → `call_put`
- `exp_date` → `expiration_date`
- `vol` → `volume`
- `iv` → `implied_volatility`

## 🔧 Configuration

### Application Settings
- **Cache TTL**: Options data cache duration (default: 5 minutes)
- **Delta Threshold**: Classification delta threshold (default: 0.18)
- **Rate Limiting**: API request delays and retry logic
- **File Size Limits**: Maximum Excel file size

### API Configuration
- **Polygon API**: Real-time options data
- **yfinance**: Spot price fallback
- **Supabase**: Database operations
- **Rate Limiting**: Automatic throttling and backoff

## 🚨 Error Handling

### Data Validation
- Missing column detection
- Data type validation
- Duplicate trade detection
- Invalid value handling

### API Resilience
- Automatic retry with exponential backoff
- Fallback to cached data
- Rate limit compliance
- Connection error recovery

### User Feedback
- Detailed error messages
- Processing progress indicators
- Validation warnings
- Success confirmations

## 📈 Performance Optimization

### Caching Strategy
- Options chain data caching
- Database query optimization
- Service-level caching
- Streamlit component caching

### Batch Processing
- Efficient bulk operations
- Progress tracking
- Memory management
- Error isolation

## 🧪 Testing

### Test Coverage
- Unit tests for all services
- Integration tests for workflows
- Mock API testing
- Data validation testing

### Test Data
- Sample Excel files
- Mock API responses
- Test database fixtures
- Performance benchmarks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the error messages in the UI
2. Review the logs for detailed error information
3. Verify your API keys and database configuration
4. Ensure your Excel file format matches requirements

## 🔄 Updates & Roadmap

### Recent Updates
- ✅ Enhanced multi-leg classification system
- ✅ Real-time Polygon API integration
- ✅ Comprehensive outcome tracking
- ✅ Predictive modeling engine
- ✅ Dynamic rules management
- ✅ Advanced analytics dashboard

### Future Enhancements
- 🔄 Machine learning model improvements
- 🔄 Additional data source integrations
- 🔄 Mobile app development
- 🔄 Advanced visualization features
- 🔄 Automated trading signals

---

**Built with ❤️ using Streamlit, Supabase, and Polygon API**
