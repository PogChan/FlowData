# 🚀 Complete Workflow Guide

## Your 6-Step Integrated Workflow

This guide explains your complete daily options flow analysis workflow, from CSV upload to predictive insights.

### 📋 Workflow Overview

```
1. Upload Multi-Leg Flows → 2. Screen & Filter → 3. Classify Trades →
4. Analyze Volatility → 5. Store in Database → 6. Generate Predictions
```

---

## 🔄 Step-by-Step Process

### **Step 1: Upload Multi-Leg Flows for the Day**
- **Page**: 🚀 Integrated Flow Processor
- **Input**: CSV file with daily options flow data
- **Requirements**: Multi-leg trades (multiple rows per timestamp)
- **Key Columns**: Symbol, Buy/Sell, CallPut, Strike, Premium, Volume, CreatedDateTime

### **Step 2: Multi-Leg Screening (is_multi_leg = True Logic)**
- **Process**: Filters using your exact `flow_screener.py` logic
- **Filters Applied**:
  - Groups trades by Symbol + CreatedDateTime
  - Creates trade signatures for identification
  - Removes straddles/strangles (40-60% premium balance rule)
  - Validates multi-leg structure (BUY+SELL, CALL+PUT)
  - Filters out WHITE color trades
  - Ensures Volume > OI for all legs
  - Applies 70% directional conviction threshold
  - Minimum $100k premium requirement

### **Step 3: Trade Classification**
- **Process**: Classifies filtered trades into categories
- **Classifications**: 11 types (ATM SAME STRIKE, ITM SAME STRIKE, etc.)
- **Output**: Classification + Expected Outcome + Confidence Score

### **Step 4: Volatility Analysis (HV vs IV)**
- **Yang-Zhang HV Calculation**: More accurate historical volatility
- **ATM IV Extraction**: From next monthly expiration
- **Flags**: EXPENSIVE (IV > HV) or CHEAP (IV < HV)
- **Purpose**: Identify optimal entry points

### **Step 5: Store Results in Supabase**
- **Database**: Enhanced schema with all analysis fields
- **Storage**: Complete trade data + analysis results
- **Indexing**: Optimized for fast queries and analysis

### **Step 6: Predictive Analysis**
- **Stock Movement Analysis**: 1d, 3d, 7d, 30d price movements
- **Outcome Prediction**: Based on direction + movement + volatility
- **Categories**: FOREVER DISCOUNTED, DISCOUNT THEN PUMP, FOREVER PUMPED, PUMP THEN DISCOUNT, MANUAL REVIEW
- **Insights**: Recommendations and pattern analysis

---

## 📊 Data Flow Architecture

```
CSV Upload
    ↓
Data Preparation & Validation
    ↓
Multi-Leg Candidate Identification
    ↓
Signature Creation & Grouping
    ↓
Multi-Leg Filtering (Your Logic)
    ↓
Trade Classification
    ↓
Volatility Analysis (HV vs IV)
    ↓
Stock Movement Analysis
    ↓
Outcome Prediction
    ↓
Database Storage
    ↓
Insights & Recommendations
```

---

## 🎯 Key Features Implemented

### **Multi-Leg Detection Logic**
- **Exact Implementation**: Your `flow_screener.py` logic
- **Signature-Based Grouping**: Identifies related trades
- **Sophisticated Filtering**: Removes noise, keeps conviction trades

### **Volatility Analysis**
- **Yang-Zhang Method**: Superior HV calculation
- **Real-Time IV**: From Polygon API options chain
- **Premium Identification**: Expensive vs cheap contracts

### **Predictive Modeling**
- **Multi-Factor Analysis**: Direction + Movement + Volatility
- **Historical Learning**: Improves over time with outcome data
- **Confidence Scoring**: Reliability indicators

### **Database Integration**
- **Enhanced Schema**: Supports all analysis fields
- **Optimized Storage**: Fast queries and retrieval
- **Outcome Tracking**: Manual review and accuracy measurement

---

## 🖥️ User Interface

### **Main Workflow Page: 🚀 Integrated Flow Processor**
- **Single Upload**: Complete workflow in one place
- **Real-Time Progress**: Step-by-step processing feedback
- **Rich Visualizations**: Charts and breakdowns
- **Export Options**: CSV downloads and reports

### **Additional Pages**:
- **🔍 Multi-Leg Flow Screener**: Standalone screening
- **📊 Volatility Analysis**: HV vs IV analysis
- **🎯 Outcome Tracking**: Record actual results
- **🔮 Predictive Insights**: Query patterns and predictions

---

## 📈 Expected Outcomes

### **Trade Categories**
1. **FOREVER DISCOUNTED**: Bearish trades with continued downward movement
2. **DISCOUNT THEN PUMP**: Initial decline followed by recovery
3. **FOREVER PUMPED**: Bullish trades with continued upward movement
4. **PUMP THEN DISCOUNT**: Initial rise followed by decline
5. **MANUAL REVIEW**: Complex patterns requiring human analysis

### **Prediction Logic**
- **Bullish + Up Movement + Cheap IV** → FOREVER PUMPED
- **Bearish + Down Movement + Cheap IV** → FOREVER DISCOUNTED
- **Contrarian Patterns** → DISCOUNT THEN PUMP / PUMP THEN DISCOUNT
- **Conflicting Signals** → MANUAL REVIEW

---

## 🔧 Configuration & Setup

### **Required API Keys**
```toml
[supabase]
url = "your_supabase_url"
key = "your_supabase_key"

[polygon]
api_key = "your_polygon_api_key"
```

### **Database Setup**
```bash
python database/migrate.py
python initialize_rules.py
```

### **Sample Data Generation**
```bash
python generate_sample_data.py
```

---

## 📋 CSV Format Requirements

### **Required Columns**
- `Symbol`: Stock ticker
- `Buy/Sell`: BUY or SELL (or use Side: A/AA=BUY)
- `CallPut`: CALL or PUT
- `Strike`: Strike price
- `Spot`: Current stock price
- `Premium`: Premium amount
- `Volume`: Trade volume
- `OI`: Open interest
- `CreatedDateTime`: Timestamp (or separate CreatedDate/CreatedTime)

### **Optional Columns**
- `Color`, `Side`, `ER`, `ImpliedVolatility`, `Dte`
- `MktCap`, `Sector`, `StockEtf`, `Uoa`, `Weekly`, `Type`

---

## 🎯 Usage Tips

### **For Best Results**
1. **Upload Complete Data**: Include all multi-leg components
2. **Consistent Timestamps**: Ensure related trades have same CreatedDateTime
3. **Clean Data**: Remove incomplete or invalid entries
4. **Regular Updates**: Process daily for trend analysis

### **Interpreting Results**
- **High Confidence (>70%)**: Strong signal alignment
- **Low Confidence (<40%)**: Conflicting signals, manual review needed
- **Expensive Options**: Consider selling strategies
- **Cheap Options**: Consider buying strategies

---

## 🔄 Continuous Improvement

### **Manual Outcome Recording**
- Record actual trade results in 🎯 Outcome Tracking
- System learns from historical performance
- Improves prediction accuracy over time

### **Pattern Recognition**
- System identifies successful patterns
- Adjusts predictions based on historical accuracy
- Provides insights for strategy refinement

---

## 🆘 Troubleshooting

### **Common Issues**
1. **No Multi-Leg Trades Found**: Check timestamp grouping and data format
2. **Classification Errors**: Verify required columns exist
3. **API Failures**: Check Polygon API key and rate limits
4. **Database Errors**: Verify Supabase connection and permissions

### **Data Quality Checks**
- Ensure Volume > 0 and Premium > 0
- Verify Strike and Spot prices are reasonable
- Check that multi-leg trades have multiple components
- Confirm timestamps are properly formatted

---

**🎉 Your complete workflow is now ready for daily options flow analysis!**
