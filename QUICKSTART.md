# 🚀 Quick Start Guide

Get your Options Flow Classifier up and running in 5 minutes!

## ⚡ Prerequisites Check

Before starting, make sure you have:
- [ ] Python 3.8+ installed
- [ ] A Supabase account (free tier works)
- [ ] A Polygon.io API key (free tier available)

## 📋 Step-by-Step Setup

### 1. Clone & Install (2 minutes)
```bash
# Clone the repository
git clone <your-repo-url>
cd options-flow-classifier

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Secrets (1 minute)
Create `.streamlit/secrets.toml`:
```toml
[supabase]
url = "https://your-project.supabase.co"
key = "your-anon-key-here"

[polygon]
api_key = "your-polygon-api-key"
```

**Where to find these:**
- **Supabase**: Project Settings → API → URL and anon/public key
- **Polygon**: Dashboard → API Keys

### 3. Setup Database (1 minute)
```bash
# Create database tables
python database/migrate.py
```

### 4. Launch Application (30 seconds)
```bash
streamlit run app.py
```

Your app will open at `http://localhost:8501` 🎉

## 🎯 First Steps in the App

### 1. Check Dashboard
- Navigate to "📈 Dashboard" to see the overview
- Should show empty state initially

### 2. Upload Sample Data
- Go to "📁 Upload & Classify"
- Upload an Excel file with options data
- Watch the magic happen!

### 3. Explore Features
- **🔍 Multi-Leg Flow Screener**: Upload CSV files to find synthetic options trades
- **📊 Volatility Analysis**: Compare HV vs IV for options pricing
- **🎯 Outcome Tracking**: Record trade results
- **🔮 Predictive Insights**: Query trade patterns
- **⚙️ Rule Management**: Customize classification rules
- **📊 Analytics**: Deep dive into performance

## 📊 Sample Data Formats

### Excel Format (for Upload & Classify)
| symbol | buy_sell | call_put | strike | spot | expiration_date | premium | volume | side |
|--------|----------|----------|--------|------|-----------------|---------|--------|------|
| AAPL   | BUY      | CALL     | 150    | 145  | 2024-01-19      | 2.50    | 100    | A    |
| AAPL   | SELL     | PUT      | 140    | 145  | 2024-01-19      | 1.25    | 100    | B    |

### CSV Format (for Multi-Leg Flow Screener)
Generate sample data with: `python generate_sample_data.py`

Required columns: Symbol, Buy/Sell, CallPut, Strike, Spot, ExpirationDate, Premium, Volume, OI, Price, Side, Color, CreatedDateTime

## 🚨 Common Issues & Solutions

### "Failed to initialize services"
- ✅ Check your `.streamlit/secrets.toml` file exists
- ✅ Verify Supabase URL and key are correct
- ✅ Ensure Polygon API key is valid

### "Database connection failed"
- ✅ Run `python database/migrate.py` first
- ✅ Check Supabase project is active
- ✅ Verify database permissions

### "No classification rules found"
- ✅ Run `python initialize_rules.py`
- ✅ Check database tables were created

### Excel upload errors
- ✅ Ensure required columns exist
- ✅ Check date format (YYYY-MM-DD)
- ✅ Verify numeric columns have valid numbers

## 🎉 You're Ready!

Once everything is working:

1. **Upload your first Excel file** with options data
2. **Watch trades get classified** automatically
3. **Record outcomes** to track accuracy
4. **Explore analytics** to gain insights
5. **Customize rules** for your specific needs

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out the [Database Schema](database/README.md)
- Explore the [API Documentation](docs/api.md) (if available)

## 🆘 Need Help?

- Check error messages in the Streamlit interface
- Review logs in the terminal where you ran `streamlit run app.py`
- Ensure all prerequisites are met
- Verify your configuration files

---

**Happy Trading! 📈**
