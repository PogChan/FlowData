import streamlit as st
import pandas as pd
import json
from supabase import create_client, Client

# ----------------------------------
# Connect to Supabase via Streamlit secrets
# ----------------------------------
@st.cache_resource
def init_supabase_client():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase: Client = init_supabase_client()

# ----------------------------------
# Default Rules Configuration
# ----------------------------------
default_rules = {
    "ATM SAME STRIKE": {
        "expected_hypothesis": "CREATED WALL ON BUY SIDE TRADE",
        "result_keywords": ["wall", "buy side"]
    },
    "ITM BUY SAME STRIKE": {
        "expected_hypothesis": "GO TO BUY STRIKE, PUMP BUY SIDE",
        "result_keywords": ["pump", "buy strike"]
    },
    "OTM BUY SAME STRIKE": {
        "expected_hypothesis": "GO TO BUY STRIKE",
        "result_keywords": ["buy strike"]
    },
    "WITHIN RANGE OTMS": {
        "expected_hypothesis": "DISCOUNT SELL SIDE, RUN BUY SIDE",
        "result_keywords": ["discount", "sell side", "run buy side"]
    },
    "OUTSIDE RANGE OTMS": {
        "expected_hypothesis": "GO TO BUY SIDE IMMEDIATELY",
        "result_keywords": ["buy side immediately"]
    },
    "BLANK SIDE": {
        "expected_hypothesis": "FOLLOW BUY SIDE",
        "result_keywords": ["follow buy side"]
    },
    "WITHIN RANGE ITMS": {
        "expected_hypothesis": "TAG BUY STRIKE",
        "result_keywords": ["tag buy strike"]
    },
    "STRADDLE": {
        "expected_hypothesis": "RUN CHEAPER SIDE FIRST, THEN OTHER",
        "result_keywords": ["cheaper side", "then other"]
    },
    "NEGATIVE ITM": {
        "expected_hypothesis": "DROP TO ITM STRIKE",
        "result_keywords": ["drop", "itm strike"]
    },
    "DEBIT AND SELL": {
        "expected_hypothesis": "DEBIT WORKED OUT",
        "result_keywords": ["debit worked"]
    },
    "EARNINGS": {
        "expected_hypothesis": "WORKED (EARNINGS)",
        "result_keywords": ["worked", "earnings"]
    },
    "WEEKLY": {
        "expected_hypothesis": "WORKED",
        "result_keywords": ["worked"]
    }
}

# ----------------------------------
# Classification Function
# ----------------------------------
def classify_trade(row, rules):
    """
    Classify a trade row using the provided rules.
    For the special case of EARNINGS, we examine both the 'ER' and 'EarningsDate' fields.
    """
    trade_type = row.get("TYPE", "").strip()
    classification = rules.get(trade_type, {}).get("expected_hypothesis", "Unclassified")

    # Special case: EARNINGS
    if trade_type.upper() == "EARNINGS":
        er_value = str(row.get("ER", "")).strip().upper()
        earnings_date = str(row.get("EarningsDate", "")).strip()
        # If no earnings date or ER indicates false, add a note for manual check.
        if not earnings_date or er_value in ("F", "NO", "0"):
            classification += " (Check earnings data)"
        else:
            classification += " (Earnings confirmed)"

    return classification

# ----------------------------------
# Retrieve Data from Supabase
# ----------------------------------
@st.cache_data(ttl=60)
def get_flows():
    """
    Retrieve options flow data from Supabase.
    Assumes there is a table called 'options_flow'.
    """
    response = supabase.table("options_flow").select("*").execute()
    data = response.data
    if data:
        df = pd.DataFrame(data)
        return df
    else:
        return pd.DataFrame()

# ----------------------------------
# Search Functionality
# ----------------------------------
def search_flows(df, search_query):
    """
    Search the DataFrame for the query in key columns.
    """
    if search_query:
        search_query = search_query.lower()
        mask = df.apply(
            lambda row: (search_query in str(row.get("Symbol", "")).lower()) or
                        (search_query in str(row.get("CreatedDate", "")).lower()) or
                        (search_query in str(row.get("TYPE", "")).lower()),
            axis=1
        )
        return df[mask]
    else:
        return df

# ----------------------------------
# Main App
# ----------------------------------
def main():
    st.title("Options Flow Trade Classifier with Supabase")

    st.sidebar.header("Search Options Flow")
    search_query = st.sidebar.text_input("Search by Symbol, Date, or Type")

    st.sidebar.header("Rule Configuration (JSON)")
    rules_json = st.sidebar.text_area("Edit Rule Configuration (JSON)",
                                      value=json.dumps(default_rules, indent=4),
                                      height=300)
    try:
        rules = json.loads(rules_json)
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON format in rules configuration.")
        rules = default_rules

    st.header("Options Flow Data")
    df_flows = get_flows()
    if df_flows.empty:
        st.write("No data found in the 'options_flow' table.")
    else:
        # Apply search filter
        df_filtered = search_flows(df_flows, search_query)

        # Apply classification for each row
        df_filtered["Classification"] = df_filtered.apply(lambda row: classify_trade(row, rules), axis=1)

        st.dataframe(df_filtered)

        # Option to download filtered data
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download Filtered Data as CSV", data=csv, file_name="filtered_options_flow.csv", mime="text/csv")

    st.write("Use the sidebar to search and modify rules. The classification logic (including earnings rules) can be extended by editing the `classify_trade` function.")

if __name__ == "__main__":
    main()
