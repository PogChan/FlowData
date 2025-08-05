"""
Enhanced Options Flow Classifier - Streamlit Dashboard (Clean Version)
Simplified interface focusing on your core workflow.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import logging

# Import our services
from supabase import create_client
from utils.config import config
from services.database_service import SupabaseService
from services.polygon_api_client import PolygonAPIClient
from services.outcome_tracker import OutcomeTracker
from services.enhanced_predictive_model import EnhancedPredictiveModel
from services.volatility_calculator import VolatilityCalculator
from services.integrated_flow_processor import IntegratedFlowProcessor
from models.data_models import OptionsFlow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Options Flow Classifier",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_services():
    """Initialize all services with caching."""
    try:
        # Initialize Supabase client
        supabase_client = create_client(config.database.url, config.database.key)

        # Initialize services
        db_service = SupabaseService(supabase_client)
        polygon_client = PolygonAPIClient(db_service)
        outcome_tracker = OutcomeTracker(supabase_client)
        volatility_calculator = VolatilityCalculator(polygon_client)
        predictive_model = EnhancedPredictiveModel(outcome_tracker)
        integrated_processor = IntegratedFlowProcessor(
            polygon_client, volatility_calculator, db_service, predictive_model, outcome_tracker
        )

        return {
            'db_service': db_service,
            'polygon_client': polygon_client,
            'outcome_tracker': outcome_tracker,
            'predictive_model': predictive_model,
            'volatility_calculator': volatility_calculator,
            'integrated_processor': integrated_processor
        }
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return None


def main():
    """Main application entry point."""
    st.markdown('<h1 class="main-header">📊 Options Flow Classifier</h1>', unsafe_allow_html=True)

    # Initialize services
    services = initialize_services()
    if not services:
        st.error("Failed to initialize application services. Please check your configuration.")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        [
            "📈 Dashboard",
            "🚀 Integrated Flow Processor",
            "📊 Volatility Analysis",
            "🎯 Outcome Tracking",
            "🔮 Predictive Insights"
        ]
    )

    # Route to appropriate page
    if page == "📈 Dashboard":
        render_dashboard(services)
    elif page == "🚀 Integrated Flow Processor":
        render_integrated_processor(services)
    elif page == "📊 Volatility Analysis":
        render_volatility_analysis(services)
    elif page == "🎯 Outcome Tracking":
        render_outcome_tracking(services)
    elif page == "🔮 Predictive Insights":
        render_predictive_insights(services)


def render_dashboard(services):
    """Render the main dashboard with key metrics and charts."""
    st.header("📈 Dashboard Overview")

    try:
        # Get recent data
        recent_trades = services['db_service'].get_options_flows({
            'date_from': (datetime.now() - timedelta(days=30)).isoformat()
        })

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Trades (30d)", len(recent_trades))

        with col2:
            classified_trades = [t for t in recent_trades if t.classification]
            classification_rate = len(classified_trades) / len(recent_trades) * 100 if recent_trades else 0
            st.metric("Classification Rate", f"{classification_rate:.1f}%")

        with col3:
            trades_with_outcomes = [t for t in recent_trades if t.actual_outcome]
            outcome_rate = len(trades_with_outcomes) / len(recent_trades) * 100 if recent_trades else 0
            st.metric("Outcome Tracking", f"{outcome_rate:.1f}%")

        with col4:
            if trades_with_outcomes:
                correct_predictions = sum(1 for t in trades_with_outcomes
                                        if services['outcome_tracker']._outcomes_match(t.expected_outcome, t.actual_outcome))
                accuracy = correct_predictions / len(trades_with_outcomes) * 100
                st.metric("Overall Accuracy", f"{accuracy:.1f}%")
            else:
                st.metric("Overall Accuracy", "N/A")

        # Charts
        if recent_trades:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Classification Distribution")
                class_counts = {}
                for trade in recent_trades:
                    if trade.classification:
                        class_counts[trade.classification] = class_counts.get(trade.classification, 0) + 1

                if class_counts:
                    fig = px.pie(
                        values=list(class_counts.values()),
                        names=list(class_counts.keys()),
                        title="Trade Classifications (Last 30 Days)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Daily Trade Volume")
                df = pd.DataFrame([{
                    'date': t.created_datetime.date() if t.created_datetime else datetime.now().date(),
                    'count': 1
                } for t in recent_trades])

                if not df.empty:
                    daily_counts = df.groupby('date')['count'].sum().reset_index()
                    fig = px.line(
                        daily_counts,
                        x='date',
                        y='count',
                        title="Daily Trade Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Recent activity
        st.subheader("Recent Activity")
        if recent_trades:
            recent_df = pd.DataFrame([{
                'Symbol': t.symbol,
                'Classification': t.classification or 'Unclassified',
                'Expected Outcome': t.expected_outcome or 'N/A',
                'Actual Outcome': t.actual_outcome or 'Pending',
                'Trade Value': f"${t.trade_value:,.0f}" if t.trade_value else 'N/A',
                'Date': t.created_datetime.strftime('%Y-%m-%d %H:%M') if t.created_datetime else 'N/A'
            } for t in recent_trades[-10:]])  # Last 10 trades

            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No recent trades found. Upload some data to get started!")

    except Exception as e:
        st.error(f"Error loading dashboard: {e}")


def render_integrated_processor(services):
    """Render the complete integrated flow processor - your main workflow."""
    st.header("🚀 Integrated Flow Processor")

    st.markdown("""
    **Complete Workflow**: Upload → Screen → Classify → Analyze → Store → Predict

    This is your main workflow that processes daily multi-leg flows through the entire pipeline:
    1. **Upload multi-leg flows** for the day
    2. **Screen using multi-leg logic** (is_multi_leg = True path)
    3. **Classify trades** into respective classes
    4. **Analyze volatility** (HV vs IV) and identify expensive/cheap contracts
    5. **Store results** in Supabase database
    6. **Generate predictions** and insights for analysis
    """)

    # File upload
    st.subheader("📁 Upload Daily Flow Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with daily options flow data",
        type=['csv'],
        help="Upload your daily multi-leg options flow data"
    )

    if uploaded_file is not None:
        try:
            # Read and preview the data
            flows_df = pd.read_csv(uploaded_file)

            st.subheader("📊 Data Preview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Flows", len(flows_df))
            with col2:
                unique_symbols = flows_df['Symbol'].nunique() if 'Symbol' in flows_df.columns else 0
                st.metric("Unique Symbols", unique_symbols)
            with col3:
                if 'CreatedDateTime' in flows_df.columns:
                    date_range = f"{flows_df['CreatedDateTime'].min()} to {flows_df['CreatedDateTime'].max()}"
                else:
                    date_range = "Unknown"
                st.write(f"**Date Range:** {date_range}")

            # Show sample data
            st.dataframe(flows_df.head(10), use_container_width=True)

            # Process button
            if st.button("🚀 Process Complete Workflow", type="primary", use_container_width=True):

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(progress: int, message: str):
                    progress_bar.progress(progress)
                    status_text.text(f"⏳ {message}")

                # Process through integrated workflow
                with st.spinner("Processing complete workflow..."):
                    results = services['integrated_processor'].process_daily_flows(
                        flows_df, progress_callback=update_progress
                    )

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Display results
                if results.get('success'):
                    st.success("🎉 Complete workflow processed successfully!")

                    # Summary metrics
                    st.subheader("📊 Processing Summary")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Original Flows", results['original_flows'])
                    with col2:
                        st.metric("Multi-Leg Trades", results['multi_leg_trades'])
                    with col3:
                        st.metric("Classified Trades", results['classified_trades'])
                    with col4:
                        st.metric("Stored in DB", results['stored_trades'])

                    # Display processed trades
                    trades_data = results.get('trades_data')
                    if trades_data is not None and not trades_data.empty:
                        st.subheader("📋 Processed Trades")

                        # Format the display
                        display_columns = [
                            'Symbol', 'Buy/Sell', 'CallPut', 'Strike', 'Premium',
                            'Volume', 'Direction', 'classification', 'expected_outcome',
                            'volatility_flag', 'moneiness', 'movement_direction'
                        ]

                        # Filter to existing columns
                        available_columns = [col for col in display_columns if col in trades_data.columns]
                        display_df = trades_data[available_columns].copy()

                        # Format premium
                        if 'Premium' in display_df.columns:
                            display_df['Premium'] = display_df['Premium'].apply(
                                lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
                            )

                        st.dataframe(display_df, use_container_width=True)

                        # Export option
                        if st.button("📥 Download Results"):
                            csv = trades_data.to_csv(index=False)
                            st.download_button(
                                label="📄 Download CSV",
                                data=csv,
                                file_name=f"processed_flows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                else:
                    error_msg = results.get('error', 'Unknown error occurred')
                    st.error(f"❌ Processing failed: {error_msg}")

        except Exception as e:
            st.error(f"Error reading file: {e}")

    else:
        st.info("👆 Upload a CSV file to start the complete workflow")


def render_volatility_analysis(services):
    """Render volatility analysis dashboard."""
    st.header("📊 Volatility Analysis Dashboard")

    st.markdown("""
    Compare **Historical Volatility (Yang-Zhang method)** vs **Implied Volatility** to identify
    expensive or cheap options contracts.
    """)

    # Symbol input
    symbol_input = st.text_input("Enter Symbol:", value="AAPL", help="Enter a stock symbol to analyze")

    if symbol_input and st.button("🔍 Analyze Volatility", type="primary"):
        with st.spinner(f"Analyzing volatility for {symbol_input}..."):
            analysis = services['volatility_calculator'].analyze_volatility_premium(symbol_input.upper())

            if analysis['flag'] != 'ERROR':
                # Display results
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    hv_value = analysis.get('hv', 0)
                    st.metric("Historical Volatility", f"{hv_value:.2%}" if hv_value else "N/A")

                with col2:
                    iv_value = analysis.get('iv', 0)
                    st.metric("Implied Volatility", f"{iv_value:.2%}" if iv_value else "N/A")

                with col3:
                    premium = analysis.get('volatility_premium', 0)
                    st.metric("Volatility Premium", f"{premium:.2%}" if premium else "N/A")

                with col4:
                    flag = analysis.get('flag', 'UNKNOWN')
                    color = "green" if flag == "CHEAP" else "red" if flag == "EXPENSIVE" else "gray"
                    st.markdown(f"<h3 style='color: {color};'>{flag}</h3>", unsafe_allow_html=True)

                # Message
                st.info(analysis.get('message', 'Analysis completed'))

            else:
                st.error(f"Analysis failed: {analysis.get('message', 'Unknown error')}")


def render_outcome_tracking(services):
    """Render the outcome tracking interface."""
    st.header("🎯 Outcome Tracking")

    # Get trades without outcomes
    pending_trades = services['outcome_tracker'].get_trades_without_outcomes(50)

    if pending_trades:
        st.subheader("Record Trade Outcomes")

        # Select trade to update
        trade_options = {
            f"{trade['symbol']} - {trade['classification']} ({trade['created_datetime'][:10]})": trade['id']
            for trade in pending_trades
        }

        selected_trade_key = st.selectbox("Select trade to update:", list(trade_options.keys()))

        if selected_trade_key:
            trade_id = trade_options[selected_trade_key]
            selected_trade = next(t for t in pending_trades if t['id'] == trade_id)

            # Display trade details
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Trade Details:**")
                st.write(f"Symbol: {selected_trade['symbol']}")
                st.write(f"Classification: {selected_trade['classification']}")
                st.write(f"Expected Outcome: {selected_trade['expected_outcome']}")
                st.write(f"Trade Value: ${selected_trade['trade_value']:,.0f}")

            with col2:
                st.write("**Record Outcome:**")
                outcome = st.selectbox(
                    "Actual Outcome:",
                    services['outcome_tracker'].valid_outcomes
                )

                if st.button("Record Outcome", type="primary"):
                    if services['outcome_tracker'].record_outcome(trade_id, outcome):
                        st.success("Outcome recorded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to record outcome.")

    else:
        st.info("No trades pending outcome recording.")


def render_predictive_insights(services):
    """Render the predictive insights interface."""
    st.header("🔮 Predictive Insights")

    st.markdown("""
    Query trade patterns and get predictions based on your processed data.
    """)

    # Get recent trades for analysis
    recent_trades = services['db_service'].get_options_flows({
        'date_from': (datetime.now() - timedelta(days=7)).isoformat()
    })

    if recent_trades:
        st.subheader("Recent Trade Analysis")

        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame([{
            'Symbol': t.symbol,
            'Direction': getattr(t, 'direction', 'UNKNOWN'),
            'Classification': t.classification,
            'Volatility Flag': getattr(t, 'volatility_flag', 'UNKNOWN'),
            'Movement Direction': getattr(t, 'movement_direction', 'UNKNOWN'),
            'Expected Outcome': t.expected_outcome,
            'Actual Outcome': t.actual_outcome or 'Pending'
        } for t in recent_trades])

        st.dataframe(trades_df, use_container_width=True)

        # Simple analysis
        if len(trades_df) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Direction Distribution")
                if 'Direction' in trades_df.columns:
                    direction_counts = trades_df['Direction'].value_counts()
                    fig = px.pie(values=direction_counts.values, names=direction_counts.index)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Volatility Analysis")
                if 'Volatility Flag' in trades_df.columns:
                    vol_counts = trades_df['Volatility Flag'].value_counts()
                    fig = px.bar(x=vol_counts.index, y=vol_counts.values)
                    st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No recent trades found. Process some data first!")


if __name__ == "__main__":
    main()
