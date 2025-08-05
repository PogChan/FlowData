"""
Enhanced Options Flow Classifier - Streamlit Dashboard
Comprehensive multi-leg options trade analysis system with real-time classification,
outcome tracking, and predictive modeling capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Any, Optional
import logging

# Import our services
from supabase import create_client
from utils.config import config
from services.database_service import SupabaseService
from services.polygon_api_client import PolygonAPIClient
from services.trade_classifier import TradeClassifier
from services.rules_engine import RulesEngine
from services.outcome_tracker import OutcomeTracker
from services.predictive_model import PredictiveModel
from services.excel_processor import ExcelDataProcessor
from services.volatility_calculator import VolatilityCalculator
from services.enhanced_flow_screener import EnhancedFlowScreener
from services.integrated_flow_processor import IntegratedFlowProcessor
from models.data_models import OptionsFlow, ClassificationRule

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

# Custom CSS for better styling
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
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
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
        rules_engine = RulesEngine(supabase_client)
        classifier = TradeClassifier(polygon_client, rules_engine)
        outcome_tracker = OutcomeTracker(supabase_client)
        predictive_model = PredictiveModel(outcome_tracker)
        excel_processor = ExcelDataProcessor()
        volatility_calculator = VolatilityCalculator(polygon_client)
        flow_screener = EnhancedFlowScreener(polygon_client, volatility_calculator)
        integrated_processor = IntegratedFlowProcessor(
            polygon_client, volatility_calculator, db_service, classifier, outcome_tracker
        )

        return {
            'db_service': db_service,
            'polygon_client': polygon_client,
            'classifier': classifier,
            'rules_engine': rules_engine,
            'outcome_tracker': outcome_tracker,
            'predictive_model': predictive_model,
            'excel_processor': excel_processor,
            'volatility_calculator': volatility_calculator,
            'flow_screener': flow_screener,
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
            "📁 Upload & Classify",
            "🚀 Integrated Flow Processor",
            "🔍 Multi-Leg Flow Screener",
            "📊 Volatility Analysis",
            "🎯 Outcome Tracking",
            "🔮 Predictive Insights",
            "⚙️ Rule Management",
            "📊 Analytics"
        ]
    )

    # Route to appropriate page
    if page == "📈 Dashboard":
        render_dashboard(services)
    elif page == "📁 Upload & Classify":
        render_upload_page(services)
    elif page == "🚀 Integrated Flow Processor":
        render_integrated_processor(services)
    elif page == "🔍 Multi-Leg Flow Screener":
        render_flow_screener(services)
    elif page == "📊 Volatility Analysis":
        render_volatility_analysis(services)
    elif page == "🎯 Outcome Tracking":
        render_outcome_tracking(services)
    elif page == "🔮 Predictive Insights":
        render_predictive_insights(services)
    elif page == "⚙️ Rule Management":
        render_rule_management(services)
    elif page == "📊 Analytics":
        render_analytics(services)


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


def render_upload_page(services):
    """Render the Excel upload and classification interface."""
    st.header("📁 Upload & Classify Trades")

    # File upload section
    st.subheader("Upload Excel File")
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload an Excel file containing options flow data"
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_{uuid.uuid4()}.xlsx"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # File validation
        is_valid, validation_errors = services['excel_processor'].validate_file_format(temp_path)

        if validation_errors:
            st.warning("File validation warnings:")
            for error in validation_errors:
                st.write(f"• {error}")

        if is_valid:
            # File preview
            st.subheader("File Preview")
            preview_df, preview_info = services['excel_processor'].get_file_preview(temp_path)

            if not preview_df.empty:
                st.dataframe(preview_df, use_container_width=True)

                for info in preview_info:
                    st.info(info)

                # Processing options
                st.subheader("Processing Options")

                col1, col2 = st.columns(2)
                with col1:
                    classify_trades = st.checkbox("Classify trades automatically", value=True)
                with col2:
                    save_to_db = st.checkbox("Save to database", value=True)

                # Process button
                if st.button("Process File", type="primary"):
                    process_excel_file(services, temp_path, classify_trades, save_to_db)
            else:
                st.error("Unable to preview file. Please check the file format.")
        else:
            st.error("File validation failed. Please check your file format.")


def process_excel_file(services, file_path: str, classify_trades: bool, save_to_db: bool):
    """Process the uploaded Excel file."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(progress: int, message: str):
        progress_bar.progress(progress)
        status_text.text(message)

    try:
        # Process Excel file
        trades, errors = services['excel_processor'].process_upload(file_path, update_progress)

        if errors:
            st.warning("Processing completed with warnings:")
            for error in errors:
                st.write(f"• {error}")

        if trades:
            st.success(f"Successfully processed {len(trades)} trades!")

            # Classify trades if requested
            if classify_trades:
                st.subheader("Classification Results")
                classified_trades = []

                classification_progress = st.progress(0)
                classification_status = st.empty()

                for i, trade in enumerate(trades):
                    try:
                        # Classify individual trade (treating as single-leg for now)
                        classification, expected_outcome, confidence = services['classifier'].classify_multi_leg_trade([trade])

                        trade.classification = classification
                        trade.expected_outcome = expected_outcome
                        trade.confidence_score = confidence

                        classified_trades.append(trade)

                        # Update progress
                        progress = int((i + 1) / len(trades) * 100)
                        classification_progress.progress(progress)
                        classification_status.text(f"Classifying trade {i + 1}/{len(trades)}")

                    except Exception as e:
                        logger.error(f"Classification failed for trade {i}: {e}")
                        trade.classification = "CLASSIFICATION_ERROR"
                        trade.expected_outcome = f"Error: {str(e)}"
                        trade.confidence_score = 0.0
                        classified_trades.append(trade)

                trades = classified_trades
                st.success("Classification completed!")

            # Display results
            display_processed_trades(trades)

            # Save to database if requested
            if save_to_db:
                save_trades_to_database(services, trades)

        else:
            st.error("No valid trades found in the file.")

    except Exception as e:
        st.error(f"Processing failed: {e}")

    finally:
        progress_bar.empty()
        status_text.empty()


def display_processed_trades(trades: List[OptionsFlow]):
    """Display processed trades in a nice format."""
    st.subheader("Processed Trades")

    # Create summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Trades", len(trades))

    with col2:
        classified_count = sum(1 for t in trades if t.classification and t.classification != "UNCLASSIFIED")
        st.metric("Classified", classified_count)

    with col3:
        avg_confidence = np.mean([t.confidence_score for t in trades if t.confidence_score > 0])
        st.metric("Avg Confidence", f"{avg_confidence:.2f}" if not np.isnan(avg_confidence) else "N/A")

    # Classification distribution
    if trades:
        classifications = {}
        for trade in trades:
            if trade.classification:
                classifications[trade.classification] = classifications.get(trade.classification, 0) + 1

        if classifications:
            fig = px.bar(
                x=list(classifications.keys()),
                y=list(classifications.values()),
                title="Classification Distribution"
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    trades_df = pd.DataFrame([{
        'Symbol': t.symbol,
        'Type': f"{t.buy_sell} {t.call_put}",
        'Strike': t.strike,
        'Expiration': t.expiration_date.strftime('%Y-%m-%d'),
        'Classification': t.classification or 'Unclassified',
        'Expected Outcome': t.expected_outcome or 'N/A',
        'Confidence': f"{t.confidence_score:.2f}" if t.confidence_score > 0 else 'N/A',
        'Trade Value': f"${t.trade_value:,.0f}" if t.trade_value else 'N/A'
    } for t in trades])

    st.dataframe(trades_df, use_container_width=True)


def save_trades_to_database(services, trades: List[OptionsFlow]):
    """Save processed trades to the database."""
    st.subheader("Saving to Database")

    save_progress = st.progress(0)
    save_status = st.empty()

    saved_count = 0
    failed_count = 0

    for i, trade in enumerate(trades):
        try:
            if services['db_service'].save_options_flow(trade):
                saved_count += 1
            else:
                failed_count += 1
        except Exception as e:
            logger.error(f"Failed to save trade {i}: {e}")
            failed_count += 1

        # Update progress
        progress = int((i + 1) / len(trades) * 100)
        save_progress.progress(progress)
        save_status.text(f"Saving trade {i + 1}/{len(trades)}")

    save_progress.empty()
    save_status.empty()

    if saved_count > 0:
        st.success(f"Successfully saved {saved_count} trades to database!")

    if failed_count > 0:
        st.warning(f"Failed to save {failed_count} trades.")


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

                notes = st.text_area("Notes (optional):")

                if st.button("Record Outcome", type="primary"):
                    if services['outcome_tracker'].record_outcome(trade_id, outcome):
                        st.success("Outcome recorded successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to record outcome.")

    else:
        st.info("No trades pending outcome recording.")

    # Outcome statistics
    st.subheader("Outcome Statistics")

    # Get classification summary
    summary = services['outcome_tracker'].get_classification_summary()

    if summary:
        summary_df = pd.DataFrame([{
            'Classification': classification,
            'Total Trades': data['total_trades'],
            'Correct Predictions': data['correct_predictions'],
            'Accuracy': f"{data['accuracy']:.1%}",
            'Avg Trade Value': f"${data['avg_trade_value']:,.0f}",
            'Confidence Interval': f"({data['confidence_interval'][0]:.1%}, {data['confidence_interval'][1]:.1%})"
        } for classification, data in summary.items()])

        st.dataframe(summary_df, use_container_width=True)

        # Accuracy chart
        fig = px.bar(
            summary_df,
            x='Classification',
            y='Accuracy',
            title="Classification Accuracy by Type"
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No outcome data available yet.")


def render_predictive_insights(services):
    """Render the predictive insights query interface."""
    st.header("🔮 Predictive Insights")

    # Query interface
    st.subheader("Ask About Trade Patterns")

    query_examples = [
        "What is the success rate of NEGATIVE ITM trades?",
        "How do earnings trades perform compared to regular trades?",
        "What happens with trades under $100k?",
        "Show me STRADDLE trade outcomes",
        "What are the best performing classifications?"
    ]

    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input(
            "Enter your query:",
            placeholder="e.g., What is the success rate of NEGATIVE ITM trades?"
        )

    with col2:
        example_query = st.selectbox("Or select an example:", [""] + query_examples)
        if example_query:
            query = example_query

    if query:
        if st.button("Generate Insights", type="primary"):
            with st.spinner("Analyzing data..."):
                insights = services['predictive_model'].generate_insights(query)

                if insights.get('insights'):
                    st.subheader("Analysis Results")

                    for insight in insights['insights']:
                        st.write(f"• {insight}")

                    # Display confidence intervals if available
                    if insights.get('confidence_intervals'):
                        st.subheader("Confidence Intervals")
                        ci_df = pd.DataFrame([{
                            'Outcome': outcome,
                            'Lower Bound': f"{ci[0]:.1%}",
                            'Upper Bound': f"{ci[1]:.1%}"
                        } for outcome, ci in insights['confidence_intervals'].items()])
                        st.dataframe(ci_df)

                    # Display sample sizes
                    if insights.get('sample_sizes'):
                        st.subheader("Sample Sizes")
                        for category, size in insights['sample_sizes'].items():
                            st.write(f"• {category}: {size} trades")

                    # Display recommendations
                    if insights.get('recommendations'):
                        st.subheader("Recommendations")
                        for rec in insights['recommendations']:
                            st.write(f"💡 {rec}")

                else:
                    st.warning("No insights generated. Try a different query.")

    # Probability calculator
    st.subheader("Outcome Probability Calculator")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Get available classifications
        all_trades = services['db_service'].get_options_flows()
        classifications = list(set(t.classification for t in all_trades if t.classification))

        selected_classification = st.selectbox("Classification:", classifications if classifications else ["No data"])

    with col2:
        trade_value = st.number_input("Trade Value ($):", min_value=0, value=50000, step=1000)

    with col3:
        is_earnings = st.checkbox("Earnings Trade")

    if selected_classification and selected_classification != "No data":
        if st.button("Calculate Probabilities"):
            probabilities = services['predictive_model'].predict_outcome_probability(
                selected_classification, trade_value, is_earnings
            )

            if probabilities:
                st.subheader("Outcome Probabilities")

                # Create probability chart
                outcomes = [k for k in probabilities.keys() if k in services['outcome_tracker'].valid_outcomes]
                probs = [probabilities[k] for k in outcomes]

                fig = px.bar(
                    x=outcomes,
                    y=probs,
                    title=f"Predicted Outcomes for {selected_classification}",
                    labels={'x': 'Outcome', 'y': 'Probability'}
                )
                fig.update_yaxis(tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)

                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence Score", f"{probabilities.get('confidence_score', 0):.2f}")
                with col2:
                    st.metric("Sample Size", probabilities.get('sample_size', 0))


def render_rule_management(services):
    """Render the rule management interface."""
    st.header("⚙️ Rule Management")

    # Get all rules
    all_rules = services['rules_engine'].get_all_rules()

    # Rule management tabs
    tab1, tab2, tab3 = st.tabs(["View Rules", "Add Rule", "Rule Analytics"])

    with tab1:
        st.subheader("Current Classification Rules")

        if all_rules:
            for rule in all_rules:
                with st.expander(f"{rule.name} ({'Active' if rule.is_active else 'Inactive'})"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Description:** {rule.description}")
                        st.write(f"**Expected Outcome:** {rule.expected_outcome}")
                        st.write(f"**Success Rate:** {rule.success_rate:.1%}" if rule.success_rate else "**Success Rate:** Not calculated")

                    with col2:
                        st.write(f"**Created:** {rule.created_date.strftime('%Y-%m-%d')}" if rule.created_date else "**Created:** Unknown")
                        st.write(f"**Last Updated:** {rule.updated_date.strftime('%Y-%m-%d')}" if rule.updated_date else "**Last Updated:** Unknown")
                        st.write(f"**Keywords:** {', '.join(rule.result_keywords)}")

                    # Rule actions
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if rule.is_active:
                            if st.button(f"Deactivate", key=f"deactivate_{rule.rule_id}"):
                                if services['rules_engine'].deactivate_rule(rule.rule_id):
                                    st.success("Rule deactivated!")
                                    st.rerun()
                        else:
                            if st.button(f"Activate", key=f"activate_{rule.rule_id}"):
                                if services['rules_engine'].activate_rule(rule.rule_id):
                                    st.success("Rule activated!")
                                    st.rerun()

                    with col2:
                        if st.button(f"Update Metrics", key=f"update_{rule.rule_id}"):
                            metrics = services['rules_engine'].get_rule_effectiveness_metrics(rule.rule_id)
                            st.write(f"Updated metrics: {metrics}")

                    with col3:
                        if st.button(f"Delete", key=f"delete_{rule.rule_id}", type="secondary"):
                            if st.confirm(f"Delete rule '{rule.name}'?"):
                                if services['rules_engine'].delete_rule(rule.rule_id):
                                    st.success("Rule deleted!")
                                    st.rerun()
        else:
            st.info("No rules found.")

    with tab2:
        st.subheader("Add New Classification Rule")

        with st.form("add_rule_form"):
            rule_name = st.text_input("Rule Name:")
            rule_description = st.text_area("Description:")
            expected_outcome = st.text_input("Expected Outcome:")

            # Simple logic builder
            st.write("**Rule Logic (simplified):**")
            logic_field = st.selectbox("Field:", ["classification", "trade_value", "symbol", "er_flag"])
            logic_operator = st.selectbox("Operator:", ["==", ">=", "<=", "!=", "in"])
            logic_value = st.text_input("Value:")

            result_keywords = st.text_input("Result Keywords (comma-separated):")

            if st.form_submit_button("Add Rule"):
                if rule_name and expected_outcome:
                    # Build logic dict
                    logic = {logic_field: {logic_operator: logic_value}}
                    keywords = [k.strip() for k in result_keywords.split(",") if k.strip()]

                    new_rule = ClassificationRule(
                        rule_id=str(uuid.uuid4()),
                        name=rule_name,
                        description=rule_description,
                        classification_logic=logic,
                        expected_outcome=expected_outcome,
                        result_keywords=keywords
                    )

                    if services['rules_engine'].add_rule(new_rule):
                        st.success("Rule added successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to add rule. Check for conflicts.")
                else:
                    st.error("Please fill in required fields.")

    with tab3:
        st.subheader("Rule Performance Analytics")

        if all_rules:
            # Rule effectiveness chart
            rule_metrics = []
            for rule in all_rules:
                if rule.success_rate is not None:
                    rule_metrics.append({
                        'Rule': rule.name,
                        'Success Rate': rule.success_rate,
                        'Status': 'Active' if rule.is_active else 'Inactive'
                    })

            if rule_metrics:
                metrics_df = pd.DataFrame(rule_metrics)

                fig = px.bar(
                    metrics_df,
                    x='Rule',
                    y='Success Rate',
                    color='Status',
                    title="Rule Performance Comparison"
                )
                fig.update_xaxis(tickangle=45)
                fig.update_yaxis(tickformat='.1%')
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.info("No rule performance data available.")
        else:
            st.info("No rules to analyze.")


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

            # Processing options
            st.subheader("⚙️ Processing Options")
            col1, col2, col3 = st.columns(3)

            with col1:
                auto_predict = st.checkbox("Auto-predict outcomes", value=True,
                                         help="Automatically predict trade outcomes based on analysis")
            with col2:
                store_results = st.checkbox("Store in database", value=True,
                                          help="Save processed results to Supabase")
            with col3:
                generate_insights = st.checkbox("Generate insights", value=True,
                                               help="Create predictive insights and recommendations")

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

                    # Efficiency metrics
                    summary = results.get('summary', {})
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        efficiency = summary.get('screening_efficiency', 0)
                        st.metric("Screening Efficiency", f"{efficiency:.1f}%")
                    with col2:
                        total_premium = summary.get('total_premium', 0)
                        st.metric("Total Premium", f"${total_premium:,.0f}")
                    with col3:
                        unique_symbols = summary.get('unique_symbols', 0)
                        st.metric("Unique Symbols", unique_symbols)

                    # Detailed results
                    trades_data = results.get('trades_data')
                    if trades_data is not None and not trades_data.empty:

                        # Classification breakdown
                        if 'classification_breakdown' in summary:
                            st.subheader("🎯 Classification Breakdown")
                            class_df = pd.DataFrame(
                                list(summary['classification_breakdown'].items()),
                                columns=['Classification', 'Count']
                            )
                            fig = px.pie(class_df, values='Count', names='Classification',
                                       title="Trade Classifications")
                            st.plotly_chart(fig, use_container_width=True)

                        # Volatility analysis
                        if 'volatility_breakdown' in summary:
                            st.subheader("📊 Volatility Analysis")
                            vol_df = pd.DataFrame(
                                list(summary['volatility_breakdown'].items()),
                                columns=['Volatility Flag', 'Count']
                            )
                            fig = px.bar(vol_df, x='Volatility Flag', y='Count',
                                       title="Volatility Premium Analysis",
                                       color='Volatility Flag')
                            st.plotly_chart(fig, use_container_width=True)

                        # Direction analysis
                        if 'direction_breakdown' in summary:
                            st.subheader("📈 Direction Analysis")
                            dir_df = pd.DataFrame(
                                list(summary['direction_breakdown'].items()),
                                columns=['Direction', 'Count']
                            )
                            fig = px.bar(dir_df, x='Direction', y='Count',
                                       title="Trade Direction Distribution",
                                       color='Direction')
                            st.plotly_chart(fig, use_container_width=True)

                        # Processed trades table
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

                        # Color code by volatility flag
                        def highlight_volatility(row):
                            if 'volatility_flag' in row:
                                if row['volatility_flag'] == 'EXPENSIVE':
                                    return ['background-color: #ffebee'] * len(row)
                                elif row['volatility_flag'] == 'CHEAP':
                                    return ['background-color: #e8f5e8'] * len(row)
                            return [''] * len(row)

                        styled_df = display_df.style.apply(highlight_volatility, axis=1)
                        st.dataframe(styled_df, use_container_width=True)

                        # Insights and recommendations
                        insights = results.get('insights', {})
                        if insights:
                            st.subheader("🔮 Insights & Recommendations")

                            recommendations = insights.get('recommendations', [])
                            if recommendations:
                                for i, rec in enumerate(recommendations, 1):
                                    st.info(f"💡 **Recommendation {i}:** {rec}")

                            # Movement correlations
                            movement_corr = insights.get('movement_correlations', {})
                            if movement_corr:
                                st.write("**Movement Correlations:**")
                                for key, count in movement_corr.items():
                                    st.write(f"• {key}: {count} trades")

                        # Export options
                        st.subheader("📥 Export Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button("Download Processed Trades"):
                                csv = trades_data.to_csv(index=False)
                                st.download_button(
                                    label="📄 Download CSV",
                                    data=csv,
                                    file_name=f"processed_flows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )

                        with col2:
                            if st.button("Download Summary Report"):
                                summary_text = f"""
# Flow Processing Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Processing Statistics
- Original Flows: {results['original_flows']}
- Multi-Leg Trades: {results['multi_leg_trades']}
- Classified Trades: {results['classified_trades']}
- Stored in Database: {results['stored_trades']}
- Screening Efficiency: {summary.get('screening_efficiency', 0):.1f}%

## Insights
{chr(10).join(f"• {rec}" for rec in insights.get('recommendations', []))}
                                """
                                st.download_button(
                                    label="📊 Download Report",
                                    data=summary_text,
                                    file_name=f"flow_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain"
                                )

                    else:
                        st.warning("No trades data available for display.")

                else:
                    error_msg = results.get('error', 'Unknown error occurred')
                    st.error(f"❌ Processing failed: {error_msg}")

                    st.info("""
                    **Troubleshooting Tips:**
                    - Ensure your CSV has the required columns
                    - Check that Symbol, Buy/Sell, CallPut, Strike, Premium, Volume columns exist
                    - Verify CreatedDateTime format or separate CreatedDate/CreatedTime columns
                    - Make sure data contains multi-leg trades (multiple rows per timestamp)
                    """)

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.info("Please ensure your CSV file has the correct format and required columns.")

    else:
        st.info("👆 Upload a CSV file to start the complete workflow")

        # Show expected format
        st.subheader("📋 Expected CSV Format")
        st.markdown("""
        Your CSV should contain multi-leg options flow data with these columns:

        **Required Columns:**
        - `Symbol`: Stock ticker (e.g., AAPL, MSFT)
        - `Buy/Sell`: BUY or SELL (or use `Side` with A/AA=BUY, others=SELL)
        - `CallPut`: CALL or PUT
        - `Strike`: Strike price
        - `Spot`: Current stock price
        - `Premium`: Premium amount
        - `Volume`: Trade volume
        - `OI`: Open interest
        - `Price`: Price per contract
        - `CreatedDateTime`: Timestamp (or separate `CreatedDate` and `CreatedTime`)

        **Optional Columns:**
        - `Color`, `Side`, `ER`, `ImpliedVolatility`, `Dte`, `MktCap`, `Sector`, etc.
        """)

        # Generate sample data button
        if st.button("🎲 Generate Sample Data for Testing"):
            st.info("Run `python generate_sample_data.py` to create sample CSV files for testing.")


def render_flow_screener(services):
    """Render the enhanced multi-leg flow screener."""
    st.header("🔍 Multi-Leg Flow Screener")

    st.markdown("""
    This screener focuses on **synthetic multi-leg options trades** that show conviction through
    sophisticated buy/sell combinations. It filters for trades with strong directional bias and
    significant premium values.
    """)

    # File upload for daily flows
    st.subheader("Upload Daily Flow Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with daily options flow data",
        type=['csv'],
        help="Upload your daily options flow data in CSV format"
    )

    if uploaded_file is not None:
        try:
            # Read the CSV file
            flows_df = pd.read_csv(uploaded_file)

            st.subheader("Data Preview")
            st.write(f"Loaded {len(flows_df)} flow records")
            st.dataframe(flows_df.head(), use_container_width=True)

            # Processing options
            col1, col2, col3 = st.columns(3)

            with col1:
                min_premium = st.number_input("Minimum Premium ($)", value=100000, step=10000)
            with col2:
                min_volume = st.number_input("Minimum Volume", value=300, step=50)
            with col3:
                directional_threshold = st.slider("Directional Bias Threshold", 0.5, 0.9, 0.7, 0.05)

            # Process button
            if st.button("🔍 Screen Multi-Leg Flows", type="primary"):
                with st.spinner("Screening multi-leg flows..."):
                    # Process the flows
                    results = services['flow_screener'].process_daily_flows(flows_df)

                    if results['multi_leg_trades'].empty:
                        st.warning("No multi-leg trades found matching the criteria.")
                        st.info("Try adjusting the filtering parameters or check your data format.")
                    else:
                        # Display results
                        st.success(f"Found {len(results['multi_leg_trades'])} multi-leg trades!")

                        # Summary metrics
                        summary = results['summary']
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Flows", summary.get('total_flows', 0))
                        with col2:
                            st.metric("Multi-Leg Trades", summary.get('multi_leg_count', 0))
                        with col3:
                            efficiency = summary.get('screening_efficiency', 0)
                            st.metric("Screening Efficiency", f"{efficiency:.1f}%")
                        with col4:
                            total_premium = summary.get('total_premium', 0)
                            st.metric("Total Premium", f"${total_premium:,.0f}")

                        # Direction breakdown
                        if 'direction_breakdown' in summary:
                            st.subheader("Direction Breakdown")
                            direction_df = pd.DataFrame(
                                list(summary['direction_breakdown'].items()),
                                columns=['Direction', 'Count']
                            )
                            fig = px.pie(direction_df, values='Count', names='Direction',
                                       title="Trade Direction Distribution")
                            st.plotly_chart(fig, use_container_width=True)

                        # Multi-leg trades table
                        st.subheader("Multi-Leg Trades")
                        trades_df = results['multi_leg_trades']

                        # Format premium column
                        if 'Premium' in trades_df.columns:
                            trades_df['Premium_Formatted'] = trades_df['Premium'].apply(
                                lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
                            )

                        # Display with styling
                        st.dataframe(
                            trades_df.style.format({
                                'Premium': '${:,.0f}',
                                'Volume': '{:,.0f}',
                                'hv': '{:.2%}' if 'hv' in trades_df.columns else None,
                                'iv': '{:.2%}' if 'iv' in trades_df.columns else None
                            }),
                            use_container_width=True
                        )

                        # Volatility analysis
                        if not results['volatility_analysis'].empty:
                            st.subheader("Volatility Analysis")
                            vol_df = results['volatility_analysis']

                            # Volatility flags distribution
                            if 'flag' in vol_df.columns:
                                flag_counts = vol_df['flag'].value_counts()
                                fig = px.bar(
                                    x=flag_counts.index,
                                    y=flag_counts.values,
                                    title="Volatility Premium Analysis",
                                    labels={'x': 'Volatility Flag', 'y': 'Count'}
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            # Detailed volatility table
                            st.dataframe(
                                vol_df.style.format({
                                    'hv': '{:.2%}',
                                    'iv': '{:.2%}',
                                    'volatility_premium': '{:.2%}',
                                    'premium_percentage': '{:.1f}%'
                                }),
                                use_container_width=True
                            )

                        # Export options
                        st.subheader("Export Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            if st.button("📥 Download Multi-Leg Trades"):
                                csv = trades_df.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name=f"multi_leg_trades_{datetime.now().strftime('%Y%m%d')}.csv",
                                    mime="text/csv"
                                )

                        with col2:
                            if not results['volatility_analysis'].empty:
                                if st.button("📥 Download Volatility Analysis"):
                                    vol_csv = results['volatility_analysis'].to_csv(index=False)
                                    st.download_button(
                                        label="Download Volatility CSV",
                                        data=vol_csv,
                                        file_name=f"volatility_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                                        mime="text/csv"
                                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure your CSV file has the required columns: Symbol, Buy/Sell, CallPut, Strike, Spot, ExpirationDate, Premium, Volume, OI, Price, Side, Color, CreatedDateTime")

    else:
        st.info("👆 Upload a CSV file to start screening multi-leg flows")

        # Show expected format
        st.subheader("Expected CSV Format")
        sample_data = {
            'Symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
            'Buy/Sell': ['BUY', 'SELL', 'BUY', 'SELL'],
            'CallPut': ['CALL', 'PUT', 'CALL', 'PUT'],
            'Strike': [150, 145, 300, 295],
            'Spot': [148, 148, 298, 298],
            'ExpirationDate': ['2024-01-19', '2024-01-19', '2024-01-19', '2024-01-19'],
            'Premium': [250000, -180000, 150000, -120000],
            'Volume': [500, 400, 300, 350],
            'OI': [1000, 800, 600, 700],
            'Price': [2.50, 1.80, 1.50, 1.20],
            'Side': ['A', 'B', 'A', 'B'],
            'Color': ['GREEN', 'RED', 'GREEN', 'RED'],
            'CreatedDateTime': ['2024-01-15 09:30:00', '2024-01-15 09:30:00', '2024-01-15 10:15:00', '2024-01-15 10:15:00']
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)


def render_volatility_analysis(services):
    """Render volatility analysis dashboard."""
    st.header("📊 Volatility Analysis Dashboard")

    st.markdown("""
    Compare **Historical Volatility (Yang-Zhang method)** vs **Implied Volatility** to identify
    expensive or cheap options contracts. This analysis helps determine optimal entry points
    for options strategies.
    """)

    # Symbol input
    st.subheader("Analyze Individual Symbols")

    col1, col2 = st.columns(2)

    with col1:
        symbol_input = st.text_input("Enter Symbol:", value="AAPL", help="Enter a stock symbol to analyze")

    with col2:
        hv_period = st.selectbox("HV Calculation Period:", [20, 30, 60, 90], index=1)

    if symbol_input:
        if st.button("🔍 Analyze Volatility", type="primary"):
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

                    # Visualization
                    if analysis.get('hv') and analysis.get('iv'):
                        fig = go.Figure()

                        fig.add_trace(go.Bar(
                            name='Historical Volatility',
                            x=['Volatility'],
                            y=[analysis['hv']],
                            marker_color='blue'
                        ))

                        fig.add_trace(go.Bar(
                            name='Implied Volatility',
                            x=['Volatility'],
                            y=[analysis['iv']],
                            marker_color='orange'
                        ))

                        fig.update_layout(
                            title=f"HV vs IV Comparison for {symbol_input}",
                            yaxis_title="Volatility",
                            yaxis_tickformat='.1%',
                            barmode='group'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # Next expiration info
                    if analysis.get('next_expiration'):
                        st.write(f"**Next Monthly Expiration:** {analysis['next_expiration'].strftime('%Y-%m-%d')}")

                else:
                    st.error(f"Analysis failed: {analysis.get('message', 'Unknown error')}")

    # Batch analysis
    st.subheader("Batch Volatility Analysis")

    symbols_input = st.text_area(
        "Enter multiple symbols (comma-separated):",
        value="AAPL, MSFT, GOOGL, TSLA, NVDA",
        help="Enter multiple symbols separated by commas"
    )

    if symbols_input:
        if st.button("🔍 Batch Analyze", type="secondary"):
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]

            if symbols:
                with st.spinner(f"Analyzing {len(symbols)} symbols..."):
                    batch_results = services['volatility_calculator'].batch_analyze_volatility(symbols)

                    if not batch_results.empty:
                        # Summary metrics
                        summary = services['volatility_calculator'].get_volatility_summary(symbols)

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Symbols", summary.get('total_symbols', 0))
                        with col2:
                            expensive_count = summary.get('expensive_count', 0)
                            st.metric("Expensive Options", expensive_count)
                        with col3:
                            cheap_count = summary.get('cheap_count', 0)
                            st.metric("Cheap Options", cheap_count)
                        with col4:
                            avg_premium = summary.get('avg_premium', 0)
                            st.metric("Avg Vol Premium", f"{avg_premium:.2%}" if avg_premium else "N/A")

                        # Results table
                        st.subheader("Batch Analysis Results")

                        # Color code the results
                        def highlight_volatility_flag(row):
                            if row['flag'] == 'EXPENSIVE':
                                return ['background-color: #ffebee'] * len(row)
                            elif row['flag'] == 'CHEAP':
                                return ['background-color: #e8f5e8'] * len(row)
                            else:
                                return [''] * len(row)

                        styled_df = batch_results.style.apply(highlight_volatility_flag, axis=1).format({
                            'hv': '{:.2%}',
                            'iv': '{:.2%}',
                            'volatility_premium': '{:.2%}',
                            'premium_percentage': '{:.1f}%'
                        })

                        st.dataframe(styled_df, use_container_width=True)

                        # Visualization
                        valid_data = batch_results.dropna(subset=['hv', 'iv'])

                        if not valid_data.empty:
                            fig = px.scatter(
                                valid_data,
                                x='hv',
                                y='iv',
                                color='flag',
                                hover_data=['symbol'],
                                title="HV vs IV Scatter Plot",
                                labels={'hv': 'Historical Volatility', 'iv': 'Implied Volatility'}
                            )

                            # Add diagonal line (HV = IV)
                            min_vol = min(valid_data['hv'].min(), valid_data['iv'].min())
                            max_vol = max(valid_data['hv'].max(), valid_data['iv'].max())

                            fig.add_trace(go.Scatter(
                                x=[min_vol, max_vol],
                                y=[min_vol, max_vol],
                                mode='lines',
                                name='HV = IV',
                                line=dict(dash='dash', color='gray')
                            ))

                            fig.update_xaxis(tickformat='.1%')
                            fig.update_yaxis(tickformat='.1%')

                            st.plotly_chart(fig, use_container_width=True)

                        # Export option
                        if st.button("📥 Download Results"):
                            csv = batch_results.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"volatility_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )

                    else:
                        st.warning("No valid results obtained from the analysis.")
            else:
                st.warning("Please enter valid symbols.")

    # Educational content
    with st.expander("📚 Understanding Volatility Analysis"):
        st.markdown("""
        ### Yang-Zhang Historical Volatility
        - **More accurate** than simple close-to-close volatility
        - **Accounts for overnight gaps** and intraday price movements
        - **Formula includes**: Open, High, Low, Close prices

        ### Interpretation
        - **IV > HV (Expensive)**: Options are trading at a premium to historical volatility
        - **IV < HV (Cheap)**: Options are trading at a discount to historical volatility
        - **Consider market conditions**: Earnings, events, market stress can affect IV

        ### Trading Implications
        - **Expensive options**: Consider selling strategies (covered calls, cash-secured puts)
        - **Cheap options**: Consider buying strategies (long calls/puts, straddles)
        - **Always consider**: Time decay, directional bias, and risk management
        """)


def render_analytics(services):
    """Render comprehensive analytics dashboard."""
    st.header("📊 Analytics Dashboard")

    # Time period selector
    col1, col2 = st.columns(2)

    with col1:
        time_period = st.selectbox("Time Period:", ["Last 7 days", "Last 30 days", "Last 90 days", "All time"])

    with col2:
        analysis_type = st.selectbox("Analysis Type:", ["Overview", "Classification Analysis", "Earnings Analysis", "Value Analysis"])

    # Calculate date filter
    if time_period == "Last 7 days":
        date_filter = (datetime.now() - timedelta(days=7)).isoformat()
    elif time_period == "Last 30 days":
        date_filter = (datetime.now() - timedelta(days=30)).isoformat()
    elif time_period == "Last 90 days":
        date_filter = (datetime.now() - timedelta(days=90)).isoformat()
    else:
        date_filter = None

    # Get filtered data
    filters = {'date_from': date_filter} if date_filter else {}
    trades = services['db_service'].get_options_flows(filters)

    if not trades:
        st.warning("No data available for the selected time period.")
        return

    if analysis_type == "Overview":
        render_overview_analytics(trades, services)
    elif analysis_type == "Classification Analysis":
        render_classification_analytics(trades, services)
    elif analysis_type == "Earnings Analysis":
        render_earnings_analytics(trades, services)
    elif analysis_type == "Value Analysis":
        render_value_analytics(trades, services)


def render_overview_analytics(trades, services):
    """Render overview analytics."""
    st.subheader("Overview Analytics")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", len(trades))

    with col2:
        unique_symbols = len(set(t.symbol for t in trades))
        st.metric("Unique Symbols", unique_symbols)

    with col3:
        total_value = sum(t.trade_value for t in trades if t.trade_value)
        st.metric("Total Trade Value", f"${total_value:,.0f}")

    with col4:
        avg_value = total_value / len(trades) if trades else 0
        st.metric("Avg Trade Value", f"${avg_value:,.0f}")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Symbol distribution
        symbol_counts = {}
        for trade in trades:
            symbol_counts[trade.symbol] = symbol_counts.get(trade.symbol, 0) + 1

        top_symbols = dict(sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10])

        fig = px.bar(
            x=list(top_symbols.keys()),
            y=list(top_symbols.values()),
            title="Top 10 Symbols by Trade Count"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Call vs Put distribution
        call_put_counts = {'CALL': 0, 'PUT': 0}
        for trade in trades:
            call_put_counts[trade.call_put] = call_put_counts.get(trade.call_put, 0) + 1

        fig = px.pie(
            values=list(call_put_counts.values()),
            names=list(call_put_counts.keys()),
            title="Call vs Put Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_classification_analytics(trades, services):
    """Render classification-specific analytics."""
    st.subheader("Classification Analytics")

    # Classification performance
    classified_trades = [t for t in trades if t.classification]

    if classified_trades:
        # Accuracy by classification
        accuracy_data = []
        for classification in set(t.classification for t in classified_trades):
            class_trades = [t for t in classified_trades if t.classification == classification]
            trades_with_outcomes = [t for t in class_trades if t.actual_outcome]

            if trades_with_outcomes:
                correct = sum(1 for t in trades_with_outcomes
                            if services['outcome_tracker']._outcomes_match(t.expected_outcome, t.actual_outcome))
                accuracy = correct / len(trades_with_outcomes)

                accuracy_data.append({
                    'Classification': classification,
                    'Accuracy': accuracy,
                    'Total Trades': len(class_trades),
                    'Trades with Outcomes': len(trades_with_outcomes)
                })

        if accuracy_data:
            accuracy_df = pd.DataFrame(accuracy_data)

            fig = px.scatter(
                accuracy_df,
                x='Total Trades',
                y='Accuracy',
                size='Trades with Outcomes',
                hover_data=['Classification'],
                title="Classification Performance (Accuracy vs Volume)"
            )
            fig.update_yaxis(tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(accuracy_df, use_container_width=True)

    else:
        st.info("No classified trades found.")


def render_earnings_analytics(trades, services):
    """Render earnings-specific analytics."""
    st.subheader("Earnings Analytics")

    earnings_trades = [t for t in trades if t.er_flag]
    regular_trades = [t for t in trades if not t.er_flag]

    if earnings_trades:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Earnings Trades", len(earnings_trades))
            earnings_value = sum(t.trade_value for t in earnings_trades if t.trade_value)
            st.metric("Earnings Trade Value", f"${earnings_value:,.0f}")

        with col2:
            st.metric("Regular Trades", len(regular_trades))
            regular_value = sum(t.trade_value for t in regular_trades if t.trade_value)
            st.metric("Regular Trade Value", f"${regular_value:,.0f}")

        # Earnings vs Regular performance
        comparison = services['outcome_tracker'].get_earnings_vs_regular_performance()

        if comparison:
            comparison_df = pd.DataFrame([
                {
                    'Type': 'Earnings',
                    'Accuracy': comparison['earnings_trades'].get('accuracy', 0),
                    'Total Trades': comparison['earnings_trades'].get('total_trades', 0),
                    'Avg Value': comparison['earnings_trades'].get('avg_trade_value', 0)
                },
                {
                    'Type': 'Regular',
                    'Accuracy': comparison['regular_trades'].get('accuracy', 0),
                    'Total Trades': comparison['regular_trades'].get('total_trades', 0),
                    'Avg Value': comparison['regular_trades'].get('avg_trade_value', 0)
                }
            ])

            fig = px.bar(
                comparison_df,
                x='Type',
                y='Accuracy',
                title="Earnings vs Regular Trade Accuracy"
            )
            fig.update_yaxis(tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No earnings trades found.")


def render_value_analytics(trades, services):
    """Render value-based analytics."""
    st.subheader("Value Analytics")

    trades_with_value = [t for t in trades if t.trade_value and t.trade_value > 0]

    if trades_with_value:
        # Value distribution
        values = [t.trade_value for t in trades_with_value]

        fig = px.histogram(
            x=values,
            nbins=20,
            title="Trade Value Distribution"
        )
        fig.update_xaxis(title="Trade Value ($)")
        st.plotly_chart(fig, use_container_width=True)

        # High vs Low value performance
        threshold = 100000  # $100k
        high_value = [t for t in trades_with_value if t.trade_value >= threshold]
        low_value = [t for t in trades_with_value if t.trade_value < threshold]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("High Value Trades (≥$100k)", len(high_value))
            if high_value:
                high_value_outcomes = [t for t in high_value if t.actual_outcome]
                if high_value_outcomes:
                    high_accuracy = sum(1 for t in high_value_outcomes
                                      if services['outcome_tracker']._outcomes_match(t.expected_outcome, t.actual_outcome))
                    high_accuracy_rate = high_accuracy / len(high_value_outcomes)
                    st.metric("High Value Accuracy", f"{high_accuracy_rate:.1%}")

        with col2:
            st.metric("Low Value Trades (<$100k)", len(low_value))
            if low_value:
                low_value_outcomes = [t for t in low_value if t.actual_outcome]
                if low_value_outcomes:
                    low_accuracy = sum(1 for t in low_value_outcomes
                                     if services['outcome_tracker']._outcomes_match(t.expected_outcome, t.actual_outcome))
                    low_accuracy_rate = low_accuracy / len(low_value_outcomes)
                    st.metric("Low Value Accuracy", f"{low_accuracy_rate:.1%}")

    else:
        st.info("No trade value data available.")


if __name__ == "__main__":
    main()
