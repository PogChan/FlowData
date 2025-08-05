"""
Integrated Flow Processor - Complete workflow implementation
Handles the entire process from CSV upload to predictive analysis.
"""

import pandas as pd
import numpy as np
import uuid
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from services.volatility_calculator import VolatilityCalculator
from services.polygon_api_client import PolygonAPIClient
from services.database_service import SupabaseService
# Removed unused import
from services.outcome_tracker import OutcomeTracker
from models.data_models import OptionsFlow

logger = logging.getLogger(__name__)


class IntegratedFlowProcessor:
    """
    Complete workflow processor for multi-leg options flow analysis.
    Implements your exact 6-step workflow.
    """

    def __init__(self, polygon_client: PolygonAPIClient, volatility_calculator: VolatilityCalculator,
                 db_service: SupabaseService, predictive_model, outcome_tracker: OutcomeTracker):
        self.polygon_client = polygon_client
        self.volatility_calculator = volatility_calculator
        self.db_service = db_service
        self.predictive_model = predictive_model
        self.outcome_tracker = outcome_tracker

    def process_daily_flows(self, flows_df: pd.DataFrame, progress_callback=None) -> Dict[str, Any]:
        """
        Complete workflow: Upload -> Screen -> Classify -> Analyze -> Store -> Predict
        """
        try:
            logger.info(f"Starting integrated flow processing for {len(flows_df)} flows")

            # Step 1: Prepare and validate data
            if progress_callback:
                progress_callback(10, "Preparing flow data...")

            prepared_flows = self._prepare_flow_data(flows_df)
            if prepared_flows.empty:
                return {'error': 'No valid flows after data preparation'}

            # Step 2: Multi-leg screening (your exact logic)
            if progress_callback:
                progress_callback(25, "Screening multi-leg trades...")

            multi_leg_trades = self._screen_multi_leg_trades(prepared_flows)
            if multi_leg_trades.empty:
                return {'error': 'No multi-leg trades found'}

            # Step 3: Classification and volatility analysis
            if progress_callback:
                progress_callback(50, "Classifying trades and analyzing volatility...")

            classified_trades = self._classify_and_analyze(multi_leg_trades)

            # Step 4: Stock movement analysis for predictive model
            if progress_callback:
                progress_callback(70, "Analyzing stock movements...")

            trades_with_movement = self._analyze_stock_movements(classified_trades)

            # Step 5: Store in Supabase
            if progress_callback:
                progress_callback(85, "Storing results in database...")

            stored_count = self._store_results(trades_with_movement)

            # Step 6: Generate predictive insights
            if progress_callback:
                progress_callback(95, "Generating predictive insights...")

            insights = self._generate_insights(trades_with_movement)

            if progress_callback:
                progress_callback(100, "Processing complete!")

            return {
                'success': True,
                'original_flows': len(flows_df),
                'multi_leg_trades': len(multi_leg_trades),
                'classified_trades': len(classified_trades),
                'stored_trades': stored_count,
                'trades_data': trades_with_movement,
                'insights': insights,
                'summary': self._generate_summary(flows_df, trades_with_movement)
            }

        except Exception as e:
            logger.error(f"Integrated flow processing failed: {e}")
            return {'error': str(e)}

    def _prepare_flow_data(self, flows_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare flow data exactly as in your flow_screener.py
        """
        try:
            df = flows_df.copy()

            # Handle datetime creation from separate date/time columns
            if 'CreatedDate' in df.columns and 'CreatedTime' in df.columns:
                df['CreatedDateTime'] = pd.to_datetime(
                    df['CreatedDate'] + ' ' + df['CreatedTime'],
                    format='%m/%d/%Y %I:%M:%S %p',
                    errors='coerce'
                )
                df = df.drop(columns=['CreatedDate', 'CreatedTime'])
            elif 'CreatedDateTime' not in df.columns:
                df['CreatedDateTime'] = datetime.now()

            # Data type conversions
            df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], errors='coerce')
            df['CreatedDateTime'] = pd.to_datetime(df['CreatedDateTime'], errors='coerce')

            # Create Buy/Sell from Side if needed
            if 'Buy/Sell' not in df.columns and 'Side' in df.columns:
                df['Buy/Sell'] = df['Side'].apply(lambda x: 'BUY' if x in ['A', 'AA'] else 'SELL')

            # Clean and standardize
            df['Buy/Sell'] = df['Buy/Sell'].str.upper()
            df['CallPut'] = df['CallPut'].str.upper()
            df['Color'] = df['Color'].str.upper() if 'Color' in df.columns else 'UNKNOWN'

            # Remove invalid data
            df = df.dropna(subset=['Symbol', 'Strike', 'Premium', 'Volume'])
            df = df[df['Volume'] > 0]
            df = df[df['Premium'] > 0]

            # Sort by created datetime
            df = df.sort_values('CreatedDateTime', ascending=True).reset_index(drop=True)

            logger.info(f"Prepared {len(df)} valid flows")
            return df

        except Exception as e:
            logger.error(f"Failed to prepare flow data: {e}")
            return pd.DataFrame()

    def _screen_multi_leg_trades(self, flows_df: pd.DataFrame) -> pd.DataFrame:
        """
        Multi-leg screening using your exact logic from flow_screener.py
        """
        try:
            # 1) Filter down to true multi-legs
            multi_leg_candidates = (
                flows_df
                .groupby(['Symbol', 'CreatedDateTime'])
                .filter(lambda g: len(g) > 1)
                .copy()
            )

            if multi_leg_candidates.empty:
                return pd.DataFrame()

            # Make sell premiums negative
            multi_leg_candidates['Premium'] = multi_leg_candidates.apply(
                lambda r: -r['Premium'] if r['Buy/Sell'] == 'SELL' else r['Premium'],
                axis=1
            )

            # 2) Build signature for each timestamped group
            def make_signature(g):
                cnt = (
                    g
                    .groupby(['Buy/Sell', 'CallPut', 'Strike', 'ExpirationDate'])
                    .size()
                    .reset_index(name='count')
                )
                return ";".join(
                    sorted(
                        f"{row['Buy/Sell']}_{row['CallPut']}_{row['Strike']}_{row['ExpirationDate']}_{row['count']}"
                        for _, row in cnt.iterrows()
                    )
                )

            # Create signatures
            sigs = (
                multi_leg_candidates
                .groupby(['Symbol', 'CreatedDateTime'])
                .apply(lambda g: pd.Series({'Signature': make_signature(g)}))
                .reset_index()
            )

            # Attach signatures
            multi_leg_candidates = multi_leg_candidates.merge(
                sigs, on=['Symbol', 'CreatedDateTime'], how='left'
            )

            # 3) Aggregate within each unique leg
            agg = (
                multi_leg_candidates
                .groupby(['Symbol', 'CallPut', 'Strike', 'Buy/Sell', 'ExpirationDate', 'Signature'], as_index=False)
                .agg(
                    TotalVolume=('Volume', 'sum'),
                    TotalPremium=('Premium', 'sum'),
                    MinOI=('OI', 'min'),
                    PriceMean=('Price', 'mean'),
                    SetCount=('Symbol', 'size'),
                    CreatedDateTime=('CreatedDateTime', 'first'),
                    Spot=('Spot', 'first'),
                    Side=('Side', 'first'),
                    Color=('Color', 'first'),
                    ER=('ER', 'first') if 'ER' in multi_leg_candidates.columns else ('Symbol', lambda x: 'F'),
                    ImpliedVolatility=('ImpliedVolatility', 'mean') if 'ImpliedVolatility' in multi_leg_candidates.columns else ('Symbol', lambda x: 0.0),
                    Dte=('Dte', 'first') if 'Dte' in multi_leg_candidates.columns else ('Symbol', lambda x: 0),
                    MktCap=('MktCap', 'first') if 'MktCap' in multi_leg_candidates.columns else ('Symbol', lambda x: 'Unknown'),
                    Sector=('Sector', 'first') if 'Sector' in multi_leg_candidates.columns else ('Symbol', lambda x: 'Unknown'),
                    StockEtf=('StockEtf', 'first') if 'StockEtf' in multi_leg_candidates.columns else ('Symbol', lambda x: 'Stock'),
                    Uoa=('Uoa', 'first') if 'Uoa' in multi_leg_candidates.columns else ('Symbol', lambda x: 'Normal'),
                    Weekly=('Weekly', 'first') if 'Weekly' in multi_leg_candidates.columns else ('Symbol', lambda x: 'N'),
                    Type=('Type', 'first') if 'Type' in multi_leg_candidates.columns else ('Symbol', lambda x: 'Options')
                )
            )

            # Rename columns
            agg = agg.rename(columns={
                'TotalVolume': 'Volume',
                'TotalPremium': 'Premium',
                'PriceMean': 'Price'
            })

            # Filter by minimum premium threshold
            agg = agg[abs(agg['Premium']) > 100000]

            # Apply your multi-leg filters
            filtered_trades = self._apply_multi_leg_filters(agg)

            logger.info(f"Screened to {len(filtered_trades)} multi-leg trades")
            return filtered_trades

        except Exception as e:
            logger.error(f"Multi-leg screening failed: {e}")
            return pd.DataFrame()

    def _apply_multi_leg_filters(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply your exact multi-leg filtering logic
        """
        try:
            if candidates_df.empty:
                return pd.DataFrame()

            # Filter out straddles/strangles
            def filter_out_straddles_strangles(group):
                buy_call = (group['Buy/Sell'] == 'BUY') & (group['CallPut'] == 'CALL')
                buy_put = (group['Buy/Sell'] == 'BUY') & (group['CallPut'] == 'PUT')

                if buy_call.any() and buy_put.any():
                    buy_call_premium = group.loc[buy_call, 'Premium'].sum()
                    buy_put_premium = group.loc[buy_put, 'Premium'].sum()
                    total_premium = buy_call_premium + buy_put_premium

                    if total_premium > 0:
                        call_contribution = buy_call_premium / total_premium
                        put_contribution = buy_put_premium / total_premium

                        # Filter out balanced straddles/strangles
                        if 0.4 <= call_contribution <= 0.6 and 0.4 <= put_contribution <= 0.6:
                            return False

                return True

            # Multi-leg validation
            def is_multi_leg(group):
                has_buy = (group['Buy/Sell'] == 'BUY').any()
                has_sell = (group['Buy/Sell'] == 'SELL').any()
                call_put_check = {'CALL', 'PUT'}.issubset(group['CallPut'].unique())
                white_count = (group['Color'] == 'WHITE').sum()

                if has_buy and has_sell and call_put_check and white_count == 0:
                    # Volume > OI check
                    if not (group['Volume'] > group['MinOI']).all():
                        return False
                    return True
                return False

            # Apply filters
            filtered_df = candidates_df.groupby(['Symbol', 'CreatedDateTime']).filter(
                filter_out_straddles_strangles
            )

            multi_leg_df = filtered_df.groupby(['Symbol', 'CreatedDateTime']).filter(
                is_multi_leg
            )

            if multi_leg_df.empty:
                return pd.DataFrame()

            # Additional volume and OI filters
            multi_leg_df = multi_leg_df[multi_leg_df['Volume'] > multi_leg_df['MinOI']]
            multi_leg_df = multi_leg_df[multi_leg_df['Volume'] > 300]

            # Calculate direction
            conditions = [
                (multi_leg_df['Buy/Sell'] == 'SELL') & (multi_leg_df['CallPut'] == 'CALL'),
                (multi_leg_df['Buy/Sell'] == 'BUY') & (multi_leg_df['CallPut'] == 'CALL'),
                (multi_leg_df['Buy/Sell'] == 'SELL') & (multi_leg_df['CallPut'] == 'PUT'),
                (multi_leg_df['Buy/Sell'] == 'BUY') & (multi_leg_df['CallPut'] == 'PUT')
            ]
            directions = ['BEARISH', 'BULLISH', 'BULLISH', 'BEARISH']
            multi_leg_df['Direction'] = np.select(conditions, directions, default='NEUTRAL')

            # Apply directional conviction filter (70% threshold)
            filtered_symbols = []
            for symbol in multi_leg_df['Symbol'].unique():
                symbol_df = multi_leg_df[multi_leg_df['Symbol'] == symbol].copy()
                direction_premiums = symbol_df.groupby('Direction')['Premium'].sum()
                total_premium = direction_premiums.sum()

                if total_premium != 0:
                    max_direction_premium = direction_premiums.abs().max()
                    if max_direction_premium / abs(total_premium) >= 0.7:
                        filtered_symbols.append(symbol)

            multi_leg_df = multi_leg_df[multi_leg_df['Symbol'].isin(filtered_symbols)]

            # Final premium check by symbol
            symbol_premiums = multi_leg_df.groupby(['Symbol', 'Direction']).agg({
                'Volume': 'sum',
                'Premium': 'sum'
            }).reset_index()

            significant_symbols = symbol_premiums[
                symbol_premiums['Premium'].abs() > 100000
            ]['Symbol'].unique()

            multi_leg_df = multi_leg_df[multi_leg_df['Symbol'].isin(significant_symbols)]

            return multi_leg_df

        except Exception as e:
            logger.error(f"Multi-leg filtering failed: {e}")
            return candidates_df

    def _classify_and_analyze(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify trades and add volatility analysis
        """
        try:
            if trades_df.empty:
                return pd.DataFrame()

            classified_trades = []
            unique_symbols = trades_df['Symbol'].unique()

            # Get volatility analysis for all symbols
            volatility_analysis = self.volatility_calculator.batch_analyze_volatility(unique_symbols)
            volatility_dict = volatility_analysis.set_index('symbol').to_dict('index') if not volatility_analysis.empty else {}

            # Process each trade group
            for (symbol, created_datetime), group in trades_df.groupby(['Symbol', 'CreatedDateTime']):
                try:
                    # Convert group to OptionsFlow objects for classification
                    options_flows = []
                    trade_group_id = str(uuid.uuid4())

                    for _, row in group.iterrows():
                        flow = OptionsFlow(
                            id=str(uuid.uuid4()),
                            created_datetime=row['CreatedDateTime'],
                            symbol=row['Symbol'],
                            buy_sell=row['Buy/Sell'],
                            call_put=row['CallPut'],
                            strike=float(row['Strike']),
                            spot=float(row['Spot']),
                            expiration_date=row['ExpirationDate'].date() if pd.notnull(row['ExpirationDate']) else datetime.now().date(),
                            premium=float(row['Premium']),
                            volume=int(row['Volume']),
                            open_interest=int(row.get('MinOI', 0)),
                            price=float(row.get('Price', 0)),
                            side=str(row.get('Side', '')),
                            color=str(row.get('Color', '')),
                            set_count=int(row.get('SetCount', 0)),
                            implied_volatility=float(row.get('ImpliedVolatility', 0)),
                            dte=int(row.get('Dte', 0)),
                            er_flag=bool(row.get('ER', 'F') == 'T'),
                            trade_value=abs(float(row['Premium']))
                        )
                        options_flows.append(flow)

                    # Classify the trade group
                    classification, expected_outcome, confidence = self._classify_trade_group(options_flows)

                    # Add volatility analysis
                    vol_data = volatility_dict.get(symbol, {})

                    # Calculate moneiness
                    moneiness = self._calculate_moneiness(row['Strike'], row['Spot'], row['CallPut'])

                    # Add enhanced data to each trade in the group
                    for _, row in group.iterrows():
                        enhanced_trade = row.copy()
                        enhanced_trade['trade_group_id'] = trade_group_id
                        enhanced_trade['trade_signature'] = group['Signature'].iloc[0] if 'Signature' in group.columns else ''
                        enhanced_trade['classification'] = classification
                        enhanced_trade['expected_outcome'] = expected_outcome
                        enhanced_trade['confidence_score'] = confidence
                        enhanced_trade['historical_volatility'] = vol_data.get('hv', None)
                        enhanced_trade['implied_volatility_atm'] = vol_data.get('iv', None)
                        enhanced_trade['volatility_flag'] = vol_data.get('flag', 'UNKNOWN')
                        enhanced_trade['volatility_premium'] = vol_data.get('volatility_premium', None)
                        enhanced_trade['moneiness'] = moneiness

                        classified_trades.append(enhanced_trade)

                except Exception as e:
                    logger.error(f"Failed to classify trade group {symbol} {created_datetime}: {e}")
                    continue

            if not classified_trades:
                return pd.DataFrame()

            result_df = pd.DataFrame(classified_trades)
            logger.info(f"Classified {len(result_df)} trades")
            return result_df

        except Exception as e:
            logger.error(f"Classification and analysis failed: {e}")
            return trades_df

    def _calculate_moneiness(self, strike: float, spot: float, call_put: str) -> str:
        """
        Calculate moneiness for options
        """
        try:
            if call_put == 'CALL':
                if strike < spot * 0.95:
                    return 'ITM'
                elif strike > spot * 1.05:
                    return 'OTM'
                else:
                    return 'ATM'
            else:  # PUT
                if strike > spot * 1.05:
                    return 'ITM'
                elif strike < spot * 0.95:
                    return 'OTM'
                else:
                    return 'ATM'
        except:
            return 'UNKNOWN'

    def _analyze_stock_movements(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze stock movements for predictive modeling
        """
        try:
            if trades_df.empty:
                return pd.DataFrame()

            unique_symbols = trades_df['Symbol'].unique()
            movement_data = {}

            for symbol in unique_symbols:
                try:
                    # Get stock data for the last 35 days
                    ticker = yf.Ticker(symbol)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=35)

                    hist = ticker.history(start=start_date, end=end_date)

                    if len(hist) >= 5:  # Need at least 5 days of data
                        current_price = hist['Close'].iloc[-1]

                        # Calculate movements
                        movement_1d = (current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] if len(hist) >= 2 else 0
                        movement_3d = (current_price - hist['Close'].iloc[-4]) / hist['Close'].iloc[-4] if len(hist) >= 4 else 0
                        movement_7d = (current_price - hist['Close'].iloc[-8]) / hist['Close'].iloc[-8] if len(hist) >= 8 else 0
                        movement_30d = (current_price - hist['Close'].iloc[-31]) / hist['Close'].iloc[-31] if len(hist) >= 31 else 0

                        # Determine overall movement direction
                        avg_movement = (movement_1d + movement_3d + movement_7d + movement_30d) / 4
                        if avg_movement > 0.02:
                            movement_direction = 'STRONG_UP'
                        elif avg_movement > 0.005:
                            movement_direction = 'UP'
                        elif avg_movement < -0.02:
                            movement_direction = 'STRONG_DOWN'
                        elif avg_movement < -0.005:
                            movement_direction = 'DOWN'
                        else:
                            movement_direction = 'SIDEWAYS'

                        movement_data[symbol] = {
                            'stock_movement_1d': movement_1d,
                            'stock_movement_3d': movement_3d,
                            'stock_movement_7d': movement_7d,
                            'stock_movement_30d': movement_30d,
                            'movement_direction': movement_direction
                        }

                except Exception as e:
                    logger.error(f"Failed to get movement data for {symbol}: {e}")
                    movement_data[symbol] = {
                        'stock_movement_1d': 0,
                        'stock_movement_3d': 0,
                        'stock_movement_7d': 0,
                        'stock_movement_30d': 0,
                        'movement_direction': 'UNKNOWN'
                    }

            # Add movement data to trades
            for symbol, data in movement_data.items():
                mask = trades_df['Symbol'] == symbol
                for key, value in data.items():
                    trades_df.loc[mask, key] = value

            logger.info(f"Added movement analysis for {len(unique_symbols)} symbols")
            return trades_df

        except Exception as e:
            logger.error(f"Stock movement analysis failed: {e}")
            return trades_df

    def _store_results(self, trades_df: pd.DataFrame) -> int:
        """
        Store results in Supabase database
        """
        try:
            if trades_df.empty:
                return 0

            stored_count = 0

            for _, row in trades_df.iterrows():
                try:
                    # Create OptionsFlow object with all enhanced data
                    flow = OptionsFlow(
                        id=str(uuid.uuid4()),
                        created_datetime=row['CreatedDateTime'],
                        symbol=row['Symbol'],
                        buy_sell=row['Buy/Sell'],
                        call_put=row['CallPut'],
                        strike=float(row['Strike']),
                        spot=float(row['Spot']),
                        expiration_date=row['ExpirationDate'].date() if pd.notnull(row['ExpirationDate']) else datetime.now().date(),
                        premium=float(row['Premium']),
                        volume=int(row['Volume']),
                        open_interest=int(row.get('MinOI', 0)),
                        price=float(row.get('Price', 0)),
                        side=str(row.get('Side', '')),
                        color=str(row.get('Color', '')),
                        set_count=int(row.get('SetCount', 0)),
                        implied_volatility=float(row.get('ImpliedVolatility', 0)),
                        dte=int(row.get('Dte', 0)),
                        er_flag=bool(row.get('ER', 'F') == 'T'),
                        classification=row.get('classification'),
                        expected_outcome=row.get('expected_outcome'),
                        trade_value=abs(float(row['Premium'])),
                        confidence_score=float(row.get('confidence_score', 0))
                    )

                    # Add enhanced fields to the flow object dynamically
                    for field in ['trade_group_id', 'trade_signature', 'historical_volatility',
                                'implied_volatility_atm', 'volatility_flag', 'volatility_premium',
                                'direction', 'moneiness', 'stock_movement_1d', 'stock_movement_3d',
                                'stock_movement_7d', 'stock_movement_30d', 'movement_direction']:
                        if field in row and pd.notnull(row[field]):
                            setattr(flow, field, row[field])

                    if self.db_service.save_options_flow(flow):
                        stored_count += 1

                except Exception as e:
                    logger.error(f"Failed to store individual trade: {e}")
                    continue

            logger.info(f"Stored {stored_count} trades in database")
            return stored_count

        except Exception as e:
            logger.error(f"Failed to store results: {e}")
            return 0

    def _generate_insights(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate predictive insights based on the processed trades
        """
        try:
            if trades_df.empty:
                return {}

            insights = {
                'trade_patterns': {},
                'volatility_insights': {},
                'movement_correlations': {},
                'recommendations': []
            }

            # Trade pattern analysis
            if 'classification' in trades_df.columns:
                classification_counts = trades_df['classification'].value_counts().to_dict()
                insights['trade_patterns'] = classification_counts

            # Volatility insights
            if 'volatility_flag' in trades_df.columns:
                vol_flag_counts = trades_df['volatility_flag'].value_counts().to_dict()
                insights['volatility_insights'] = vol_flag_counts

            # Movement correlations
            if 'movement_direction' in trades_df.columns and 'direction' in trades_df.columns:
                correlation_analysis = trades_df.groupby(['direction', 'movement_direction']).size().to_dict()
                insights['movement_correlations'] = correlation_analysis

            # Generate recommendations
            recommendations = []

            # Volatility-based recommendations
            expensive_count = trades_df[trades_df.get('volatility_flag') == 'EXPENSIVE'].shape[0]
            cheap_count = trades_df[trades_df.get('volatility_flag') == 'CHEAP'].shape[0]

            if expensive_count > cheap_count:
                recommendations.append("Market shows high IV premium - consider selling strategies")
            elif cheap_count > expensive_count:
                recommendations.append("Market shows low IV premium - consider buying strategies")

            # Direction-based recommendations
            if 'direction' in trades_df.columns:
                bullish_count = trades_df[trades_df['direction'] == 'BULLISH'].shape[0]
                bearish_count = trades_df[trades_df['direction'] == 'BEARISH'].shape[0]

                if bullish_count > bearish_count * 1.5:
                    recommendations.append("Strong bullish bias detected in multi-leg flows")
                elif bearish_count > bullish_count * 1.5:
                    recommendations.append("Strong bearish bias detected in multi-leg flows")

            insights['recommendations'] = recommendations

            return insights

        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return {}

    def _generate_summary(self, original_df: pd.DataFrame, final_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate processing summary
        """
        try:
            summary = {
                'processing_date': datetime.now().isoformat(),
                'original_flows': len(original_df),
                'final_trades': len(final_df),
                'screening_efficiency': len(final_df) / len(original_df) * 100 if len(original_df) > 0 else 0,
                'unique_symbols': len(final_df['Symbol'].unique()) if not final_df.empty else 0,
                'total_premium': final_df['Premium'].sum() if not final_df.empty else 0,
                'avg_premium': final_df['Premium'].mean() if not final_df.empty else 0
            }

            if not final_df.empty:
                # Add breakdowns
                if 'direction' in final_df.columns:
                    summary['direction_breakdown'] = final_df['direction'].value_counts().to_dict()

                if 'classification' in final_df.columns:
                    summary['classification_breakdown'] = final_df['classification'].value_counts().to_dict()

                if 'volatility_flag' in final_df.columns:
                    summary['volatility_breakdown'] = final_df['volatility_flag'].value_counts().to_dict()

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return {'error': str(e)}

    def _classify_trade_group(self, options_flows: List[OptionsFlow]) -> Tuple[str, str, float]:
        """
        Simple classification for trade groups.
        """
        try:
            if not options_flows:
                return "UNCLASSIFIED", "No trades provided", 0.0

            # Simple classification based on trade characteristics
            has_calls = any(flow.call_put == 'CALL' for flow in options_flows)
            has_puts = any(flow.call_put == 'PUT' for flow in options_flows)
            has_buys = any(flow.buy_sell == 'BUY' for flow in options_flows)
            has_sells = any(flow.buy_sell == 'SELL' for flow in options_flows)

            # Multi-leg classification
            if has_calls and has_puts and has_buys and has_sells:
                # Check for straddle pattern
                buy_calls = [f for f in options_flows if f.buy_sell == 'BUY' and f.call_put == 'CALL']
                buy_puts = [f for f in options_flows if f.buy_sell == 'BUY' and f.call_put == 'PUT']

                if buy_calls and buy_puts:
                    return "STRADDLE", "Volatility play - expects big move either direction", 0.8
                else:
                    return "COMPLEX_SPREAD", "Multi-leg spread strategy", 0.7

            elif has_calls and has_puts:
                return "CALL_PUT_SPREAD", "Directional spread with calls and puts", 0.7

            elif has_buys and has_sells:
                if has_calls:
                    return "CALL_SPREAD", "Call spread strategy", 0.8
                else:
                    return "PUT_SPREAD", "Put spread strategy", 0.8

            else:
                return "SIMPLE_POSITION", "Single-sided position", 0.6

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "CLASSIFICATION_ERROR", f"Error: {str(e)}", 0.0

    def get_earnings_date(self, symbol: str) -> Optional[datetime]:
        """
        Get earnings date for a symbol using yfinance
        """
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is not None and 'Earnings Date' in cal.index:
                return cal.loc['Earnings Date'][0]
        except Exception as e:
            logger.error(f"Failed to get earnings date for {symbol}: {e}")

        return None

    def predict_trade_outcome(self, trade_data: Dict[str, Any]) -> str:
        """
        Predict trade outcome based on direction, movement, and volatility
        This is a simplified model that can be enhanced over time
        """
        try:
            direction = trade_data.get('direction', 'NEUTRAL')
            movement_direction = trade_data.get('movement_direction', 'UNKNOWN')
            volatility_flag = trade_data.get('volatility_flag', 'UNKNOWN')
            classification = trade_data.get('classification', 'UNCLASSIFIED')

            # Simple rule-based prediction (can be enhanced with ML)
            if direction == 'BULLISH' and movement_direction in ['UP', 'STRONG_UP']:
                if volatility_flag == 'CHEAP':
                    return 'FOREVER PUMPED'
                else:
                    return 'PUMP THEN DISCOUNT'

            elif direction == 'BEARISH' and movement_direction in ['DOWN', 'STRONG_DOWN']:
                if volatility_flag == 'CHEAP':
                    return 'FOREVER DISCOUNTED'
                else:
                    return 'DISCOUNT THEN PUMP'

            elif direction == 'BULLISH' and movement_direction in ['DOWN', 'STRONG_DOWN']:
                return 'DISCOUNT THEN PUMP'

            elif direction == 'BEARISH' and movement_direction in ['UP', 'STRONG_UP']:
                return 'PUMP THEN DISCOUNT'

            else:
                return 'MANUAL REVIEW'

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 'MANUAL REVIEW'
