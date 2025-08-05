"""
Enhanced Flow Screener for Multi-Leg Options Analysis.
Integrates volatility analysis and sophisticated multi-leg trade detection.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging
from services.volatility_calculator import VolatilityCalculator
from services.polygon_api_client import PolygonAPIClient
from models.data_models import OptionsFlow

logger = logging.getLogger(__name__)


class EnhancedFlowScreener:
    """
    Enhanced flow screener focusing on multi-leg synthetic options trades.
    Includes volatility analysis and sophisticated filtering.
    """

    def __init__(self, polygon_client: PolygonAPIClient, volatility_calculator: VolatilityCalculator):
        self.polygon_client = polygon_client
        self.volatility_calculator = volatility_calculator

    def process_daily_flows(self, flows_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process daily flow data focusing on multi-leg trades.
        """
        try:
            logger.info(f"Processing {len(flows_df)} flows")

            # Prepare the data
            prepared_flows = self._prepare_flow_data(flows_df)

            # Focus only on multi-leg candidates
            multi_leg_candidates = self._identify_multi_leg_candidates(prepared_flows)

            if multi_leg_candidates.empty:
                return {
                    'multi_leg_trades': pd.DataFrame(),
                    'volatility_analysis': pd.DataFrame(),
                    'summary': {'total_flows': len(flows_df), 'multi_leg_count': 0}
                }

            # Apply sophisticated filtering
            filtered_trades = self._apply_multi_leg_filters(multi_leg_candidates)

            # Add volatility analysis
            volatility_analysis = self._add_volatility_analysis(filtered_trades)

            # Final screening and ranking
            final_trades = self._final_screening(filtered_trades, volatility_analysis)

            return {
                'multi_leg_trades': final_trades,
                'volatility_analysis': volatility_analysis,
                'summary': self._generate_summary(flows_df, final_trades)
            }

        except Exception as e:
            logger.error(f"Failed to process daily flows: {e}")
            return {
                'multi_leg_trades': pd.DataFrame(),
                'volatility_analysis': pd.DataFrame(),
                'summary': {'error': str(e)}
            }

    def _prepare_flow_data(self, flows_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and clean flow data for analysis.
        """
        try:
            df = flows_df.copy()

            # Ensure required columns exist
            required_columns = ['Symbol', 'Buy/Sell', 'CallPut', 'Strike', 'Spot',
                              'ExpirationDate', 'Premium', 'Volume', 'OI', 'Price',
                              'Side', 'Color', 'CreatedDateTime']

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                return pd.DataFrame()

            # Data type conversions
            df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], errors='coerce')
            df['CreatedDateTime'] = pd.to_datetime(df['CreatedDateTime'], errors='coerce')
            df['Strike'] = pd.to_numeric(df['Strike'], errors='coerce')
            df['Spot'] = pd.to_numeric(df['Spot'], errors='coerce')
            df['Premium'] = pd.to_numeric(df['Premium'], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df['OI'] = pd.to_numeric(df['OI'], errors='coerce')

            # Clean up buy/sell and call/put
            df['Buy/Sell'] = df['Buy/Sell'].str.upper()
            df['CallPut'] = df['CallPut'].str.upper()
            df['Color'] = df['Color'].str.upper()

            # Remove invalid data
            df = df.dropna(subset=['Symbol', 'Strike', 'Premium', 'Volume'])
            df = df[df['Volume'] > 0]
            df = df[df['Premium'] > 0]

            logger.info(f"Prepared {len(df)} valid flows")
            return df

        except Exception as e:
            logger.error(f"Failed to prepare flow data: {e}")
            return pd.DataFrame()

    def _identify_multi_leg_candidates(self, flows_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify multi-leg trade candidates based on timestamp grouping.
        """
        try:
            # Filter to true multi-legs (more than one trade per timestamp)
            multi_leg_candidates = (
                flows_df
                .groupby(['Symbol', 'CreatedDateTime'])
                .filter(lambda g: len(g) > 1)
                .copy()
            )

            if multi_leg_candidates.empty:
                return pd.DataFrame()

            # Make sell premiums negative for net calculation
            multi_leg_candidates['Premium'] = multi_leg_candidates.apply(
                lambda r: -r['Premium'] if r['Buy/Sell'] == 'SELL' else r['Premium'],
                axis=1
            )

            # Build signature for each timestamped group
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

            # Aggregate within each unique leg
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
                    ER=('ER', 'first') if 'ER' in multi_leg_candidates.columns else ('Symbol', lambda x: 'F')
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

            logger.info(f"Identified {len(agg)} multi-leg candidates")
            return agg

        except Exception as e:
            logger.error(f"Failed to identify multi-leg candidates: {e}")
            return pd.DataFrame()

    def _apply_multi_leg_filters(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sophisticated filters to identify valid multi-leg trades.
        """
        try:
            if candidates_df.empty:
                return pd.DataFrame()

            # Filter out straddles/strangles
            filtered_df = candidates_df.groupby(['Symbol', 'CreatedDateTime']).filter(
                self._filter_out_straddles_strangles
            )

            # Apply multi-leg validation
            multi_leg_df = filtered_df.groupby(['Symbol', 'CreatedDateTime']).filter(
                self._is_valid_multi_leg
            )

            if multi_leg_df.empty:
                return pd.DataFrame()

            # Additional volume and OI filters
            multi_leg_df = multi_leg_df[multi_leg_df['Volume'] > multi_leg_df['MinOI']]
            multi_leg_df = multi_leg_df[multi_leg_df['Volume'] > 300]

            # Group by symbol and direction for final filtering
            multi_leg_df = self._apply_directional_filters(multi_leg_df)

            logger.info(f"Filtered to {len(multi_leg_df)} valid multi-leg trades")
            return multi_leg_df

        except Exception as e:
            logger.error(f"Failed to apply multi-leg filters: {e}")
            return pd.DataFrame()

    def _filter_out_straddles_strangles(self, group: pd.DataFrame) -> bool:
        """
        Filter out straddles and strangles based on premium contribution.
        """
        try:
            # Check if there is both a BUY CALL and BUY PUT
            buy_call = (group['Buy/Sell'] == 'BUY') & (group['CallPut'] == 'CALL')
            buy_put = (group['Buy/Sell'] == 'BUY') & (group['CallPut'] == 'PUT')

            # Only proceed if both BUY CALL and BUY PUT exist
            if buy_call.any() and buy_put.any():
                # Calculate total premiums for BUY CALL and BUY PUT
                buy_call_premium = group.loc[buy_call, 'Premium'].sum()
                buy_put_premium = group.loc[buy_put, 'Premium'].sum()

                # Calculate the total premium for both BUY CALL and BUY PUT combined
                total_premium = buy_call_premium + buy_put_premium

                if total_premium > 0:
                    # Check if both sides contribute within a similar range (40% - 60%)
                    call_contribution = buy_call_premium / total_premium
                    put_contribution = buy_put_premium / total_premium

                    # If both sides contribute within a similar range, filter out (return False)
                    if 0.4 <= call_contribution <= 0.6 and 0.4 <= put_contribution <= 0.6:
                        return False

            return True

        except Exception as e:
            logger.error(f"Error in straddle filter: {e}")
            return True

    def _is_valid_multi_leg(self, group: pd.DataFrame) -> bool:
        """
        Determine if a group represents a valid multi-leg trade.
        """
        try:
            # Ensure there is at least one BUY and one SELL
            has_buy = (group['Buy/Sell'] == 'BUY').any()
            has_sell = (group['Buy/Sell'] == 'SELL').any()

            # Ensure there is at least one CALL and one PUT
            call_put_check = {'CALL', 'PUT'}.issubset(group['CallPut'].unique())

            # Check for no 'WHITE' color trades
            white_count = (group['Color'] == 'WHITE').sum()

            if has_buy and has_sell and call_put_check and white_count == 0:
                # Volume should be greater than OI for all legs
                if not (group['Volume'] > group['MinOI']).all():
                    return False

                # Calculate net premium
                total_buy_premium = group[group['Buy/Sell'] == 'BUY']['Premium'].sum()
                total_sell_premium = group[group['Buy/Sell'] == 'SELL']['Premium'].sum()
                net_premium_spent = total_buy_premium + total_sell_premium

                # Additional validation can be added here
                return True

            return False

        except Exception as e:
            logger.error(f"Error in multi-leg validation: {e}")
            return False

    def _apply_directional_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply directional bias filters to ensure conviction trades.
        """
        try:
            # Calculate direction for each trade
            conditions = [
                (df['Buy/Sell'] == 'SELL') & (df['CallPut'] == 'CALL'),
                (df['Buy/Sell'] == 'BUY') & (df['CallPut'] == 'CALL'),
                (df['Buy/Sell'] == 'SELL') & (df['CallPut'] == 'PUT'),
                (df['Buy/Sell'] == 'BUY') & (df['CallPut'] == 'PUT')
            ]
            directions = ['BEARISH', 'BULLISH', 'BULLISH', 'BEARISH']
            df['Direction'] = np.select(conditions, directions, default='NEUTRAL')

            # Group by symbol and check for directional conviction
            filtered_symbols = []

            for symbol in df['Symbol'].unique():
                symbol_df = df[df['Symbol'] == symbol].copy()

                # Calculate total premium by direction
                direction_premiums = symbol_df.groupby('Direction')['Premium'].sum()
                total_premium = direction_premiums.sum()

                if total_premium > 0:
                    # Check if one direction dominates (70% threshold)
                    max_direction_premium = direction_premiums.abs().max()
                    if max_direction_premium / abs(total_premium) >= 0.7:
                        filtered_symbols.append(symbol)

            # Filter to only symbols with strong directional bias
            df = df[df['Symbol'].isin(filtered_symbols)]

            # Aggregate by symbol and direction for final premium check
            symbol_premiums = df.groupby(['Symbol', 'Direction']).agg({
                'Volume': 'sum',
                'Premium': 'sum'
            }).reset_index()

            # Keep only symbols with significant total premium
            significant_symbols = symbol_premiums[
                symbol_premiums['Premium'].abs() > 100000
            ]['Symbol'].unique()

            df = df[df['Symbol'].isin(significant_symbols)]

            return df

        except Exception as e:
            logger.error(f"Failed to apply directional filters: {e}")
            return df

    def _add_volatility_analysis(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility analysis for each symbol in the trades.
        """
        try:
            if trades_df.empty:
                return pd.DataFrame()

            unique_symbols = trades_df['Symbol'].unique()
            volatility_analysis = self.volatility_calculator.batch_analyze_volatility(unique_symbols)

            return volatility_analysis

        except Exception as e:
            logger.error(f"Failed to add volatility analysis: {e}")
            return pd.DataFrame()

    def _calculate_moneiness(self, row: pd.Series) -> str:
        """
        Calculate moneiness for options.
        """
        try:
            spot = row['Spot']
            strike = row['Strike']
            call_put = row['CallPut']

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

        except Exception as e:
            logger.error(f"Failed to calculate moneiness: {e}")
            return 'UNKNOWN'

    def _final_screening(self, trades_df: pd.DataFrame, volatility_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply final screening and ranking to trades.
        """
        try:
            if trades_df.empty:
                return pd.DataFrame()

            # Add moneiness calculation
            trades_df['Moneiness'] = trades_df.apply(self._calculate_moneiness, axis=1)

            # Merge with volatility analysis
            if not volatility_df.empty:
                trades_df = trades_df.merge(
                    volatility_df[['symbol', 'flag', 'hv', 'iv', 'volatility_premium']],
                    left_on='Symbol',
                    right_on='symbol',
                    how='left'
                )
                trades_df = trades_df.drop('symbol', axis=1)

            # Sort by premium (absolute value) descending
            trades_df['AbsPremium'] = trades_df['Premium'].abs()
            trades_df = trades_df.sort_values('AbsPremium', ascending=False)

            # Add ranking
            trades_df['Rank'] = range(1, len(trades_df) + 1)

            # Reorder columns for better display
            display_columns = [
                'Rank', 'Symbol', 'Buy/Sell', 'CallPut', 'Strike', 'Spot',
                'ExpirationDate', 'Moneiness', 'Volume', 'Premium', 'Direction'
            ]

            # Add volatility columns if available
            if 'flag' in trades_df.columns:
                display_columns.extend(['flag', 'hv', 'iv', 'volatility_premium'])

            # Add remaining columns
            remaining_columns = [col for col in trades_df.columns if col not in display_columns]
            final_columns = display_columns + remaining_columns

            # Filter to existing columns
            final_columns = [col for col in final_columns if col in trades_df.columns]

            trades_df = trades_df[final_columns]

            return trades_df

        except Exception as e:
            logger.error(f"Failed in final screening: {e}")
            return trades_df

    def _generate_summary(self, original_df: pd.DataFrame, final_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the screening process.
        """
        try:
            summary = {
                'total_flows': len(original_df),
                'multi_leg_count': len(final_df),
                'screening_efficiency': len(final_df) / len(original_df) * 100 if len(original_df) > 0 else 0,
                'unique_symbols': len(final_df['Symbol'].unique()) if not final_df.empty else 0,
                'total_premium': final_df['Premium'].sum() if not final_df.empty else 0,
                'avg_premium': final_df['Premium'].mean() if not final_df.empty else 0
            }

            if not final_df.empty:
                # Direction breakdown
                direction_counts = final_df['Direction'].value_counts().to_dict()
                summary['direction_breakdown'] = direction_counts

                # Moneiness breakdown
                if 'Moneiness' in final_df.columns:
                    moneiness_counts = final_df['Moneiness'].value_counts().to_dict()
                    summary['moneiness_breakdown'] = moneiness_counts

                # Volatility breakdown
                if 'flag' in final_df.columns:
                    volatility_counts = final_df['flag'].value_counts().to_dict()
                    summary['volatility_breakdown'] = volatility_counts

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return {'error': str(e)}

    def get_earnings_date(self, symbol: str) -> Optional[datetime]:
        """
        Get earnings date for a symbol using yfinance.
        """
        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar
            if cal is not None and 'Earnings Date' in cal.index:
                return cal.loc['Earnings Date'][0]
        except Exception as e:
            logger.error(f"Failed to get earnings date for {symbol}: {e}")

        return None

    def calculate_put_call_ratio(self, symbol: str, expiration_date: datetime) -> Optional[float]:
        """
        Calculate put/call ratio for a symbol and expiration.
        """
        try:
            chain_data = self.polygon_client.fetch_options_chain(symbol, expiration_date.strftime('%Y-%m-%d'))

            if not chain_data:
                return None

            call_volume = sum(contract.volume or 0 for contract in chain_data.get('calls', []))
            put_volume = sum(contract.volume or 0 for contract in chain_data.get('puts', []))

            if call_volume > 0:
                return put_volume / call_volume

            return None

        except Exception as e:
            logger.error(f"Failed to calculate P/C ratio for {symbol}: {e}")
            return None
