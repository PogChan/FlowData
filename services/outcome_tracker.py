"""
Outcome tracking system for recording and analyzing trade results.
Provides accuracy metrics and performance analysis capabilities.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from supabase import Client
import logging

logger = logging.getLogger(__name__)


class OutcomeTracker:
    """
    Trade outcome tracking and analysis system.
    Records actual outcomes and calculates classification accuracy metrics.
    """

    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.valid_outcomes = [
            'FOREVER DISCOUNTED',
            'DISCOUNT THEN PUMP',
            'FOREVER PUMPED',
            'PUMP THEN DISCOUNT'
        ]

    def record_outcome(self, trade_id: str, actual_outcome: str) -> bool:
        """Record actual trade outcome."""
        try:
            # Validate outcome
            if actual_outcome.upper() not in self.valid_outcomes:
                logger.error(f"Invalid outcome: {actual_outcome}")
                return False

            # Update the trade record
            result = self.supabase.table('options_flow').update({
                'actual_outcome': actual_outcome.upper(),
                'updated_at': datetime.now().isoformat()
            }).eq('id', trade_id).execute()

            if result.data:
                logger.info(f"Recorded outcome for trade {trade_id}: {actual_outcome}")
                return True
            else:
                logger.error(f"Trade not found: {trade_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
            return False

    def record_bulk_outcomes(self, outcomes: Dict[str, str]) -> Dict[str, bool]:
        """Record multiple outcomes at once."""
        results = {}

        for trade_id, outcome in outcomes.items():
            results[trade_id] = self.record_outcome(trade_id, outcome)

        return results

    def calculate_accuracy_metrics(self, classification: str) -> Dict[str, float]:
        """Calculate accuracy for classification type."""
        try:
            # Get all trades with this classification that have outcomes
            result = self.supabase.table('options_flow').select(
                'id, classification, expected_outcome, actual_outcome, trade_value, created_datetime'
            ).eq(
                'classification', classification
            ).not_.is_(
                'actual_outcome', 'null'
            ).execute()

            if not result.data:
                return {
                    'accuracy': 0.0,
                    'total_trades': 0,
                    'correct_predictions': 0,
                    'confidence_interval': (0.0, 0.0),
                    'sample_size': 0
                }

            trades = result.data
            total = len(trades)
            correct = 0

            # Calculate accuracy based on outcome matching
            for trade in trades:
                expected = trade['expected_outcome']
                actual = trade['actual_outcome']

                if self._outcomes_match(expected, actual):
                    correct += 1

            accuracy = correct / total if total > 0 else 0.0

            # Calculate confidence interval (simple approximation)
            confidence_interval = self._calculate_confidence_interval(correct, total)

            return {
                'accuracy': accuracy,
                'total_trades': total,
                'correct_predictions': correct,
                'confidence_interval': confidence_interval,
                'sample_size': total,
                'error_rate': 1 - accuracy
            }

        except Exception as e:
            logger.error(f"Failed to calculate accuracy: {e}")
            return {
                'accuracy': 0.0,
                'total_trades': 0,
                'correct_predictions': 0,
                'confidence_interval': (0.0, 0.0),
                'sample_size': 0
            }

    def _outcomes_match(self, expected: str, actual: str) -> bool:
        """Determine if expected and actual outcomes match."""
        if not expected or not actual:
            return False

        expected_lower = expected.lower()
        actual_lower = actual.lower()

        # Define matching logic
        pump_keywords = ['pump', 'pumped', 'up', 'gain', 'profit']
        discount_keywords = ['discount', 'discounted', 'down', 'loss', 'drop']

        expected_is_pump = any(keyword in expected_lower for keyword in pump_keywords)
        expected_is_discount = any(keyword in expected_lower for keyword in discount_keywords)

        actual_is_pump = any(keyword in actual_lower for keyword in pump_keywords)
        actual_is_discount = any(keyword in actual_lower for keyword in discount_keywords)

        # Match if both indicate same direction
        if expected_is_pump and actual_is_pump:
            return True
        if expected_is_discount and actual_is_discount:
            return True

        # Handle mixed outcomes (pump then discount, etc.)
        if 'then' in expected_lower or 'then' in actual_lower:
            # More complex matching for sequential outcomes
            return self._match_sequential_outcomes(expected_lower, actual_lower)

        return False

    def _match_sequential_outcomes(self, expected: str, actual: str) -> bool:
        """Match sequential outcomes like 'pump then discount'."""
        # Simplified matching for sequential patterns
        if 'pump then discount' in expected and 'pump then discount' in actual:
            return True
        if 'discount then pump' in expected and 'discount then pump' in actual:
            return True

        return False

    def _calculate_confidence_interval(self, successes: int, total: int, confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for accuracy."""
        if total == 0:
            return (0.0, 0.0)

        import math

        p = successes / total
        z = 1.96  # 95% confidence

        margin = z * math.sqrt((p * (1 - p)) / total)

        lower = max(0.0, p - margin)
        upper = min(1.0, p + margin)

        return (lower, upper)

    def get_historical_performance(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Get historical performance data with optional filters."""
        try:
            query = self.supabase.table('options_flow').select(
                'id, symbol, classification, expected_outcome, actual_outcome, '
                'trade_value, created_datetime, dte, er_flag'
            ).not_.is_('actual_outcome', 'null')

            # Apply filters
            if filters:
                if 'symbol' in filters:
                    query = query.eq('symbol', filters['symbol'])
                if 'classification' in filters:
                    query = query.eq('classification', filters['classification'])
                if 'er_flag' in filters:
                    query = query.eq('er_flag', filters['er_flag'])
                if 'date_from' in filters:
                    query = query.gte('created_datetime', filters['date_from'])
                if 'date_to' in filters:
                    query = query.lte('created_datetime', filters['date_to'])
                if 'min_trade_value' in filters:
                    query = query.gte('trade_value', filters['min_trade_value'])
                if 'max_trade_value' in filters:
                    query = query.lte('trade_value', filters['max_trade_value'])

            result = query.execute()

            if not result.data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(result.data)

            # Add derived columns
            df['outcome_match'] = df.apply(
                lambda row: self._outcomes_match(row['expected_outcome'], row['actual_outcome']),
                axis=1
            )

            df['created_datetime'] = pd.to_datetime(df['created_datetime'])
            df['trade_value'] = pd.to_numeric(df['trade_value'], errors='coerce')

            return df

        except Exception as e:
            logger.error(f"Failed to get historical performance: {e}")
            return pd.DataFrame()

    def get_classification_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for all classifications."""
        try:
            result = self.supabase.table('options_flow').select(
                'classification, expected_outcome, actual_outcome, trade_value'
            ).not_.is_('actual_outcome', 'null').execute()

            if not result.data:
                return {}

            df = pd.DataFrame(result.data)
            summary = {}

            for classification in df['classification'].unique():
                if pd.isna(classification):
                    continue

                class_data = df[df['classification'] == classification]
                metrics = self.calculate_accuracy_metrics(classification)

                summary[classification] = {
                    'accuracy': metrics['accuracy'],
                    'total_trades': metrics['total_trades'],
                    'correct_predictions': metrics['correct_predictions'],
                    'avg_trade_value': class_data['trade_value'].mean() if 'trade_value' in class_data else 0,
                    'confidence_interval': metrics['confidence_interval']
                }

            return summary

        except Exception as e:
            logger.error(f"Failed to get classification summary: {e}")
            return {}

    def get_outcome_distribution(self, classification: Optional[str] = None) -> Dict[str, int]:
        """Get distribution of actual outcomes."""
        try:
            query = self.supabase.table('options_flow').select('actual_outcome').not_.is_('actual_outcome', 'null')

            if classification:
                query = query.eq('classification', classification)

            result = query.execute()

            if not result.data:
                return {}

            outcomes = [row['actual_outcome'] for row in result.data]
            distribution = {}

            for outcome in outcomes:
                distribution[outcome] = distribution.get(outcome, 0) + 1

            return distribution

        except Exception as e:
            logger.error(f"Failed to get outcome distribution: {e}")
            return {}

    def get_performance_by_time_period(self, period: str = 'month') -> pd.DataFrame:
        """Get performance metrics grouped by time period."""
        try:
            df = self.get_historical_performance()

            if df.empty:
                return pd.DataFrame()

            # Group by time period
            if period == 'day':
                df['period'] = df['created_datetime'].dt.date
            elif period == 'week':
                df['period'] = df['created_datetime'].dt.to_period('W')
            elif period == 'month':
                df['period'] = df['created_datetime'].dt.to_period('M')
            else:
                df['period'] = df['created_datetime'].dt.to_period('Y')

            # Calculate metrics by period
            period_stats = df.groupby('period').agg({
                'outcome_match': ['count', 'sum', 'mean'],
                'trade_value': ['mean', 'sum']
            }).round(4)

            period_stats.columns = [
                'total_trades', 'correct_predictions', 'accuracy',
                'avg_trade_value', 'total_trade_value'
            ]

            return period_stats.reset_index()

        except Exception as e:
            logger.error(f"Failed to get performance by time period: {e}")
            return pd.DataFrame()

    def get_earnings_vs_regular_performance(self) -> Dict[str, Dict[str, Any]]:
        """Compare performance between earnings and regular trades."""
        try:
            earnings_df = self.get_historical_performance({'er_flag': True})
            regular_df = self.get_historical_performance({'er_flag': False})

            def calculate_stats(df):
                if df.empty:
                    return {'accuracy': 0.0, 'total_trades': 0, 'avg_trade_value': 0.0}

                return {
                    'accuracy': df['outcome_match'].mean(),
                    'total_trades': len(df),
                    'avg_trade_value': df['trade_value'].mean() if 'trade_value' in df else 0.0,
                    'correct_predictions': df['outcome_match'].sum()
                }

            return {
                'earnings_trades': calculate_stats(earnings_df),
                'regular_trades': calculate_stats(regular_df)
            }

        except Exception as e:
            logger.error(f"Failed to compare earnings vs regular performance: {e}")
            return {'earnings_trades': {}, 'regular_trades': {}}

    def update_trade_outcome(self, trade_id: str, outcome: str, notes: Optional[str] = None) -> bool:
        """Update trade outcome with optional notes."""
        try:
            update_data = {
                'actual_outcome': outcome.upper(),
                'updated_at': datetime.now().isoformat()
            }

            if notes:
                update_data['outcome_notes'] = notes

            result = self.supabase.table('options_flow').update(update_data).eq('id', trade_id).execute()

            return bool(result.data)

        except Exception as e:
            logger.error(f"Failed to update trade outcome: {e}")
            return False

    def get_trades_without_outcomes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get trades that don't have recorded outcomes yet."""
        try:
            result = self.supabase.table('options_flow').select(
                'id, symbol, classification, expected_outcome, created_datetime, trade_value'
            ).is_(
                'actual_outcome', 'null'
            ).order('created_datetime', desc=True).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Failed to get trades without outcomes: {e}")
            return []
