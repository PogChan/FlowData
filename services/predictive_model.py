"""
Predictive modeling engine for generating trade outcome insights.
Provides probability calculations and query-based analysis.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.outcome_tracker import OutcomeTracker
import logging

logger = logging.getLogger(__name__)


class PredictiveModel:
    """
    Predictive modeling system for options trade outcome analysis.
    Generates probability-based insights and success rate predictions.
    """

    def __init__(self, outcome_tracker: OutcomeTracker):
        self.outcome_tracker = outcome_tracker
        self.low_value_threshold = 100000  # $100k threshold

    def predict_outcome_probability(self, classification: str, trade_value: Optional[float] = None,
                                  is_earnings: bool = False) -> Dict[str, float]:
        """Predict outcome probabilities with trade value and earnings consideration."""
        try:
            # Get historical data for this classification
            filters = {'classification': classification}
            if is_earnings is not None:
                filters['er_flag'] = is_earnings

            df = self.outcome_tracker.get_historical_performance(filters)

            if df.empty:
                return self._default_probabilities()

            # Calculate base probabilities
            outcome_counts = df['actual_outcome'].value_counts()
            total_trades = len(df)

            probabilities = {}
            for outcome in self.outcome_tracker.valid_outcomes:
                count = outcome_counts.get(outcome, 0)
                probabilities[outcome] = count / total_trades if total_trades > 0 else 0.0

            # Apply trade value adjustments
            if trade_value is not None:
                probabilities = self._adjust_for_trade_value(probabilities, trade_value, df)

            # Apply earnings adjustments
            if is_earnings:
                probabilities = self._adjust_for_earnings(probabilities, df)

            # Add confidence metrics
            probabilities['confidence_score'] = self._calculate_confidence_score(df, trade_value)
            probabilities['sample_size'] = total_trades

            return probabilities

        except Exception as e:
            logger.error(f"Failed to predict outcome probability: {e}")
            return self._default_probabilities()

    def _default_probabilities(self) -> Dict[str, float]:
        """Return default probabilities when no data is available."""
        return {
            'FOREVER DISCOUNTED': 0.25,
            'DISCOUNT THEN PUMP': 0.25,
            'FOREVER PUMPED': 0.25,
            'PUMP THEN DISCOUNT': 0.25,
            'confidence_score': 0.1,
            'sample_size': 0
        }

    def _adjust_for_trade_value(self, probabilities: Dict[str, float], trade_value: float,
                               df: pd.DataFrame) -> Dict[str, float]:
        """Adjust probabilities based on trade value."""
        try:
            if 'trade_value' not in df.columns:
                return probabilities

            # Separate high and low value trades
            low_value_df = df[df['trade_value'] < self.low_value_threshold]
            high_value_df = df[df['trade_value'] >= self.low_value_threshold]

            if trade_value < self.low_value_threshold and not low_value_df.empty:
                # Use low value trade statistics
                low_value_outcomes = low_value_df['actual_outcome'].value_counts()
                total_low_value = len(low_value_df)

                for outcome in self.outcome_tracker.valid_outcomes:
                    count = low_value_outcomes.get(outcome, 0)
                    probabilities[outcome] = count / total_low_value if total_low_value > 0 else probabilities[outcome]

            elif trade_value >= self.low_value_threshold and not high_value_df.empty:
                # Use high value trade statistics
                high_value_outcomes = high_value_df['actual_outcome'].value_counts()
                total_high_value = len(high_value_df)

                for outcome in self.outcome_tracker.valid_outcomes:
                    count = high_value_outcomes.get(outcome, 0)
                    probabilities[outcome] = count / total_high_value if total_high_value > 0 else probabilities[outcome]

            return probabilities

        except Exception as e:
            logger.error(f"Failed to adjust for trade value: {e}")
            return probabilities

    def _adjust_for_earnings(self, probabilities: Dict[str, float], df: pd.DataFrame) -> Dict[str, float]:
        """Adjust probabilities for earnings trades."""
        try:
            # Earnings trades typically have higher volatility
            # Increase probability of extreme outcomes
            earnings_multiplier = 1.2

            probabilities['FOREVER DISCOUNTED'] *= earnings_multiplier
            probabilities['FOREVER PUMPED'] *= earnings_multiplier

            # Normalize to ensure probabilities sum to 1
            total = sum(prob for key, prob in probabilities.items()
                       if key in self.outcome_tracker.valid_outcomes)

            if total > 0:
                for outcome in self.outcome_tracker.valid_outcomes:
                    probabilities[outcome] /= total

            return probabilities

        except Exception as e:
            logger.error(f"Failed to adjust for earnings: {e}")
            return probabilities

    def _calculate_confidence_score(self, df: pd.DataFrame, trade_value: Optional[float] = None) -> float:
        """Calculate confidence score based on sample size and data quality."""
        try:
            sample_size = len(df)

            # Base confidence on sample size
            if sample_size < 10:
                base_confidence = 0.3
            elif sample_size < 50:
                base_confidence = 0.6
            elif sample_size < 100:
                base_confidence = 0.8
            else:
                base_confidence = 0.9

            # Adjust for data recency
            if 'created_datetime' in df.columns:
                recent_cutoff = datetime.now() - timedelta(days=90)
                recent_trades = df[pd.to_datetime(df['created_datetime']) > recent_cutoff]
                recency_factor = len(recent_trades) / sample_size if sample_size > 0 else 0
                base_confidence *= (0.7 + 0.3 * recency_factor)

            return min(0.95, base_confidence)

        except Exception as e:
            logger.error(f"Failed to calculate confidence score: {e}")
            return 0.5

    def analyze_earnings_success_rate(self) -> Dict[str, float]:
        """Analyze earnings trade success rates."""
        try:
            comparison = self.outcome_tracker.get_earnings_vs_regular_performance()

            earnings_stats = comparison.get('earnings_trades', {})
            regular_stats = comparison.get('regular_trades', {})

            return {
                'earnings_accuracy': earnings_stats.get('accuracy', 0.0),
                'regular_accuracy': regular_stats.get('accuracy', 0.0),
                'earnings_total_trades': earnings_stats.get('total_trades', 0),
                'regular_total_trades': regular_stats.get('total_trades', 0),
                'earnings_advantage': earnings_stats.get('accuracy', 0.0) - regular_stats.get('accuracy', 0.0),
                'earnings_avg_value': earnings_stats.get('avg_trade_value', 0.0),
                'regular_avg_value': regular_stats.get('avg_trade_value', 0.0)
            }

        except Exception as e:
            logger.error(f"Failed to analyze earnings success rate: {e}")
            return {}

    def generate_insights(self, query: str) -> Dict[str, Any]:
        """Generate predictive insights based on query with confidence intervals."""
        try:
            query_lower = query.lower()
            insights = {
                'query': query,
                'insights': [],
                'confidence_intervals': {},
                'sample_sizes': {},
                'recommendations': []
            }

            # Parse query for classification patterns
            if 'negative itm' in query_lower:
                insights.update(self._analyze_classification_pattern('NEGATIVE ITM'))
            elif 'straddle' in query_lower:
                insights.update(self._analyze_classification_pattern('STRADDLE'))
            elif 'atm' in query_lower:
                insights.update(self._analyze_classification_pattern('ATM SAME STRIKE'))
            elif 'earnings' in query_lower:
                insights.update(self._analyze_earnings_patterns())
            elif 'low value' in query_lower or 'under 100k' in query_lower:
                insights.update(self._analyze_low_value_trades())
            else:
                # General analysis
                insights.update(self._generate_general_insights())

            return insights

        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return {'query': query, 'insights': ['Error generating insights'], 'error': str(e)}

    def _analyze_classification_pattern(self, classification: str) -> Dict[str, Any]:
        """Analyze specific classification pattern."""
        try:
            df = self.outcome_tracker.get_historical_performance({'classification': classification})

            if df.empty:
                return {
                    'insights': [f'No historical data available for {classification}'],
                    'confidence_intervals': {},
                    'sample_sizes': {classification: 0}
                }

            # Calculate outcome probabilities
            outcome_dist = df['actual_outcome'].value_counts(normalize=True)
            accuracy = df['outcome_match'].mean()

            insights = [
                f'{classification} trades have {accuracy:.1%} accuracy rate',
                f'Most common outcome: {outcome_dist.index[0]} ({outcome_dist.iloc[0]:.1%})',
                f'Based on {len(df)} historical trades'
            ]

            # Add value-based insights
            if 'trade_value' in df.columns:
                avg_value = df['trade_value'].mean()
                high_value_accuracy = df[df['trade_value'] >= self.low_value_threshold]['outcome_match'].mean()
                low_value_accuracy = df[df['trade_value'] < self.low_value_threshold]['outcome_match'].mean()

                insights.extend([
                    f'Average trade value: ${avg_value:,.0f}',
                    f'High value trades (>$100k): {high_value_accuracy:.1%} accuracy',
                    f'Low value trades (<$100k): {low_value_accuracy:.1%} accuracy'
                ])

            # Calculate confidence intervals
            confidence_intervals = {}
            for outcome in outcome_dist.index:
                count = (df['actual_outcome'] == outcome).sum()
                total = len(df)
                ci = self.outcome_tracker._calculate_confidence_interval(count, total)
                confidence_intervals[outcome] = ci

            return {
                'insights': insights,
                'confidence_intervals': confidence_intervals,
                'sample_sizes': {classification: len(df)},
                'accuracy': accuracy,
                'outcome_distribution': outcome_dist.to_dict()
            }

        except Exception as e:
            logger.error(f"Failed to analyze classification pattern: {e}")
            return {'insights': ['Error analyzing pattern'], 'error': str(e)}

    def _analyze_earnings_patterns(self) -> Dict[str, Any]:
        """Analyze earnings-specific patterns."""
        try:
            earnings_analysis = self.analyze_earnings_success_rate()

            insights = [
                f"Earnings trades accuracy: {earnings_analysis.get('earnings_accuracy', 0):.1%}",
                f"Regular trades accuracy: {earnings_analysis.get('regular_accuracy', 0):.1%}",
                f"Earnings advantage: {earnings_analysis.get('earnings_advantage', 0):+.1%}",
                f"Average earnings trade value: ${earnings_analysis.get('earnings_avg_value', 0):,.0f}"
            ]

            # Get earnings trade distribution by classification
            earnings_df = self.outcome_tracker.get_historical_performance({'er_flag': True})
            if not earnings_df.empty:
                class_dist = earnings_df['classification'].value_counts()
                insights.append(f"Most common earnings classification: {class_dist.index[0]} ({class_dist.iloc[0]} trades)")

            return {
                'insights': insights,
                'sample_sizes': {
                    'earnings': earnings_analysis.get('earnings_total_trades', 0),
                    'regular': earnings_analysis.get('regular_total_trades', 0)
                }
            }

        except Exception as e:
            logger.error(f"Failed to analyze earnings patterns: {e}")
            return {'insights': ['Error analyzing earnings patterns'], 'error': str(e)}

    def _analyze_low_value_trades(self) -> Dict[str, Any]:
        """Analyze low value trade patterns."""
        try:
            all_df = self.outcome_tracker.get_historical_performance()

            if all_df.empty or 'trade_value' not in all_df.columns:
                return {'insights': ['No trade value data available']}

            low_value_df = all_df[all_df['trade_value'] < self.low_value_threshold]
            high_value_df = all_df[all_df['trade_value'] >= self.low_value_threshold]

            low_accuracy = low_value_df['outcome_match'].mean() if not low_value_df.empty else 0
            high_accuracy = high_value_df['outcome_match'].mean() if not high_value_df.empty else 0

            insights = [
                f"Low value trades (<$100k): {low_accuracy:.1%} accuracy ({len(low_value_df)} trades)",
                f"High value trades (≥$100k): {high_accuracy:.1%} accuracy ({len(high_value_df)} trades)",
                f"Value threshold impact: {high_accuracy - low_accuracy:+.1%}"
            ]

            # Analyze classification distribution for low value trades
            if not low_value_df.empty:
                low_value_classes = low_value_df['classification'].value_counts()
                insights.append(f"Most common low-value classification: {low_value_classes.index[0]}")

            return {
                'insights': insights,
                'sample_sizes': {
                    'low_value': len(low_value_df),
                    'high_value': len(high_value_df)
                }
            }

        except Exception as e:
            logger.error(f"Failed to analyze low value trades: {e}")
            return {'insights': ['Error analyzing low value trades'], 'error': str(e)}

    def _generate_general_insights(self) -> Dict[str, Any]:
        """Generate general insights across all data."""
        try:
            summary = self.outcome_tracker.get_classification_summary()

            if not summary:
                return {'insights': ['No historical data available for analysis']}

            # Find best and worst performing classifications
            best_class = max(summary.keys(), key=lambda k: summary[k]['accuracy'])
            worst_class = min(summary.keys(), key=lambda k: summary[k]['accuracy'])

            total_trades = sum(data['total_trades'] for data in summary.values())
            overall_accuracy = sum(data['correct_predictions'] for data in summary.values()) / total_trades if total_trades > 0 else 0

            insights = [
                f"Overall system accuracy: {overall_accuracy:.1%} across {total_trades} trades",
                f"Best performing classification: {best_class} ({summary[best_class]['accuracy']:.1%})",
                f"Worst performing classification: {worst_class} ({summary[worst_class]['accuracy']:.1%})",
                f"Total classifications analyzed: {len(summary)}"
            ]

            # Add recommendations
            recommendations = []
            if summary[best_class]['accuracy'] > 0.7:
                recommendations.append(f"Focus on {best_class} trades for higher success rate")
            if summary[worst_class]['accuracy'] < 0.4:
                recommendations.append(f"Review {worst_class} classification rules for improvement")

            return {
                'insights': insights,
                'recommendations': recommendations,
                'sample_sizes': {k: v['total_trades'] for k, v in summary.items()}
            }

        except Exception as e:
            logger.error(f"Failed to generate general insights: {e}")
            return {'insights': ['Error generating general insights'], 'error': str(e)}

    def get_low_value_trade_adjustments(self, trade_value: float) -> Dict[str, float]:
        """Calculate probability adjustments for trades under $100k."""
        try:
            if trade_value >= self.low_value_threshold:
                return {'adjustment_factor': 1.0, 'confidence_adjustment': 0.0}

            # Get historical data for low value trades
            all_df = self.outcome_tracker.get_historical_performance()

            if all_df.empty or 'trade_value' not in all_df.columns:
                return {'adjustment_factor': 0.9, 'confidence_adjustment': -0.1}

            low_value_df = all_df[all_df['trade_value'] < self.low_value_threshold]
            high_value_df = all_df[all_df['trade_value'] >= self.low_value_threshold]

            if low_value_df.empty or high_value_df.empty:
                return {'adjustment_factor': 0.9, 'confidence_adjustment': -0.1}

            low_accuracy = low_value_df['outcome_match'].mean()
            high_accuracy = high_value_df['outcome_match'].mean()

            adjustment_factor = low_accuracy / high_accuracy if high_accuracy > 0 else 0.9
            confidence_adjustment = low_accuracy - high_accuracy

            return {
                'adjustment_factor': adjustment_factor,
                'confidence_adjustment': confidence_adjustment,
                'low_value_accuracy': low_accuracy,
                'high_value_accuracy': high_accuracy
            }

        except Exception as e:
            logger.error(f"Failed to calculate low value adjustments: {e}")
            return {'adjustment_factor': 0.9, 'confidence_adjustment': -0.1}

    def get_confidence_intervals(self, classification: str) -> Dict[str, Tuple[float, float]]:
        """Get confidence intervals for outcome predictions."""
        try:
            df = self.outcome_tracker.get_historical_performance({'classification': classification})

            if df.empty:
                return {}

            confidence_intervals = {}
            outcome_counts = df['actual_outcome'].value_counts()
            total = len(df)

            for outcome in self.outcome_tracker.valid_outcomes:
                count = outcome_counts.get(outcome, 0)
                ci = self.outcome_tracker._calculate_confidence_interval(count, total)
                confidence_intervals[outcome] = ci

            return confidence_intervals

        except Exception as e:
            logger.error(f"Failed to get confidence intervals: {e}")
            return {}
