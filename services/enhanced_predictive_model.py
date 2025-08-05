"""
Enhanced Predictive Model with Stock Movement Analysis.
Incorporates directional bias, volatility analysis, and stock movement patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import yfinance as yf
from services.outcome_tracker import OutcomeTracker
import logging

logger = logging.getLogger(__name__)


class EnhancedPredictiveModel:
    """
    Enhanced predictive model that incorporates:
    - Trade direction vs stock movement correlation
    - Volatility premium analysis
    - Historical pattern recognition
    - Multi-factor outcome prediction
    """

    def __init__(self, outcome_tracker: OutcomeTracker):
        self.outcome_tracker = outcome_tracker
        self.outcome_categories = [
            'FOREVER DISCOUNTED',
            'DISCOUNT THEN PUMP',
            'FOREVER PUMPED',
            'PUMP THEN DISCOUNT',
            'MANUAL REVIEW'
        ]

    def predict_trade_outcome(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced prediction incorporating multiple factors.
        """
        try:
            # Extract key factors
            direction = trade_data.get('direction', 'NEUTRAL')
            movement_direction = trade_data.get('movement_direction', 'UNKNOWN')
            volatility_flag = trade_data.get('volatility_flag', 'UNKNOWN')
            classification = trade_data.get('classification', 'UNCLASSIFIED')
            trade_value = trade_data.get('trade_value', 0)
            er_flag = trade_data.get('er_flag', False)

            # Get historical performance for this classification
            historical_performance = self._get_classification_performance(classification)

            # Calculate base probabilities
            base_probabilities = self._calculate_base_probabilities(
                direction, movement_direction, volatility_flag
            )

            # Apply classification-specific adjustments
            adjusted_probabilities = self._apply_classification_adjustments(
                base_probabilities, classification, historical_performance
            )

            # Apply trade value adjustments
            value_adjusted_probabilities = self._apply_value_adjustments(
                adjusted_probabilities, trade_value
            )

            # Apply earnings adjustments
            if er_flag:
                final_probabilities = self._apply_earnings_adjustments(value_adjusted_probabilities)
            else:
                final_probabilities = value_adjusted_probabilities

            # Determine most likely outcome
            predicted_outcome = max(final_probabilities.items(), key=lambda x: x[1])[0]
            confidence = final_probabilities[predicted_outcome]

            # Generate explanation
            explanation = self._generate_prediction_explanation(
                predicted_outcome, direction, movement_direction, volatility_flag, confidence
            )

            return {
                'predicted_outcome': predicted_outcome,
                'confidence': confidence,
                'probabilities': final_probabilities,
                'explanation': explanation,
                'factors': {
                    'direction': direction,
                    'movement_direction': movement_direction,
                    'volatility_flag': volatility_flag,
                    'classification': classification,
                    'trade_value': trade_value,
                    'earnings_trade': er_flag
                }
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'predicted_outcome': 'MANUAL REVIEW',
                'confidence': 0.5,
                'probabilities': {outcome: 0.2 for outcome in self.outcome_categories},
                'explanation': f"Prediction failed: {str(e)}",
                'factors': trade_data
            }

    def _calculate_base_probabilities(self, direction: str, movement_direction: str,
                                    volatility_flag: str) -> Dict[str, float]:
        """
        Calculate base probabilities based on direction and movement correlation.
        """
        # Initialize with equal probabilities
        probabilities = {outcome: 0.2 for outcome in self.outcome_categories}

        # Direction and movement correlation logic
        if direction == 'BULLISH':
            if movement_direction in ['UP', 'STRONG_UP']:
                # Bullish trade with upward movement
                if volatility_flag == 'CHEAP':
                    probabilities['FOREVER PUMPED'] = 0.45
                    probabilities['PUMP THEN DISCOUNT'] = 0.25
                    probabilities['DISCOUNT THEN PUMP'] = 0.15
                    probabilities['FOREVER DISCOUNTED'] = 0.10
                    probabilities['MANUAL REVIEW'] = 0.05
                else:  # EXPENSIVE
                    probabilities['PUMP THEN DISCOUNT'] = 0.40
                    probabilities['FOREVER PUMPED'] = 0.30
                    probabilities['DISCOUNT THEN PUMP'] = 0.15
                    probabilities['FOREVER DISCOUNTED'] = 0.10
                    probabilities['MANUAL REVIEW'] = 0.05

            elif movement_direction in ['DOWN', 'STRONG_DOWN']:
                # Bullish trade with downward movement (contrarian)
                probabilities['DISCOUNT THEN PUMP'] = 0.40
                probabilities['FOREVER DISCOUNTED'] = 0.25
                probabilities['PUMP THEN DISCOUNT'] = 0.15
                probabilities['FOREVER PUMPED'] = 0.10
                probabilities['MANUAL REVIEW'] = 0.10

            else:  # SIDEWAYS or UNKNOWN
                probabilities['DISCOUNT THEN PUMP'] = 0.30
                probabilities['PUMP THEN DISCOUNT'] = 0.25
                probabilities['FOREVER PUMPED'] = 0.20
                probabilities['FOREVER DISCOUNTED'] = 0.15
                probabilities['MANUAL REVIEW'] = 0.10

        elif direction == 'BEARISH':
            if movement_direction in ['DOWN', 'STRONG_DOWN']:
                # Bearish trade with downward movement
                if volatility_flag == 'CHEAP':
                    probabilities['FOREVER DISCOUNTED'] = 0.45
                    probabilities['DISCOUNT THEN PUMP'] = 0.25
                    probabilities['PUMP THEN DISCOUNT'] = 0.15
                    probabilities['FOREVER PUMPED'] = 0.10
                    probabilities['MANUAL REVIEW'] = 0.05
                else:  # EXPENSIVE
                    probabilities['DISCOUNT THEN PUMP'] = 0.40
                    probabilities['FOREVER DISCOUNTED'] = 0.30
                    probabilities['PUMP THEN DISCOUNT'] = 0.15
                    probabilities['FOREVER PUMPED'] = 0.10
                    probabilities['MANUAL REVIEW'] = 0.05

            elif movement_direction in ['UP', 'STRONG_UP']:
                # Bearish trade with upward movement (contrarian)
                probabilities['PUMP THEN DISCOUNT'] = 0.40
                probabilities['FOREVER PUMPED'] = 0.25
                probabilities['DISCOUNT THEN PUMP'] = 0.15
                probabilities['FOREVER DISCOUNTED'] = 0.10
                probabilities['MANUAL REVIEW'] = 0.10

            else:  # SIDEWAYS or UNKNOWN
                probabilities['PUMP THEN DISCOUNT'] = 0.30
                probabilities['DISCOUNT THEN PUMP'] = 0.25
                probabilities['FOREVER DISCOUNTED'] = 0.20
                probabilities['FOREVER PUMPED'] = 0.15
                probabilities['MANUAL REVIEW'] = 0.10

        else:  # NEUTRAL or unknown direction
            probabilities['MANUAL REVIEW'] = 0.40
            probabilities['DISCOUNT THEN PUMP'] = 0.20
            probabilities['PUMP THEN DISCOUNT'] = 0.20
            probabilities['FOREVER DISCOUNTED'] = 0.10
            probabilities['FOREVER PUMPED'] = 0.10

        return probabilities

    def _apply_classification_adjustments(self, base_probabilities: Dict[str, float],
                                        classification: str,
                                        historical_performance: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply classification-specific adjustments based on historical performance.
        """
        adjusted = base_probabilities.copy()

        # Classification-specific logic
        classification_adjustments = {
            'STRADDLE': {
                'DISCOUNT THEN PUMP': 1.2,
                'PUMP THEN DISCOUNT': 1.2,
                'FOREVER DISCOUNTED': 0.8,
                'FOREVER PUMPED': 0.8
            },
            'NEGATIVE ITM': {
                'FOREVER DISCOUNTED': 1.3,
                'DISCOUNT THEN PUMP': 1.1,
                'FOREVER PUMPED': 0.7,
                'PUMP THEN DISCOUNT': 0.9
            },
            'ATM SAME STRIKE': {
                'DISCOUNT THEN PUMP': 1.2,
                'PUMP THEN DISCOUNT': 1.2,
                'MANUAL REVIEW': 1.1
            },
            'WITHIN RANGE OTMS': {
                'FOREVER PUMPED': 1.1,
                'FOREVER DISCOUNTED': 1.1,
                'DISCOUNT THEN PUMP': 0.9,
                'PUMP THEN DISCOUNT': 0.9
            }
        }

        if classification in classification_adjustments:
            adjustments = classification_adjustments[classification]
            for outcome, multiplier in adjustments.items():
                if outcome in adjusted:
                    adjusted[outcome] *= multiplier

        # Apply historical performance adjustments
        if historical_performance.get('accuracy', 0) > 0.7:
            # High accuracy classification - increase confidence in primary outcomes
            max_outcome = max(adjusted.items(), key=lambda x: x[1])[0]
            adjusted[max_outcome] *= 1.1
            adjusted['MANUAL REVIEW'] *= 0.8
        elif historical_performance.get('accuracy', 0) < 0.4:
            # Low accuracy classification - increase manual review probability
            adjusted['MANUAL REVIEW'] *= 1.3

        # Normalize probabilities
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def _apply_value_adjustments(self, probabilities: Dict[str, float],
                               trade_value: float) -> Dict[str, float]:
        """
        Apply trade value-based adjustments.
        """
        adjusted = probabilities.copy()

        if trade_value > 500000:  # High value trades
            # High value trades tend to be more decisive
            adjusted['MANUAL REVIEW'] *= 0.7
            # Boost the top two outcomes
            sorted_outcomes = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)
            adjusted[sorted_outcomes[0][0]] *= 1.15
            adjusted[sorted_outcomes[1][0]] *= 1.05

        elif trade_value < 100000:  # Low value trades
            # Low value trades are less reliable
            adjusted['MANUAL REVIEW'] *= 1.2
            # Reduce confidence in extreme outcomes
            adjusted['FOREVER PUMPED'] *= 0.9
            adjusted['FOREVER DISCOUNTED'] *= 0.9

        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def _apply_earnings_adjustments(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Apply earnings-specific adjustments.
        """
        adjusted = probabilities.copy()

        # Earnings trades tend to be more volatile
        adjusted['FOREVER PUMPED'] *= 1.2
        adjusted['FOREVER DISCOUNTED'] *= 1.2
        adjusted['DISCOUNT THEN PUMP'] *= 0.9
        adjusted['PUMP THEN DISCOUNT'] *= 0.9
        adjusted['MANUAL REVIEW'] *= 1.1

        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def _get_classification_performance(self, classification: str) -> Dict[str, Any]:
        """
        Get historical performance data for a classification.
        """
        try:
            return self.outcome_tracker.calculate_accuracy_metrics(classification)
        except Exception as e:
            logger.error(f"Failed to get classification performance: {e}")
            return {'accuracy': 0.5, 'total_trades': 0}

    def _generate_prediction_explanation(self, predicted_outcome: str, direction: str,
                                       movement_direction: str, volatility_flag: str,
                                       confidence: float) -> str:
        """
        Generate human-readable explanation for the prediction.
        """
        explanations = []

        # Direction explanation
        if direction == 'BULLISH' and movement_direction in ['UP', 'STRONG_UP']:
            explanations.append("Bullish trade aligns with upward stock movement")
        elif direction == 'BEARISH' and movement_direction in ['DOWN', 'STRONG_DOWN']:
            explanations.append("Bearish trade aligns with downward stock movement")
        elif direction == 'BULLISH' and movement_direction in ['DOWN', 'STRONG_DOWN']:
            explanations.append("Bullish trade contrarian to downward movement - potential reversal play")
        elif direction == 'BEARISH' and movement_direction in ['UP', 'STRONG_UP']:
            explanations.append("Bearish trade contrarian to upward movement - potential reversal play")

        # Volatility explanation
        if volatility_flag == 'EXPENSIVE':
            explanations.append("Options are expensive (IV > HV) - time decay risk")
        elif volatility_flag == 'CHEAP':
            explanations.append("Options are cheap (IV < HV) - favorable entry")

        # Confidence explanation
        if confidence > 0.7:
            explanations.append("High confidence prediction based on strong factor alignment")
        elif confidence < 0.4:
            explanations.append("Low confidence - conflicting signals suggest manual review")

        # Outcome-specific explanation
        outcome_explanations = {
            'FOREVER PUMPED': "Strong bullish momentum expected to continue",
            'FOREVER DISCOUNTED': "Strong bearish momentum expected to continue",
            'PUMP THEN DISCOUNT': "Initial upward move followed by reversal",
            'DISCOUNT THEN PUMP': "Initial downward move followed by reversal",
            'MANUAL REVIEW': "Complex pattern requires manual analysis"
        }

        explanations.append(outcome_explanations.get(predicted_outcome, "Unknown outcome pattern"))

        return ". ".join(explanations) + "."

    def batch_predict_outcomes(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict outcomes for multiple trades.
        """
        try:
            predictions = []

            for _, row in trades_df.iterrows():
                trade_data = row.to_dict()
                prediction = self.predict_trade_outcome(trade_data)

                predictions.append({
                    'trade_id': row.get('id', ''),
                    'symbol': row.get('Symbol', ''),
                    'predicted_outcome': prediction['predicted_outcome'],
                    'confidence': prediction['confidence'],
                    'explanation': prediction['explanation']
                })

            return pd.DataFrame(predictions)

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return pd.DataFrame()

    def analyze_prediction_accuracy(self, symbol: str = None,
                                  classification: str = None) -> Dict[str, Any]:
        """
        Analyze prediction accuracy for specific symbol or classification.
        """
        try:
            filters = {}
            if symbol:
                filters['symbol'] = symbol
            if classification:
                filters['classification'] = classification

            # Get historical data with outcomes
            historical_df = self.outcome_tracker.get_historical_performance(filters)

            if historical_df.empty:
                return {'error': 'No historical data available'}

            # Calculate prediction accuracy
            correct_predictions = 0
            total_predictions = 0

            for _, row in historical_df.iterrows():
                if pd.notnull(row['actual_outcome']):
                    # Re-predict based on historical data
                    trade_data = row.to_dict()
                    prediction = self.predict_trade_outcome(trade_data)

                    if prediction['predicted_outcome'] == row['actual_outcome']:
                        correct_predictions += 1
                    total_predictions += 1

            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

            return {
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions,
                'sample_size': len(historical_df)
            }

        except Exception as e:
            logger.error(f"Accuracy analysis failed: {e}")
            return {'error': str(e)}

    def get_market_sentiment_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Analyze overall market sentiment based on recent trades.
        """
        try:
            sentiment_data = {
                'bullish_signals': 0,
                'bearish_signals': 0,
                'neutral_signals': 0,
                'expensive_options': 0,
                'cheap_options': 0,
                'high_conviction_trades': 0,
                'symbols_analyzed': len(symbols)
            }

            # Get recent trades for analysis
            recent_date = (datetime.now() - timedelta(days=7)).isoformat()
            recent_trades = self.outcome_tracker.get_historical_performance({
                'date_from': recent_date
            })

            if not recent_trades.empty:
                # Analyze sentiment signals
                for _, trade in recent_trades.iterrows():
                    direction = trade.get('direction', 'NEUTRAL')
                    volatility_flag = trade.get('volatility_flag', 'UNKNOWN')
                    trade_value = trade.get('trade_value', 0)

                    # Count directional signals
                    if direction == 'BULLISH':
                        sentiment_data['bullish_signals'] += 1
                    elif direction == 'BEARISH':
                        sentiment_data['bearish_signals'] += 1
                    else:
                        sentiment_data['neutral_signals'] += 1

                    # Count volatility signals
                    if volatility_flag == 'EXPENSIVE':
                        sentiment_data['expensive_options'] += 1
                    elif volatility_flag == 'CHEAP':
                        sentiment_data['cheap_options'] += 1

                    # Count high conviction trades
                    if trade_value > 200000:
                        sentiment_data['high_conviction_trades'] += 1

                # Calculate percentages
                total_trades = len(recent_trades)
                if total_trades > 0:
                    sentiment_data['bullish_percentage'] = sentiment_data['bullish_signals'] / total_trades * 100
                    sentiment_data['bearish_percentage'] = sentiment_data['bearish_signals'] / total_trades * 100
                    sentiment_data['expensive_percentage'] = sentiment_data['expensive_options'] / total_trades * 100
                    sentiment_data['cheap_percentage'] = sentiment_data['cheap_options'] / total_trades * 100

            return sentiment_data

        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            return {'error': str(e)}
