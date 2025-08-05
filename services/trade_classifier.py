"""
Trade classification engine for multi-leg options analysis.
Implements all 11 classification rules with real-time market data.
"""

from typing import List, Tuple, Dict, Any, Optional
from datetime import date
from models.data_models import OptionsFlow
from services.polygon_api_client import PolygonAPIClient
from services.rules_engine import RulesEngine
from utils.config import config
import logging

logger = logging.getLogger(__name__)


class TradeClassifier:
    """
    Multi-leg options trade classifier implementing all 11 classification rules.
    Uses real-time market data for accurate ATM/ITM/OTM determination.
    """

    def __init__(self, polygon_client: PolygonAPIClient, rules_engine: RulesEngine):
        self.polygon_client = polygon_client
        self.rules_engine = rules_engine
        self.delta_threshold = config.app.delta_threshold

    def classify_multi_leg_trade(self, trades: List[OptionsFlow]) -> Tuple[str, str, float]:
        """
        Classify multi-leg options trade and return classification, expected hypothesis, and confidence score.
        """
        if not trades:
            return "UNCLASSIFIED", "No trades provided", 0.0

        try:
            # Group trades by symbol and expiration for analysis
            trade_groups = self._group_trades(trades)

            # For now, handle single symbol trades (can be extended for multi-symbol)
            if len(trade_groups) > 1:
                return "MULTI_SYMBOL", "Multi-symbol trades require manual review", 0.5

            symbol = list(trade_groups.keys())[0]
            symbol_trades = trade_groups[symbol]

            # Get market data for classification
            market_data = self._get_market_data(symbol_trades)
            if not market_data:
                return "DATA_UNAVAILABLE", "Market data unavailable", 0.3

            # Apply classification rules in order of priority
            classification_result = self._apply_classification_rules(symbol_trades, market_data)

            return classification_result

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return "CLASSIFICATION_ERROR", "Error during classification", 0.0

    def _group_trades(self, trades: List[OptionsFlow]) -> Dict[str, List[OptionsFlow]]:
        """Group trades by symbol."""
        groups = {}
        for trade in trades:
            if trade.symbol not in groups:
                groups[trade.symbol] = []
            groups[trade.symbol].append(trade)
        return groups

    def _get_market_data(self, trades: List[OptionsFlow]) -> Optional[Dict[str, Any]]:
        """Get market data needed for classification."""
        if not trades:
            return None

        symbol = trades[0].symbol
        expiration = trades[0].expiration_date

        try:
            # Get current spot price
            spot_price = self.polygon_client.get_current_spot_price(symbol)
            if not spot_price:
                return None

            # Get ATM strike
            atm_strike = self.polygon_client.find_atm_strike(symbol, expiration.isoformat())

            # Get options chain data
            chain_data = self.polygon_client.fetch_options_chain(symbol, expiration.isoformat())

            return {
                'spot_price': spot_price,
                'atm_strike': atm_strike,
                'chain_data': chain_data,
                'symbol': symbol,
                'expiration': expiration
            }

        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None

    def _apply_classification_rules(self, trades: List[OptionsFlow], market_data: Dict[str, Any]) -> Tuple[str, str, float]:
        """Apply classification rules in priority order."""

        # Rule 7: Check for blank sides first
        if self.classify_blank_side(trades):
            return "BLANK SIDE", "Missing side information requires manual review", 0.9

        # Rule 9: Check for straddle
        if self.classify_straddle(trades):
            return "STRADDLE", "Neutral strategy expecting high volatility", 0.95

        # Rule 10: Check for negative ITM
        if self.classify_negative_itm(trades, market_data):
            return "NEGATIVE ITM", "Sell side exceeds buy side - bearish bias", 0.85

        # Rule 11: Check for debit and sell
        if self.classify_debit_and_sell(trades):
            return "DEBIT AND SELL", "Complex spread with opposing positions", 0.8

        # Determine strike relationships for remaining rules
        strike_analysis = self.determine_strike_relationship(trades, market_data)

        # Rules 1-3: Same strike classifications
        if self.classify_atm_same_strike(trades, market_data):
            return "ATM SAME STRIKE", "Both legs at-the-money - high gamma risk", 0.9

        if self.classify_itm_same_strike(trades, market_data):
            return "ITM SAME STRIKE", "Both legs in-the-money - high intrinsic value", 0.9

        if self.classify_otm_same_strike(trades, market_data):
            return "OTM SAME STRIKE", "Both legs out-of-the-money - time decay risk", 0.9

        # Rules 4-6, 8: Range-based classifications
        if self.classify_within_range_otms(trades, market_data):
            return "WITHIN RANGE OTMS", "OTM strikes within delta range - moderate risk", 0.8

        if self.classify_within_range_itms(trades, market_data):
            return "WITHIN RANGE ITMS", "ITM strikes within delta range - conservative play", 0.8

        if self.classify_outside_range_otms(trades, market_data):
            return "OUTSIDE RANGE OTMS", "Wide strike spread - high risk/reward", 0.75

        # Default fallback
        return "UNCLASSIFIED", "Trade pattern not recognized", 0.5

    def classify_blank_side(self, trades: List[OptionsFlow]) -> bool:
        """Check if side values are blank or none."""
        for trade in trades:
            if not trade.side or trade.side.strip() == "":
                return True
        return False

    def classify_straddle(self, trades: List[OptionsFlow]) -> bool:
        """Check if there is both a buy call and buy put."""
        has_buy_call = False
        has_buy_put = False

        for trade in trades:
            if trade.buy_sell.upper() == 'BUY':
                if trade.call_put.upper() == 'CALL':
                    has_buy_call = True
                elif trade.call_put.upper() == 'PUT':
                    has_buy_put = True

        return has_buy_call and has_buy_put

    def classify_negative_itm(self, trades: List[OptionsFlow], market_data: Dict[str, Any]) -> bool:
        """Check if sell side aggregate value exceeds buy side."""
        buy_value = 0
        sell_value = 0
        spot_price = market_data.get('spot_price', 0)

        for trade in trades:
            # Calculate intrinsic value for ITM determination
            if trade.call_put.upper() == 'CALL':
                intrinsic_value = max(0, spot_price - trade.strike)
            else:  # PUT
                intrinsic_value = max(0, trade.strike - spot_price)

            # Only consider ITM trades
            if intrinsic_value > 0:
                trade_value = trade.premium * trade.volume
                if trade.buy_sell.upper() == 'BUY':
                    buy_value += trade_value
                else:
                    sell_value += trade_value

        return sell_value > buy_value and buy_value > 0

    def classify_debit_and_sell(self, trades: List[OptionsFlow]) -> bool:
        """Check if there is a debit spread with opposite sell leg."""
        # Look for debit spread pattern
        buy_trades = [t for t in trades if t.buy_sell.upper() == 'BUY']
        sell_trades = [t for t in trades if t.buy_sell.upper() == 'SELL']

        if len(buy_trades) < 1 or len(sell_trades) < 2:
            return False

        # Check for debit spread (buy higher premium, sell lower premium)
        # and additional sell leg on opposite side
        call_buys = [t for t in buy_trades if t.call_put.upper() == 'CALL']
        put_buys = [t for t in buy_trades if t.call_put.upper() == 'PUT']
        call_sells = [t for t in sell_trades if t.call_put.upper() == 'CALL']
        put_sells = [t for t in sell_trades if t.call_put.upper() == 'PUT']

        # Simple heuristic: if we have buys and sells on both sides
        has_call_activity = len(call_buys) > 0 and len(call_sells) > 0
        has_put_activity = len(put_buys) > 0 and len(put_sells) > 0

        return has_call_activity and has_put_activity

    def classify_atm_same_strike(self, trades: List[OptionsFlow], market_data: Dict[str, Any]) -> bool:
        """Check if both legs have same ATM strike."""
        atm_strike = market_data.get('atm_strike')
        if not atm_strike:
            return False

        # Check if all trades are at ATM strike
        atm_trades = [t for t in trades if abs(t.strike - atm_strike) < 0.01]
        return len(atm_trades) >= 2 and len(atm_trades) == len(trades)

    def classify_itm_same_strike(self, trades: List[OptionsFlow], market_data: Dict[str, Any]) -> bool:
        """Check if both legs have same ITM strike."""
        spot_price = market_data.get('spot_price', 0)

        # Group by strike
        strike_groups = {}
        for trade in trades:
            strike = trade.strike
            if strike not in strike_groups:
                strike_groups[strike] = []
            strike_groups[strike].append(trade)

        # Check if any strike group has multiple trades and all are ITM
        for strike, strike_trades in strike_groups.items():
            if len(strike_trades) >= 2:
                all_itm = True
                for trade in strike_trades:
                    if trade.call_put.upper() == 'CALL':
                        is_itm = strike < spot_price
                    else:  # PUT
                        is_itm = strike > spot_price

                    if not is_itm:
                        all_itm = False
                        break

                if all_itm:
                    return True

        return False

    def classify_otm_same_strike(self, trades: List[OptionsFlow], market_data: Dict[str, Any]) -> bool:
        """Check if both legs have same OTM strike."""
        spot_price = market_data.get('spot_price', 0)

        # Group by strike
        strike_groups = {}
        for trade in trades:
            strike = trade.strike
            if strike not in strike_groups:
                strike_groups[strike] = []
            strike_groups[strike].append(trade)

        # Check if any strike group has multiple trades and all are OTM
        for strike, strike_trades in strike_groups.items():
            if len(strike_trades) >= 2:
                all_otm = True
                for trade in strike_trades:
                    if trade.call_put.upper() == 'CALL':
                        is_otm = strike > spot_price
                    else:  # PUT
                        is_otm = strike < spot_price

                    if not is_otm:
                        all_otm = False
                        break

                if all_otm:
                    return True

        return False

    def classify_within_range_otms(self, trades: List[OptionsFlow], market_data: Dict[str, Any]) -> bool:
        """Check if both legs strikes are within 0.18 delta range of buy side direction."""
        buy_trades = [t for t in trades if t.buy_sell.upper() == 'BUY']
        if not buy_trades:
            return False

        # Determine buy side direction
        buy_side_direction = buy_trades[0].call_put.lower()

        # Get delta range
        symbol = market_data['symbol']
        expiration = market_data['expiration']
        lower_strike, upper_strike = self.polygon_client.calculate_delta_range(
            symbol, expiration.isoformat(), buy_side_direction
        )

        if not lower_strike or not upper_strike:
            return False

        # Check if all strikes are within range and OTM
        spot_price = market_data.get('spot_price', 0)

        for trade in trades:
            # Check if OTM
            if trade.call_put.upper() == 'CALL':
                is_otm = trade.strike > spot_price
            else:  # PUT
                is_otm = trade.strike < spot_price

            if not is_otm:
                return False

            # Check if within delta range
            if not (min(lower_strike, upper_strike) <= trade.strike <= max(lower_strike, upper_strike)):
                return False

        return True

    def classify_outside_range_otms(self, trades: List[OptionsFlow], market_data: Dict[str, Any]) -> bool:
        """Check if either legs strikes are outside 0.18 delta range."""
        buy_trades = [t for t in trades if t.buy_sell.upper() == 'BUY']
        if not buy_trades:
            return False

        # Determine buy side direction
        buy_side_direction = buy_trades[0].call_put.lower()

        # Get delta range
        symbol = market_data['symbol']
        expiration = market_data['expiration']
        lower_strike, upper_strike = self.polygon_client.calculate_delta_range(
            symbol, expiration.isoformat(), buy_side_direction
        )

        if not lower_strike or not upper_strike:
            return False

        # Check if any strike is outside range and OTM
        spot_price = market_data.get('spot_price', 0)
        has_outside_range = False
        all_otm = True

        for trade in trades:
            # Check if OTM
            if trade.call_put.upper() == 'CALL':
                is_otm = trade.strike > spot_price
            else:  # PUT
                is_otm = trade.strike < spot_price

            if not is_otm:
                all_otm = False

            # Check if outside delta range
            if not (min(lower_strike, upper_strike) <= trade.strike <= max(lower_strike, upper_strike)):
                has_outside_range = True

        return has_outside_range and all_otm

    def classify_within_range_itms(self, trades: List[OptionsFlow], market_data: Dict[str, Any]) -> bool:
        """Check if ITM strikes exist on both sides within 0.18 delta range."""
        spot_price = market_data.get('spot_price', 0)

        # Separate calls and puts
        calls = [t for t in trades if t.call_put.upper() == 'CALL']
        puts = [t for t in trades if t.call_put.upper() == 'PUT']

        if not calls or not puts:
            return False

        # Check if we have ITM on both sides
        itm_calls = [c for c in calls if c.strike < spot_price]
        itm_puts = [p for p in puts if p.strike > spot_price]

        if not itm_calls or not itm_puts:
            return False

        # Get delta ranges for both sides
        symbol = market_data['symbol']
        expiration = market_data['expiration']

        call_lower, call_upper = self.polygon_client.calculate_delta_range(
            symbol, expiration.isoformat(), 'call'
        )
        put_lower, put_upper = self.polygon_client.calculate_delta_range(
            symbol, expiration.isoformat(), 'put'
        )

        if not all([call_lower, call_upper, put_lower, put_upper]):
            return False

        # Check if ITM strikes are within their respective ranges
        calls_in_range = all(
            min(call_lower, call_upper) <= c.strike <= max(call_lower, call_upper)
            for c in itm_calls
        )
        puts_in_range = all(
            min(put_lower, put_upper) <= p.strike <= max(put_lower, put_upper)
            for p in itm_puts
        )

        return calls_in_range and puts_in_range

    def determine_strike_relationship(self, trades: List[OptionsFlow], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine ATM/ITM/OTM relationships using current spot price."""
        spot_price = market_data.get('spot_price', 0)
        atm_strike = market_data.get('atm_strike', 0)

        analysis = {
            'spot_price': spot_price,
            'atm_strike': atm_strike,
            'trades_analysis': []
        }

        for trade in trades:
            if trade.call_put.upper() == 'CALL':
                if abs(trade.strike - atm_strike) < 0.01:
                    relationship = 'ATM'
                elif trade.strike < spot_price:
                    relationship = 'ITM'
                else:
                    relationship = 'OTM'
            else:  # PUT
                if abs(trade.strike - atm_strike) < 0.01:
                    relationship = 'ATM'
                elif trade.strike > spot_price:
                    relationship = 'ITM'
                else:
                    relationship = 'OTM'

            analysis['trades_analysis'].append({
                'trade': trade,
                'relationship': relationship,
                'intrinsic_value': max(0,
                    (spot_price - trade.strike) if trade.call_put.upper() == 'CALL'
                    else (trade.strike - spot_price)
                )
            })

        return analysis

    def validate_earnings_flag(self, trade: OptionsFlow) -> bool:
        """Enhanced earnings validation beyond Excel ER flag."""
        # For now, rely on the ER flag from Excel
        # This can be enhanced with external earnings calendar APIs
        return trade.er_flag

    def classify_earnings_trade(self, trade: OptionsFlow) -> bool:
        """Classify trade as earnings-related based on ER flag and additional validation."""
        return self.validate_earnings_flag(trade)

    def get_expected_outcome(self, classification: str, is_earnings: bool) -> str:
        """Get expected hypothesis based on classification and earnings status."""
        # Define expected outcomes for each classification
        outcomes = {
            "ATM SAME STRIKE": "High volatility expected - direction uncertain",
            "ITM SAME STRIKE": "Conservative play - likely to hold value",
            "OTM SAME STRIKE": "High risk/reward - needs significant move",
            "WITHIN RANGE OTMS": "Moderate risk - needs directional move",
            "OUTSIDE RANGE OTMS": "High risk - needs large directional move",
            "BLANK SIDE": "Manual review required",
            "WITHIN RANGE ITMS": "Conservative - likely profitable",
            "STRADDLE": "Volatility play - expects big move either direction",
            "NEGATIVE ITM": "Bearish bias - expects downward pressure",
            "DEBIT AND SELL": "Complex strategy - mixed directional bias",
            "UNCLASSIFIED": "Pattern unclear - manual analysis needed"
        }

        base_outcome = outcomes.get(classification, "Unknown pattern")

        if is_earnings:
            base_outcome += " (Earnings play - higher volatility expected)"

        return base_outcome
