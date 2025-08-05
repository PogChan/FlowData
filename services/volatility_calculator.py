"""
Historical Volatility Calculator using Yang-Zhang method.
Compares HV vs IV for options pricing analysis.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
from services.polygon_api_client import PolygonAPIClient

logger = logging.getLogger(__name__)


class VolatilityCalculator:
    """
    Calculate historical volatility using Yang-Zhang method and compare with implied volatility.
    """

    def __init__(self, polygon_client: PolygonAPIClient):
        self.polygon_client = polygon_client

    def calculate_yang_zhang_hv(self, symbol: str, period_days: int = 30) -> Optional[float]:
        """
        Calculate Yang-Zhang Historical Volatility.

        Yang-Zhang volatility accounts for overnight gaps and intraday price movements.
        Formula: YZ = ln(O/C_prev) + ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)
        """
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days + 10)  # Extra days for calculation

            hist = ticker.history(start=start_date, end=end_date, interval="1d")

            if len(hist) < period_days:
                logger.warning(f"Insufficient data for {symbol}: {len(hist)} days")
                return None

            # Take the most recent period_days
            hist = hist.tail(period_days)

            # Calculate Yang-Zhang components
            hist['Prev_Close'] = hist['Close'].shift(1)

            # Overnight return: ln(Open / Previous Close)
            hist['Overnight'] = np.log(hist['Open'] / hist['Prev_Close'])

            # Open-to-Close return: ln(Close / Open)
            hist['OC'] = np.log(hist['Close'] / hist['Open'])

            # High-to-Close and High-to-Open
            hist['HC'] = np.log(hist['High'] / hist['Close'])
            hist['HO'] = np.log(hist['High'] / hist['Open'])

            # Low-to-Close and Low-to-Open
            hist['LC'] = np.log(hist['Low'] / hist['Close'])
            hist['LO'] = np.log(hist['Low'] / hist['Open'])

            # Yang-Zhang estimator
            hist['YZ'] = (hist['Overnight']**2 +
                         hist['HC'] * hist['HO'] +
                         hist['LC'] * hist['LO'])

            # Remove NaN values
            yz_values = hist['YZ'].dropna()

            if len(yz_values) == 0:
                return None

            # Calculate annualized volatility
            daily_variance = yz_values.mean()
            annualized_volatility = np.sqrt(daily_variance * 252)  # 252 trading days

            return annualized_volatility

        except Exception as e:
            logger.error(f"Failed to calculate Yang-Zhang HV for {symbol}: {e}")
            return None

    def get_next_monthly_expiration(self, symbol: str) -> Optional[datetime]:
        """
        Get the next monthly expiration date, accounting for market holidays.
        Monthly expirations are typically the third Friday of each month.
        """
        try:
            today = datetime.now()

            # Start with current month
            current_month = today.month
            current_year = today.year

            for month_offset in range(3):  # Check next 3 months
                target_month = current_month + month_offset
                target_year = current_year

                if target_month > 12:
                    target_month -= 12
                    target_year += 1

                # Find third Friday of the month
                first_day = datetime(target_year, target_month, 1)
                first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
                third_friday = first_friday + timedelta(days=14)

                # If it's in the future and not today, use it
                if third_friday.date() > today.date():
                    return third_friday

            return None

        except Exception as e:
            logger.error(f"Failed to get next monthly expiration: {e}")
            return None

    def get_atm_iv(self, symbol: str, expiration_date: datetime) -> Optional[float]:
        """
        Get the implied volatility of ATM contracts for the given expiration.
        """
        try:
            # Get current spot price
            spot_price = self.polygon_client.get_current_spot_price(symbol)
            if not spot_price:
                return None

            # Get ATM strike
            atm_strike = self.polygon_client.find_atm_strike(symbol, expiration_date.strftime('%Y-%m-%d'))
            if not atm_strike:
                return None

            # Get options chain data
            chain_data = self.polygon_client.fetch_options_chain(symbol, expiration_date.strftime('%Y-%m-%d'))

            # Find ATM call and put IVs
            call_iv = None
            put_iv = None

            for contract in chain_data.get('calls', []):
                if abs(contract.strike - atm_strike) < 0.01:
                    call_iv = contract.implied_volatility
                    break

            for contract in chain_data.get('puts', []):
                if abs(contract.strike - atm_strike) < 0.01:
                    put_iv = contract.implied_volatility
                    break

            # Return average of call and put IV, or whichever is available
            if call_iv and put_iv:
                return (call_iv + put_iv) / 2
            elif call_iv:
                return call_iv
            elif put_iv:
                return put_iv
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to get ATM IV for {symbol}: {e}")
            return None

    def analyze_volatility_premium(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze volatility premium by comparing HV vs IV.
        Returns analysis with flags for expensive/cheap contracts.
        """
        try:
            # Calculate Yang-Zhang HV
            hv = self.calculate_yang_zhang_hv(symbol)
            if not hv:
                return {
                    'symbol': symbol,
                    'hv': None,
                    'iv': None,
                    'volatility_premium': None,
                    'flag': 'DATA_UNAVAILABLE',
                    'message': 'Unable to calculate historical volatility'
                }

            # Get next monthly expiration
            next_expiration = self.get_next_monthly_expiration(symbol)
            if not next_expiration:
                return {
                    'symbol': symbol,
                    'hv': hv,
                    'iv': None,
                    'volatility_premium': None,
                    'flag': 'NO_EXPIRATION',
                    'message': 'Unable to find next monthly expiration'
                }

            # Get ATM IV for next monthly expiration
            iv = self.get_atm_iv(symbol, next_expiration)
            if not iv:
                return {
                    'symbol': symbol,
                    'hv': hv,
                    'iv': None,
                    'volatility_premium': None,
                    'flag': 'NO_IV_DATA',
                    'message': 'Unable to get implied volatility data'
                }

            # Calculate volatility premium
            volatility_premium = iv - hv
            premium_percentage = (volatility_premium / hv) * 100 if hv > 0 else 0

            # Determine flag
            if iv > hv:
                flag = 'EXPENSIVE'
                message = f'Options are expensive - IV ({iv:.2%}) > HV ({hv:.2%})'
            else:
                flag = 'CHEAP'
                message = f'Options are cheap - IV ({iv:.2%}) < HV ({hv:.2%})'

            return {
                'symbol': symbol,
                'hv': hv,
                'iv': iv,
                'volatility_premium': volatility_premium,
                'premium_percentage': premium_percentage,
                'next_expiration': next_expiration,
                'flag': flag,
                'message': message
            }

        except Exception as e:
            logger.error(f"Failed to analyze volatility premium for {symbol}: {e}")
            return {
                'symbol': symbol,
                'hv': None,
                'iv': None,
                'volatility_premium': None,
                'flag': 'ERROR',
                'message': f'Analysis failed: {str(e)}'
            }

    def batch_analyze_volatility(self, symbols: list) -> pd.DataFrame:
        """
        Analyze volatility premium for multiple symbols.
        """
        results = []

        for symbol in symbols:
            analysis = self.analyze_volatility_premium(symbol)
            results.append(analysis)

        return pd.DataFrame(results)

    def get_volatility_summary(self, symbols: list) -> Dict[str, Any]:
        """
        Get summary statistics for volatility analysis.
        """
        df = self.batch_analyze_volatility(symbols)

        if df.empty:
            return {'total_symbols': 0, 'expensive_count': 0, 'cheap_count': 0}

        expensive_count = len(df[df['flag'] == 'EXPENSIVE'])
        cheap_count = len(df[df['flag'] == 'CHEAP'])

        return {
            'total_symbols': len(df),
            'expensive_count': expensive_count,
            'cheap_count': cheap_count,
            'expensive_percentage': expensive_count / len(df) * 100 if len(df) > 0 else 0,
            'cheap_percentage': cheap_count / len(df) * 100 if len(df) > 0 else 0,
            'avg_hv': df['hv'].mean() if 'hv' in df.columns else None,
            'avg_iv': df['iv'].mean() if 'iv' in df.columns else None,
            'avg_premium': df['volatility_premium'].mean() if 'volatility_premium' in df.columns else None
        }
