"""
Polygon API client for fetching real-time options chain data.
Implements caching, rate limiting, and fallback mechanisms.
"""

import time
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from polygon import RESTClient
from models.data_models import OptionsChainData
from services.database_service import SupabaseService
from utils.config import config
import logging

logger = logging.getLogger(__name__)


class PolygonAPIClient:
    """
    Polygon API client with caching and rate limiting capabilities.
    Integrates with yfinance for spot price data.
    """

    def __init__(self, supabase_service: SupabaseService):
        self.api_key = config.api.polygon_api_key
        self.client = RESTClient(self.api_key)
        self.supabase_service = supabase_service
        self.cache_ttl = timedelta(minutes=config.app.cache_ttl // 60)
        self.rate_limit_delay = config.app.api_rate_limit_delay
        self.last_request_time = 0

    def _rate_limit(self):
        """Implement rate limiting with delay."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _exponential_backoff(self, attempt: int, max_attempts: int = 3) -> bool:
        """Implement exponential backoff for retries."""
        if attempt >= max_attempts:
            return False

        delay = (2 ** attempt) + (time.time() % 1)  # Add jitter
        time.sleep(delay)
        return True

    def get_current_spot_price(self, symbol: str) -> Optional[float]:
        """Get current spot price from yfinance for ATM determination."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])

            # Fallback to info if history fails
            info = ticker.info
            return float(info.get('currentPrice', info.get('regularMarketPrice', 0)))

        except Exception as e:
            logger.error(f"Failed to get spot price for {symbol}: {e}")
            return None

    def fetch_options_chain(self, symbol: str, expiration_date: str) -> Dict[str, List[OptionsChainData]]:
        """
        Fetch and cache options chain data with exponential backoff retry.
        Returns dict with 'calls' and 'puts' keys.
        """
        options_data = {'calls': [], 'puts': []}

        for contract_type in ['call', 'put']:
            attempt = 0
            max_attempts = 3

            while attempt < max_attempts:
                try:
                    self._rate_limit()

                    # Try to get from cache first
                    cached_data = self._get_cached_chain_data(symbol, expiration_date, contract_type)
                    if cached_data:
                        options_data[f"{contract_type}s"].extend(cached_data)
                        break

                    # Fetch from Polygon API
                    response = self.client.list_options_contracts(
                        underlying_ticker=symbol,
                        contract_type=contract_type,
                        expiration_date=expiration_date,
                        limit=1000
                    )

                    if hasattr(response, 'results') and response.results:
                        for contract in response.results:
                            # Get detailed contract data
                            contract_data = self._fetch_contract_details(contract.ticker)
                            if contract_data:
                                options_data[f"{contract_type}s"].append(contract_data)

                    break

                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {symbol} {contract_type}: {e}")
                    attempt += 1
                    if not self._exponential_backoff(attempt, max_attempts):
                        logger.error(f"Max attempts reached for {symbol} {contract_type}")
                        break

        return options_data

    def _fetch_contract_details(self, ticker: str) -> Optional[OptionsChainData]:
        """Fetch detailed contract information including Greeks."""
        try:
            self._rate_limit()

            # Get contract details
            contract_info = self.client.get_option_contract(ticker)

            # Get market data
            market_data = self.client.get_last_quote_option(ticker)

            # Parse ticker for contract info
            parts = ticker.split('_')
            if len(parts) < 2:
                return None

            symbol = parts[0]
            contract_part = parts[1]

            # Extract expiration, type, and strike from contract part
            # Format: YYMMDDCXXXXX or YYMMDDPXXXXX
            if len(contract_part) < 8:
                return None

            exp_str = contract_part[:6]
            contract_type = 'call' if contract_part[6] == 'C' else 'put'
            strike_str = contract_part[7:]

            # Parse expiration date
            exp_date = datetime.strptime(f"20{exp_str}", "%Y%m%d").date()

            # Parse strike price
            strike = float(strike_str) / 1000  # Strike is in thousandths

            # Create OptionsChainData object
            chain_data = OptionsChainData(
                symbol=symbol,
                expiration_date=exp_date,
                strike=strike,
                contract_type=contract_type,
                delta=getattr(contract_info, 'delta', None) if hasattr(contract_info, 'delta') else None,
                gamma=getattr(contract_info, 'gamma', None) if hasattr(contract_info, 'gamma') else None,
                theta=getattr(contract_info, 'theta', None) if hasattr(contract_info, 'theta') else None,
                vega=getattr(contract_info, 'vega', None) if hasattr(contract_info, 'vega') else None,
                implied_volatility=getattr(contract_info, 'implied_volatility', None) if hasattr(contract_info, 'implied_volatility') else None,
                open_interest=getattr(contract_info, 'open_interest', None) if hasattr(contract_info, 'open_interest') else None,
                volume=getattr(contract_info, 'volume', None) if hasattr(contract_info, 'volume') else None,
                last_price=getattr(market_data, 'last', {}).get('price') if hasattr(market_data, 'last') else None,
                bid=getattr(market_data, 'bid', None) if hasattr(market_data, 'bid') else None,
                ask=getattr(market_data, 'ask', None) if hasattr(market_data, 'ask') else None,
                cached_at=datetime.now(),
                expires_at=datetime.now() + self.cache_ttl
            )

            # Cache the data
            self.supabase_service.save_options_chain_data(chain_data)

            return chain_data

        except Exception as e:
            logger.error(f"Failed to fetch contract details for {ticker}: {e}")
            return None

    def _get_cached_chain_data(self, symbol: str, expiration_date: str, contract_type: str) -> List[OptionsChainData]:
        """Get cached options chain data if not expired."""
        try:
            # This would need to be implemented to get all strikes for a symbol/expiration/type
            # For now, return empty list to force fresh fetch
            return []
        except Exception as e:
            logger.error(f"Failed to get cached data: {e}")
            return []

    def get_delta_for_strike(self, symbol: str, strike: float, contract_type: str, expiration_date: str) -> Optional[float]:
        """Get delta value for specific option."""
        try:
            # Check cache first
            cached_data = self.supabase_service.get_cached_options_data(
                symbol, expiration_date, strike, contract_type
            )

            if cached_data and cached_data.delta is not None:
                return cached_data.delta

            # Fetch from API if not cached
            chain_data = self.fetch_options_chain(symbol, expiration_date)
            contract_list = chain_data.get(f"{contract_type}s", [])

            for contract in contract_list:
                if abs(contract.strike - strike) < 0.01:  # Close enough match
                    return contract.delta

            return None

        except Exception as e:
            logger.error(f"Failed to get delta for {symbol} {strike} {contract_type}: {e}")
            return None

    def find_atm_strike(self, symbol: str, expiration_date: str) -> Optional[float]:
        """Find ATM strike closest to current spot price."""
        try:
            spot_price = self.get_current_spot_price(symbol)
            if not spot_price:
                return None

            # Get options chain
            chain_data = self.fetch_options_chain(symbol, expiration_date)

            # Get all available strikes
            all_strikes = set()
            for contract_type in ['calls', 'puts']:
                for contract in chain_data.get(contract_type, []):
                    all_strikes.add(contract.strike)

            if not all_strikes:
                return None

            # Find closest strike to spot price
            closest_strike = min(all_strikes, key=lambda x: abs(x - spot_price))
            return closest_strike

        except Exception as e:
            logger.error(f"Failed to find ATM strike for {symbol}: {e}")
            return None

    def find_delta_strike(self, symbol: str, target_delta: float, contract_type: str, expiration_date: str) -> Optional[float]:
        """Find strike price closest to target delta."""
        try:
            chain_data = self.fetch_options_chain(symbol, expiration_date)
            contract_list = chain_data.get(f"{contract_type}s", [])

            if not contract_list:
                return None

            # Filter contracts with delta values
            contracts_with_delta = [c for c in contract_list if c.delta is not None]

            if not contracts_with_delta:
                return None

            # Find closest delta
            closest_contract = min(
                contracts_with_delta,
                key=lambda x: abs(abs(x.delta) - abs(target_delta))
            )

            return closest_contract.strike

        except Exception as e:
            logger.error(f"Failed to find delta strike for {symbol} {target_delta}: {e}")
            return None

    def calculate_delta_range(self, symbol: str, expiration_date: str, buy_side_direction: str) -> Tuple[Optional[float], Optional[float]]:
        """Calculate 0.18 delta range boundaries for classification."""
        try:
            delta_threshold = config.app.delta_threshold

            if buy_side_direction.lower() == 'call':
                # For calls, positive delta
                lower_strike = self.find_delta_strike(symbol, delta_threshold, 'call', expiration_date)
                upper_strike = self.find_delta_strike(symbol, 0.8, 'call', expiration_date)  # High delta calls
            else:
                # For puts, negative delta (but we work with absolute values)
                lower_strike = self.find_delta_strike(symbol, delta_threshold, 'put', expiration_date)
                upper_strike = self.find_delta_strike(symbol, 0.8, 'put', expiration_date)  # High delta puts

            return lower_strike, upper_strike

        except Exception as e:
            logger.error(f"Failed to calculate delta range for {symbol}: {e}")
            return None, None

    def is_api_available(self) -> bool:
        """Check if Polygon API is available."""
        try:
            self._rate_limit()
            # Simple test call
            response = self.client.get_ticker_details("AAPL")
            return response is not None
        except Exception as e:
            logger.warning(f"Polygon API not available: {e}")
            return False
