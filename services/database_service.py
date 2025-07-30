"""
Database service implementation for the enhanced options flow classifier.
Provides concrete implementation of the database interface using Supabase.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from supabase import Client
from models.data_models import OptionsFlow, OptionsChainData, ClassificationRule
from services.base import DatabaseServiceInterface
import logging

logger = logging.getLogger(__name__)

class SupabaseService(DatabaseServiceInterface):
    """
    Supabase implementation of the database service interface.
    Handles all database operations for the enhanced options flow classifier.
    """

    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client

    def save_options_flow(self, flow: OptionsFlow) -> bool:
        """Save options flow data to the enhanced options_flow table."""
        try:
            # Convert dataclass to dict for Supabase
            flow_data = {
                'id': flow.id,
                'created_datetime': flow.created_datetime.isoformat() if flow.created_datetime else None,
                'symbol': flow.symbol,
                'buy_sell': flow.buy_sell,
                'call_put': flow.call_put,
                'strike': flow.strike,
                'spot': flow.spot,
                'expiration_date': flow.expiration_date.isoformat() if flow.expiration_date else None,
                'premium': flow.premium,
                'volume': flow.volume,
                'open_interest': flow.open_interest,
                'price': flow.price,
                'side': flow.side,
                'color': flow.color,
                'set_count': flow.set_count,
                'implied_volatility': flow.implied_volatility,
                'dte': flow.dte,
                'er_flag': flow.er_flag,
                'classification': flow.classification,
                'expected_hypothesis': flow.expected_hypothesis,
                'actual_outcome': flow.actual_outcome,
                'trade_value': flow.trade_value,
                'confidence_score': flow.confidence_score
            }

            # Use upsert to handle both insert and update
            result = self.supabase.table('options_flow').upsert(flow_data).execute()
            logger.info(f"Saved options flow: {flow.symbol} {flow.strike}")
            return True

        except Exception as e:
            logger.error(f"Failed to save options flow: {e}")
            return False

    def get_options_flows(self, filters: Optional[Dict[str, Any]] = None) -> List[OptionsFlow]:
        """Retrieve options flow data from database with optional filters."""
        try:
            query = self.supabase.table('options_flow').select('*')

            # Apply filters if provided
            if filters:
                for key, value in filters.items():
                    if key == 'symbol':
                        query = query.eq('symbol', value)
                    elif key == 'classification':
                        query = query.eq('classification', value)
                    elif key == 'er_flag':
                        query = query.eq('er_flag', value)
                    elif key == 'date_from':
                        query = query.gte('created_datetime', value)
                    elif key == 'date_to':
                        query = query.lte('created_datetime', value)

            result = query.execute()

            # Convert to OptionsFlow objects
            flows = []
            for row in result.data:
                flow = OptionsFlow(
                    id=row['id'],
                    created_datetime=datetime.fromisoformat(row['created_datetime']) if row['created_datetime'] else None,
                    symbol=row['symbol'],
                    buy_sell=row['buy_sell'],
                    call_put=row['call_put'],
                    strike=float(row['strike']),
                    spot=float(row['spot']),
                    expiration_date=datetime.fromisoformat(row['expiration_date']).date() if row['expiration_date'] else None,
                    premium=float(row['premium']) if row['premium'] else 0.0,
                    volume=int(row['volume']) if row['volume'] else 0,
                    open_interest=int(row['open_interest']) if row['open_interest'] else 0,
                    price=float(row['price']) if row['price'] else 0.0,
                    side=row['side'],
                    color=row['color'],
                    set_count=int(row['set_count']) if row['set_count'] else 0,
                    implied_volatility=float(row['implied_volatility']) if row['implied_volatility'] else 0.0,
                    dte=int(row['dte']) if row['dte'] else 0,
                    er_flag=bool(row['er_flag']) if row['er_flag'] is not None else False,
                    classification=row['classification'],
                    expected_hypothesis=row['expected_hypothesis'],
                    actual_outcome=row['actual_outcome'],
                    trade_value=float(row['trade_value']) if row['trade_value'] else 0.0,
                    confidence_score=float(row['confidence_score']) if row['confidence_score'] else 0.0,
                    created_at=datetime.fromisoformat(row['created_at']) if row.get('created_at') else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row.get('updated_at') else None
                )
                flows.append(flow)

            logger.info(f"Retrieved {len(flows)} options flows")
            return flows

        except Exception as e:
            logger.error(f"Failed to retrieve options flows: {e}")
            return []

    def save_classification_rule(self, rule: ClassificationRule) -> bool:
        """Save classification rule to database."""
        try:
            rule_data = {
                'rule_id': rule.rule_id,
                'name': rule.name,
                'description': rule.description,
                'classification_logic': rule.classification_logic,
                'expected_hypothesis': rule.expected_hypothesis,
                'result_keywords': rule.result_keywords,
                'is_active': rule.is_active,
                'success_rate': rule.success_rate
            }

            result = self.supabase.table('classification_rules').upsert(rule_data).execute()
            logger.info(f"Saved classification rule: {rule.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to save classification rule: {e}")
            return False

    def get_classification_rules(self, active_only: bool = True) -> List[ClassificationRule]:
        """Retrieve classification rules from database."""
        try:
            query = self.supabase.table('classification_rules').select('*')

            if active_only:
                query = query.eq('is_active', True)

            result = query.execute()

            rules = []
            for row in result.data:
                rule = ClassificationRule(
                    rule_id=row['rule_id'],
                    name=row['name'],
                    description=row['description'],
                    classification_logic=row['classification_logic'],
                    expected_hypothesis=row['expected_hypothesis'],
                    result_keywords=row['result_keywords'],
                    is_active=bool(row['is_active']),
                    success_rate=float(row['success_rate']) if row['success_rate'] else None,
                    created_date=datetime.fromisoformat(row['created_date']) if row.get('created_date') else None,
                    updated_date=datetime.fromisoformat(row['updated_date']) if row.get('updated_date') else None
                )
                rules.append(rule)

            logger.info(f"Retrieved {len(rules)} classification rules")
            return rules

        except Exception as e:
            logger.error(f"Failed to retrieve classification rules: {e}")
            return []

    def save_options_chain_data(self, chain_data: OptionsChainData) -> bool:
        """Save options chain data to cache table."""
        try:
            data = {
                'symbol': chain_data.symbol,
                'expiration_date': chain_data.expiration_date.isoformat() if chain_data.expiration_date else None,
                'strike': chain_data.strike,
                'contract_type': chain_data.contract_type,
                'delta': chain_data.delta,
                'gamma': chain_data.gamma,
                'theta': chain_data.theta,
                'vega': chain_data.vega,
                'implied_volatility': chain_data.implied_volatility,
                'open_interest': chain_data.open_interest,
                'volume': chain_data.volume,
                'last_price': chain_data.last_price,
                'bid': chain_data.bid,
                'ask': chain_data.ask,
                'cached_at': chain_data.cached_at.isoformat() if chain_data.cached_at else datetime.now().isoformat(),
                'expires_at': chain_data.expires_at.isoformat() if chain_data.expires_at else None
            }

            result = self.supabase.table('options_chain_cache').upsert(data).execute()
            logger.debug(f"Cached options data: {chain_data.symbol} {chain_data.strike} {chain_data.contract_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to save options chain data: {e}")
            return False

    def get_cached_options_data(self, symbol: str, expiration_date: str,
                               strike: float, contract_type: str) -> Optional[OptionsChainData]:
        """Retrieve cached options chain data if not expired."""
        try:
            result = self.supabase.table('options_chain_cache').select('*').eq(
                'symbol', symbol
            ).eq(
                'expiration_date', expiration_date
            ).eq(
                'strike', strike
            ).eq(
                'contract_type', contract_type
            ).gt(
                'expires_at', datetime.now().isoformat()
            ).order('cached_at', desc=True).limit(1).execute()

            if result.data:
                row = result.data[0]
                return OptionsChainData(
                    symbol=row['symbol'],
                    expiration_date=datetime.fromisoformat(row['expiration_date']).date(),
                    strike=float(row['strike']),
                    contract_type=row['contract_type'],
                    delta=float(row['delta']) if row['delta'] else None,
                    gamma=float(row['gamma']) if row['gamma'] else None,
                    theta=float(row['theta']) if row['theta'] else None,
                    vega=float(row['vega']) if row['vega'] else None,
                    implied_volatility=float(row['implied_volatility']) if row['implied_volatility'] else None,
                    open_interest=int(row['open_interest']) if row['open_interest'] else None,
                    volume=int(row['volume']) if row['volume'] else None,
                    last_price=float(row['last_price']) if row['last_price'] else None,
                    bid=float(row['bid']) if row['bid'] else None,
                    ask=float(row['ask']) if row['ask'] else None,
                    cached_at=datetime.fromisoformat(row['cached_at']) if row['cached_at'] else None,
                    expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None
                )

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve cached options data: {e}")
            return None

    def clean_expired_cache(self) -> int:
        """Clean expired options chain cache entries."""
        try:
            result = self.supabase.table('options_chain_cache').delete().lt(
                'expires_at', datetime.now().isoformat()
            ).execute()

            deleted_count = len(result.data) if result.data else 0
            logger.info(f"Cleaned {deleted_count} expired cache entries")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to clean expired cache: {e}")
            return 0

    def update_outcome(self, flow_id: str, actual_outcome: str) -> bool:
        """Update the actual outcome for a specific options flow."""
        try:
            result = self.supabase.table('options_flow').update({
                'actual_outcome': actual_outcome,
                'updated_at': datetime.now().isoformat()
            }).eq('id', flow_id).execute()

            logger.info(f"Updated outcome for flow {flow_id}: {actual_outcome}")
            return True

        except Exception as e:
            logger.error(f"Failed to update outcome: {e}")
            return False

    def get_classification_accuracy(self, classification: str) -> Dict[str, float]:
        """Get accuracy metrics for a specific classification."""
        try:
            # Get all flows with this classification that have outcomes
            result = self.supabase.table('options_flow').select(
                'expected_hypothesis, actual_outcome'
            ).eq(
                'classification', classification
            ).not_.is_(
                'actual_outcome', 'null'
            ).execute()

            if not result.data:
                return {'accuracy': 0.0, 'total_trades': 0}

            total = len(result.data)
            correct = 0

            # Simple accuracy calculation - can be enhanced based on business logic
            for row in result.data:
                expected = row['expected_hypothesis']
                actual = row['actual_outcome']

                # Basic matching logic - this can be made more sophisticated
                if expected and actual:
                    if 'pump' in expected.lower() and 'pump' in actual.lower():
                        correct += 1
                    elif 'discount' in expected.lower() and 'discount' in actual.lower():
                        correct += 1
                    elif 'worked' in expected.lower() and ('pump' in actual.lower() or 'worked' in actual.lower()):
                        correct += 1

            accuracy = correct / total if total > 0 else 0.0

            return {
                'accuracy': accuracy,
                'total_trades': total,
                'correct_predictions': correct
            }

        except Exception as e:
            logger.error(f"Failed to calculate accuracy: {e}")
            return {'accuracy': 0.0, 'total_trades': 0}
