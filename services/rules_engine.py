"""
Dynamic rules engine for managing classification rules.
Provides CRUD operations and rule evaluation capabilities.
"""

import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
from supabase import Client
from models.data_models import ClassificationRule
import logging

logger = logging.getLogger(__name__)


class RulesEngine:
    """
    Dynamic rules engine for managing and evaluating classification rules.
    Supports rule versioning, conflict detection, and effectiveness tracking.
    """

    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.rules_cache = {}
        self._load_rules()

    def _load_rules(self) -> None:
        """Load all active rules into cache."""
        try:
            result = self.supabase.table('classification_rules').select('*').eq('is_active', True).execute()

            self.rules_cache = {}
            for row in result.data:
                rule = ClassificationRule(
                    rule_id=row['rule_id'],
                    name=row['name'],
                    description=row['description'],
                    classification_logic=row['classification_logic'],
                    expected_outcome=row['expected_outcome'],
                    result_keywords=row['result_keywords'],
                    is_active=bool(row['is_active']),
                    success_rate=float(row['success_rate']) if row['success_rate'] else None,
                    created_date=datetime.fromisoformat(row['created_date']) if row.get('created_date') else None,
                    updated_date=datetime.fromisoformat(row['updated_date']) if row.get('updated_date') else None
                )
                self.rules_cache[rule.name] = rule

            logger.info(f"Loaded {len(self.rules_cache)} classification rules")

        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            self.rules_cache = {}

    def load_rules(self) -> Dict[str, ClassificationRule]:
        """Load classification rules from database."""
        self._load_rules()
        return self.rules_cache.copy()

    def add_rule(self, rule: ClassificationRule) -> bool:
        """Add new classification rule."""
        try:
            # Check for conflicts
            conflicts = self.detect_rule_conflicts(rule)
            if conflicts:
                logger.warning(f"Rule conflicts detected: {conflicts}")
                return False

            # Generate ID if not provided
            if not rule.rule_id:
                rule.rule_id = str(uuid.uuid4())

            # Set timestamps
            rule.created_date = datetime.now()
            rule.updated_date = datetime.now()

            # Save to database
            rule_data = {
                'rule_id': rule.rule_id,
                'name': rule.name,
                'description': rule.description,
                'classification_logic': rule.classification_logic,
                'expected_outcome': rule.expected_outcome,
                'result_keywords': rule.result_keywords,
                'is_active': rule.is_active,
                'success_rate': rule.success_rate,
                'created_date': rule.created_date.isoformat(),
                'updated_date': rule.updated_date.isoformat()
            }

            result = self.supabase.table('classification_rules').insert(rule_data).execute()

            # Update cache
            self.rules_cache[rule.name] = rule

            logger.info(f"Added new rule: {rule.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add rule: {e}")
            return False

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing rule with version tracking."""
        try:
            # Find existing rule
            existing_rule = None
            for rule in self.rules_cache.values():
                if rule.rule_id == rule_id:
                    existing_rule = rule
                    break

            if not existing_rule:
                logger.error(f"Rule not found: {rule_id}")
                return False

            # Create updated rule data
            update_data = updates.copy()
            update_data['updated_date'] = datetime.now().isoformat()

            # Update in database
            result = self.supabase.table('classification_rules').update(update_data).eq('rule_id', rule_id).execute()

            # Update cache
            for key, value in updates.items():
                if hasattr(existing_rule, key):
                    setattr(existing_rule, key, value)
            existing_rule.updated_date = datetime.now()

            logger.info(f"Updated rule: {existing_rule.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to update rule: {e}")
            return False

    def evaluate_rule(self, rule_name: str, trade_data: Dict[str, Any]) -> bool:
        """Evaluate if trade matches rule criteria."""
        try:
            rule = self.rules_cache.get(rule_name)
            if not rule or not rule.is_active:
                return False

            logic = rule.classification_logic

            # Simple rule evaluation based on logic structure
            # This can be made more sophisticated with a proper rule engine
            return self._evaluate_logic(logic, trade_data)

        except Exception as e:
            logger.error(f"Failed to evaluate rule {rule_name}: {e}")
            return False

    def _evaluate_logic(self, logic: Dict[str, Any], trade_data: Dict[str, Any]) -> bool:
        """Evaluate rule logic against trade data."""
        try:
            # Simple evaluation - can be enhanced with more complex logic
            for condition_key, condition_value in logic.items():
                if condition_key in trade_data:
                    trade_value = trade_data[condition_key]

                    if isinstance(condition_value, dict):
                        # Handle operators like {">=": 100}
                        for operator, expected_value in condition_value.items():
                            if operator == ">=":
                                if not (trade_value >= expected_value):
                                    return False
                            elif operator == "<=":
                                if not (trade_value <= expected_value):
                                    return False
                            elif operator == "==":
                                if not (trade_value == expected_value):
                                    return False
                            elif operator == "!=":
                                if not (trade_value != expected_value):
                                    return False
                            elif operator == "in":
                                if not (trade_value in expected_value):
                                    return False
                    else:
                        # Direct value comparison
                        if trade_value != condition_value:
                            return False

            return True

        except Exception as e:
            logger.error(f"Logic evaluation failed: {e}")
            return False

    def reclassify_trades_with_new_rules(self, rule_id: str) -> int:
        """Re-classify existing trades when rules are updated."""
        try:
            # This would trigger a re-classification of all trades
            # For now, return 0 as this would be implemented with the main classifier
            logger.info(f"Re-classification triggered for rule: {rule_id}")
            return 0

        except Exception as e:
            logger.error(f"Failed to reclassify trades: {e}")
            return 0

    def detect_rule_conflicts(self, new_rule: ClassificationRule) -> List[str]:
        """Detect conflicts with existing rules."""
        conflicts = []

        try:
            # Check for name conflicts
            if new_rule.name in self.rules_cache:
                conflicts.append(f"Rule name '{new_rule.name}' already exists")

            # Check for logic conflicts (simplified)
            for existing_name, existing_rule in self.rules_cache.items():
                if existing_rule.classification_logic == new_rule.classification_logic:
                    conflicts.append(f"Similar logic exists in rule '{existing_name}'")

            return conflicts

        except Exception as e:
            logger.error(f"Failed to detect conflicts: {e}")
            return ["Error detecting conflicts"]

    def resolve_rule_conflicts(self, conflicts: List[str], resolution: str) -> bool:
        """Resolve rule conflicts based on user choice."""
        try:
            # Implementation depends on resolution strategy
            # For now, just log the resolution
            logger.info(f"Resolving conflicts: {conflicts} with strategy: {resolution}")
            return True

        except Exception as e:
            logger.error(f"Failed to resolve conflicts: {e}")
            return False

    def get_rule_effectiveness_metrics(self, rule_id: str) -> Dict[str, float]:
        """Get effectiveness metrics for a specific rule."""
        try:
            # Query trades classified by this rule
            result = self.supabase.table('options_flow').select(
                'classification, expected_outcome, actual_outcome'
            ).not_.is_('actual_outcome', 'null').execute()

            if not result.data:
                return {'success_rate': 0.0, 'total_trades': 0, 'correct_predictions': 0}

            # Find the rule to get its classification name
            rule = None
            for r in self.rules_cache.values():
                if r.rule_id == rule_id:
                    rule = r
                    break

            if not rule:
                return {'success_rate': 0.0, 'total_trades': 0, 'correct_predictions': 0}

            # Filter trades for this rule's classification
            rule_trades = [
                row for row in result.data
                if row['classification'] == rule.name
            ]

            if not rule_trades:
                return {'success_rate': 0.0, 'total_trades': 0, 'correct_predictions': 0}

            total = len(rule_trades)
            correct = 0

            # Calculate success rate based on outcome matching
            for trade in rule_trades:
                expected = trade['expected_outcome']
                actual = trade['actual_outcome']

                # Simple matching logic
                if expected and actual:
                    if any(keyword.lower() in actual.lower() for keyword in rule.result_keywords):
                        correct += 1

            success_rate = correct / total if total > 0 else 0.0

            # Update rule success rate
            self.update_rule(rule_id, {'success_rate': success_rate})

            return {
                'success_rate': success_rate,
                'total_trades': total,
                'correct_predictions': correct
            }

        except Exception as e:
            logger.error(f"Failed to get effectiveness metrics: {e}")
            return {'success_rate': 0.0, 'total_trades': 0, 'correct_predictions': 0}

    def get_all_rules(self) -> List[ClassificationRule]:
        """Get all rules (active and inactive)."""
        try:
            result = self.supabase.table('classification_rules').select('*').execute()

            rules = []
            for row in result.data:
                rule = ClassificationRule(
                    rule_id=row['rule_id'],
                    name=row['name'],
                    description=row['description'],
                    classification_logic=row['classification_logic'],
                    expected_outcome=row['expected_outcome'],
                    result_keywords=row['result_keywords'],
                    is_active=bool(row['is_active']),
                    success_rate=float(row['success_rate']) if row['success_rate'] else None,
                    created_date=datetime.fromisoformat(row['created_date']) if row.get('created_date') else None,
                    updated_date=datetime.fromisoformat(row['updated_date']) if row.get('updated_date') else None
                )
                rules.append(rule)

            return rules

        except Exception as e:
            logger.error(f"Failed to get all rules: {e}")
            return []

    def deactivate_rule(self, rule_id: str) -> bool:
        """Deactivate a rule without deleting it."""
        return self.update_rule(rule_id, {'is_active': False})

    def activate_rule(self, rule_id: str) -> bool:
        """Activate a previously deactivated rule."""
        return self.update_rule(rule_id, {'is_active': True})

    def delete_rule(self, rule_id: str) -> bool:
        """Permanently delete a rule."""
        try:
            result = self.supabase.table('classification_rules').delete().eq('rule_id', rule_id).execute()

            # Remove from cache
            rule_to_remove = None
            for name, rule in self.rules_cache.items():
                if rule.rule_id == rule_id:
                    rule_to_remove = name
                    break

            if rule_to_remove:
                del self.rules_cache[rule_to_remove]

            logger.info(f"Deleted rule: {rule_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete rule: {e}")
            return False
