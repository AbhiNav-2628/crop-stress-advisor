"""Forward-chaining advisory rule engine.

Rules live in config/advisory_rules.yaml and are evaluated against a flat
dictionary of "facts" (predicted category, confidence, crop, feature values,
data-quality figures). The engine knows nothing about how the facts were
produced, so it works the same with any classifier.
"""

import math
import operator
from dataclasses import dataclass, field

SEVERITY_ORDER = {"info": 0, "low": 1, "moderate": 2, "high": 3}
REQUIRED_FIELDS = ["id", "title", "categories", "message", "verification", "severity", "priority"]

OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
    "in": lambda value, options: value in options,
    "not_in": lambda value, options: value not in options,
}


class RuleConfigError(ValueError):
    """The advisory rule file is malformed."""


@dataclass
class Advisory:
    rule_id: str
    title: str
    severity: str
    priority: int
    message: str
    verification: str
    limitations: str
    source: str
    illustrative: bool
    evidence: dict = field(default_factory=dict)
    matched_conditions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def validate_rules(rules: list[dict]) -> None:
    seen = set()
    for index, rule in enumerate(rules):
        missing = [key for key in REQUIRED_FIELDS if key not in rule]
        if missing:
            raise RuleConfigError(f"Advisory rule #{index + 1} is missing fields: {missing}")
        if rule["id"] in seen:
            raise RuleConfigError(f"Duplicate advisory rule id: {rule['id']}")
        seen.add(rule["id"])
        if rule["severity"] not in SEVERITY_ORDER:
            raise RuleConfigError(f"Rule {rule['id']} has unknown severity '{rule['severity']}'")
        for condition in rule.get("conditions", []):
            if condition.get("op") not in OPERATORS or "fact" not in condition or "value" not in condition:
                raise RuleConfigError(f"Rule {rule['id']} has an invalid condition: {condition}")


def build_facts(prediction, features: dict, crop: str, valid_fraction: float, n_warnings: int) -> dict:
    """Flatten everything the rules may refer to into one dictionary."""
    facts = {
        "category": prediction.category,
        "raw_category": prediction.raw_category,
        "confidence": prediction.confidence,
        "is_inconclusive": prediction.is_inconclusive,
        "classifier_mode": prediction.classifier.get("mode"),
        "crop": crop,
        "valid_fraction": valid_fraction,
        "n_quality_warnings": n_warnings,
    }
    facts.update(features)
    return facts


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _check_condition(condition: dict, facts: dict) -> tuple[bool, str]:
    fact, op, expected = condition["fact"], condition["op"], condition["value"]
    actual = facts.get(fact)
    if _is_missing(actual):
        return False, f"{fact} unavailable"
    try:
        passed = bool(OPERATORS[op](actual, expected))
    except TypeError:
        return False, f"{fact} could not be compared with {expected!r}"
    shown = f"{actual:.3f}" if isinstance(actual, float) else repr(actual)
    return passed, f"{fact} {op} {expected!r} (actual {shown})"


def _applies(rule: dict, facts: dict) -> bool:
    categories = rule["categories"]
    crops = rule.get("crops", ["any"])
    category_ok = "any" in categories or facts["category"] in categories
    crop_ok = "any" in crops or facts.get("crop") in crops
    return category_ok and crop_ok


def evaluate_rules(rules: list[dict], facts: dict) -> list[Advisory]:
    """Return every advisory whose category/crop scope and conditions all match.

    Results are ordered by priority (high first), then by severity.
    """
    validate_rules(rules)
    advisories = []
    for rule in rules:
        if not _applies(rule, facts):
            continue
        checks = [_check_condition(c, facts) for c in rule.get("conditions", [])]
        if not all(passed for passed, _ in checks):
            continue
        evidence = {name: facts.get(name) for name in rule.get("evidence", [])}
        advisories.append(Advisory(
            rule_id=rule["id"],
            title=rule["title"],
            severity=rule["severity"],
            priority=int(rule["priority"]),
            message=rule["message"].strip(),
            verification=rule["verification"].strip(),
            limitations=rule.get("limitations", "").strip(),
            source=rule.get("source", "Illustrative rule - requires review by an agronomist.").strip(),
            illustrative=bool(rule.get("illustrative", True)),
            evidence=evidence,
            matched_conditions=[text for _, text in checks],
        ))
    advisories.sort(key=lambda a: (a.priority, SEVERITY_ORDER[a.severity]), reverse=True)
    return advisories
