import pytest

from src.advisory.rule_engine import RuleConfigError, build_facts, evaluate_rules, validate_rules
from src.models.inference import Prediction


def prediction(category, confidence=0.9, inconclusive=False):
    return Prediction(category, category, confidence, None, inconclusive, classifier={"mode": "ml_demo"})


def facts_for(category, crop="generic", confidence=0.9, inconclusive=False, valid_fraction=1.0,
              warnings=0, **features):
    base = {"ndvi_mean": 0.7, "ndre_mean": 0.2, "gndvi_mean": 0.6, "savi_mean": 0.5,
            "ndvi_std": 0.02, "ndvi_p10": 0.65, "low_vigor_fraction": 0.0, "ndre_ndvi_ratio": 0.3}
    base.update(features)
    return build_facts(prediction(category, confidence, inconclusive), base, crop, valid_fraction, warnings)


def ids(advisories):
    return [a.rule_id for a in advisories]


def test_bundled_rules_are_valid(advisory_rules):
    validate_rules(advisory_rules)
    assert len({r["id"] for r in advisory_rules}) == len(advisory_rules)


def test_every_rule_has_verification_step(advisory_rules):
    assert all(r["verification"].strip() for r in advisory_rules)


def test_water_stress_rules_match(advisory_rules):
    result = ids(evaluate_rules(advisory_rules, facts_for("water_stress")))
    assert "ADV-WATER-01" in result
    assert "ADV-NUTR-01" not in result
    assert result[-1] == "ADV-GEN-01"  # general disclaimer always present, lowest priority


def test_crop_specific_rule(advisory_rules):
    assert "ADV-NUTR-02" in ids(evaluate_rules(advisory_rules, facts_for("nutrient_stress", crop="rice")))
    assert "ADV-NUTR-02" not in ids(evaluate_rules(advisory_rules, facts_for("nutrient_stress", crop="cotton")))


def test_priority_ordering(advisory_rules):
    advisories = evaluate_rules(advisory_rules, facts_for("pest_stress", ndvi_p10=0.3, ndvi_std=0.12))
    priorities = [a.priority for a in advisories]
    assert priorities == sorted(priorities, reverse=True)
    assert ids(advisories)[0] == "ADV-SEV-01"


def test_inconclusive_with_poor_data(advisory_rules):
    result = ids(evaluate_rules(advisory_rules, facts_for("inconclusive", inconclusive=True,
                                                          confidence=None, valid_fraction=0.3, warnings=1)))
    assert result[:2] == ["ADV-INC-02", "ADV-INC-01"]
    assert "ADV-QUAL-01" in result
    assert "ADV-CONF-01" not in result  # confidence rule must not fire without a confidence


def test_low_confidence_advisory(advisory_rules):
    assert "ADV-CONF-01" in ids(evaluate_rules(advisory_rules, facts_for("healthy", confidence=0.65)))
    assert "ADV-CONF-01" not in ids(evaluate_rules(advisory_rules, facts_for("healthy", confidence=0.9)))


def test_missing_fact_does_not_match(advisory_rules):
    advisories = evaluate_rules(advisory_rules, facts_for("water_stress", ndvi_std=float("nan")))
    assert "ADV-PATCH-01" not in ids(advisories)


def test_evidence_contains_actual_values(advisory_rules):
    advisories = evaluate_rules(advisory_rules, facts_for("nutrient_stress", ndre_ndvi_ratio=0.24))
    nutrient = next(a for a in advisories if a.rule_id == "ADV-NUTR-01")
    assert nutrient.evidence["ndre_ndvi_ratio"] == 0.24


def test_no_rule_prescribes_chemicals(advisory_rules):
    forbidden = ["kg/ha", "ml/l", "spray ", "apply urea", "dose of"]
    for rule in advisory_rules:
        text = (rule["message"] + rule["verification"]).lower()
        assert not any(word in text for word in forbidden), rule["id"]


def test_engine_is_independent_of_classifier(advisory_rules):
    """Same facts -> same advisories, whichever classifier mode produced them."""
    a = facts_for("disease_stress")
    b = dict(a, classifier_mode="rule_based")
    assert ids(evaluate_rules(advisory_rules, a)) == ids(evaluate_rules(advisory_rules, b))


def test_invalid_rule_config_rejected():
    with pytest.raises(RuleConfigError, match="missing fields"):
        validate_rules([{"id": "X"}])
    rule = {"id": "A", "title": "t", "categories": ["any"], "message": "m", "verification": "v",
            "severity": "low", "priority": 1}
    with pytest.raises(RuleConfigError, match="Duplicate"):
        validate_rules([rule, dict(rule)])
    with pytest.raises(RuleConfigError, match="invalid condition"):
        validate_rules([dict(rule, conditions=[{"fact": "x", "op": "~", "value": 1}])])
