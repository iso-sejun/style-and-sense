from collections import Counter

import pytest

from style_and_sense.style_rules import (
    StyleRuleError,
    load_style_rules,
    parse_style_rule,
    search_style_rules,
)


def test_load_style_rules_has_mvp_scale_and_unique_ids():
    rules = load_style_rules()
    ids = [rule.id for rule in rules]

    assert len(rules) >= 75
    assert len(ids) == len(set(ids))


def test_load_style_rules_covers_expected_categories():
    rules = load_style_rules()
    categories = Counter(rule.category for rule in rules)

    assert set(categories) >= {
        "color",
        "occasion",
        "weather",
        "layering",
        "silhouette",
        "formality",
        "footwear",
        "accessories",
        "pattern",
    }


def test_search_style_rules_returns_relevant_weather_rule():
    rules = load_style_rules()
    results = search_style_rules(
        "class outfit 65 degrees cool weather cardigan layer",
        rules=rules,
        limit=5,
    )

    assert len(results) <= 5
    assert any(rule.category in {"weather", "layering"} for rule in results)


def test_parse_style_rule_rejects_missing_required_fields():
    with pytest.raises(StyleRuleError):
        parse_style_rule({"id": "bad_rule"})

