from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from style_and_sense.config import STYLE_RULES_DIR


RULES_PATH = STYLE_RULES_DIR / "rules.json"
REQUIRED_FIELDS = {"id", "category", "title", "text", "tags"}


@dataclass(frozen=True)
class StyleRule:
    id: str
    category: str
    title: str
    text: str
    tags: list[str]
    applies_to: list[str]
    occasion_tags: list[str]
    season_tags: list[str]


class StyleRuleError(ValueError):
    pass


def load_style_rules(path: Path = RULES_PATH) -> list[StyleRule]:
    with path.open("r", encoding="utf-8") as file:
        raw_rules = json.load(file)

    if not isinstance(raw_rules, list):
        raise StyleRuleError("Style rules file must contain a JSON list.")

    rules = [parse_style_rule(raw_rule) for raw_rule in raw_rules]
    ids = [rule.id for rule in rules]
    if len(ids) != len(set(ids)):
        raise StyleRuleError("Style rule IDs must be unique.")
    return rules


def parse_style_rule(raw_rule: dict) -> StyleRule:
    if not isinstance(raw_rule, dict):
        raise StyleRuleError("Each style rule must be a JSON object.")

    missing = REQUIRED_FIELDS - set(raw_rule)
    if missing:
        raise StyleRuleError(f"Style rule is missing required fields: {sorted(missing)}")

    tags = raw_rule["tags"]
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise StyleRuleError("Style rule tags must be a list of strings.")

    return StyleRule(
        id=require_string(raw_rule, "id"),
        category=require_string(raw_rule, "category"),
        title=require_string(raw_rule, "title"),
        text=require_string(raw_rule, "text"),
        tags=tags,
        applies_to=optional_string_list(raw_rule, "applies_to"),
        occasion_tags=optional_string_list(raw_rule, "occasion_tags"),
        season_tags=optional_string_list(raw_rule, "season_tags"),
    )


def require_string(raw_rule: dict, key: str) -> str:
    value = raw_rule[key]
    if not isinstance(value, str) or not value.strip():
        raise StyleRuleError(f"Style rule field `{key}` must be a non-empty string.")
    return value


def optional_string_list(raw_rule: dict, key: str) -> list[str]:
    value = raw_rule.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise StyleRuleError(f"Style rule field `{key}` must be a list of strings.")
    return value


def rule_search_text(rule: StyleRule) -> str:
    return " ".join(
        [
            rule.category,
            rule.title,
            rule.text,
            " ".join(rule.tags),
            " ".join(rule.applies_to),
            " ".join(rule.occasion_tags),
            " ".join(rule.season_tags),
        ]
    ).lower()


def search_style_rules(
    query: str,
    *,
    rules: list[StyleRule] | None = None,
    limit: int = 5,
) -> list[StyleRule]:
    rules = rules if rules is not None else load_style_rules()
    terms = [term for term in query.lower().replace(",", " ").split() if term]
    if not terms:
        return rules[:limit]

    scored: list[tuple[int, StyleRule]] = []
    for rule in rules:
        text = rule_search_text(rule)
        score = sum(text.count(term) for term in terms)
        if score:
            scored.append((score, rule))

    scored.sort(key=lambda item: (-item[0], item[1].id))
    return [rule for _, rule in scored[:limit]]

