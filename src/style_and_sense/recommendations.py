from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from style_and_sense.retrieval import RetrievalResult, search_garments
from style_and_sense.storage import (
    create_outfit,
    create_recommendation_request,
    list_garments,
)
from style_and_sense.style_rules import StyleRule, load_style_rules, search_style_rules


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_DARTMOUTH_MODEL = "openai.gpt-5.6-sol"
RESPONSES_MODE = "responses"
CHAT_COMPLETIONS_MODE = "chat_completions"


OUTFIT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "outfits": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "items": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 6,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "string"},
                                "role": {"type": "string"},
                            },
                            "required": ["id", "role"],
                        },
                    },
                    "explanation": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["title", "items", "explanation", "confidence"],
            },
        }
    },
    "required": ["outfits"],
}


@dataclass(frozen=True)
class OutfitRecommendation:
    title: str
    item_ids: list[str]
    explanation: str
    confidence: float


@dataclass(frozen=True)
class RecommendationResult:
    request_id: str
    outfits: list[OutfitRecommendation]
    candidate_garments: list[dict]
    style_rules: list[StyleRule]
    latency_ms: int
    used_openai: bool
    message: str | None = None


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    model: str
    api_mode: str
    base_url: str | None = None


def generate_recommendations(
    user_prompt: str,
    *,
    weather_text: str | None = None,
    occasion: str | None = None,
    top_k_garments: int = 24,
    top_k_rules: int = 5,
) -> RecommendationResult:
    load_env_if_available()
    start = time.perf_counter()
    available_garments = [
        garment
        for garment in list_garments()
        if garment.get("laundry_status") == "available"
    ]
    context_query = build_context_query(
        user_prompt,
        weather_text=weather_text,
        occasion=occasion,
    )
    candidate_results = retrieve_candidate_garments(
        context_query,
        available_garments=available_garments,
        top_k=top_k_garments,
    )
    candidate_garments = [result.garment for result in candidate_results]
    style_rules = search_style_rules(
        build_style_rule_query(context_query, candidate_garments),
        rules=load_style_rules(),
        limit=top_k_rules,
    )

    message = validate_minimum_closet(candidate_garments)
    if message:
        latency_ms = elapsed_ms(start)
        request_id = store_recommendation_result(
            user_prompt=user_prompt,
            weather_text=weather_text,
            occasion=occasion,
            candidate_garments=candidate_garments,
            style_rules=style_rules,
            outfits=[],
            latency_ms=latency_ms,
        )
        return RecommendationResult(
            request_id=request_id,
            outfits=[],
            candidate_garments=candidate_garments,
            style_rules=style_rules,
            latency_ms=latency_ms,
            used_openai=False,
            message=message,
        )

    used_openai = False
    result_message = None
    raw_outfits: list[dict[str, Any]]
    if os.getenv("OPENAI_API_KEY"):
        try:
            raw_outfits = generate_openai_outfits(
                user_prompt=user_prompt,
                weather_text=weather_text,
                occasion=occasion,
                candidate_garments=candidate_garments,
                style_rules=style_rules,
            )
            used_openai = True
        except Exception as exc:
            raw_outfits = generate_fallback_outfits(
                user_prompt=user_prompt,
                candidate_garments=candidate_garments,
                style_rules=style_rules,
            )
            result_message = f"OpenAI generation failed, so local fallback was used: {exc}"
    else:
        raw_outfits = generate_fallback_outfits(
            user_prompt=user_prompt,
            candidate_garments=candidate_garments,
            style_rules=style_rules,
        )
        result_message = "Using local fallback because OPENAI_API_KEY is not set."

    outfits = validate_outfits(raw_outfits, candidate_garments)
    if used_openai and not outfits:
        fallback_raw_outfits = generate_fallback_outfits(
            user_prompt=user_prompt,
            candidate_garments=candidate_garments,
            style_rules=style_rules,
        )
        outfits = validate_outfits(fallback_raw_outfits, candidate_garments)
        used_openai = False
        result_message = "OpenAI returned no valid outfits, so local fallback was used."
    latency_ms = elapsed_ms(start)
    request_id = store_recommendation_result(
        user_prompt=user_prompt,
        weather_text=weather_text,
        occasion=occasion,
        candidate_garments=candidate_garments,
        style_rules=style_rules,
        outfits=outfits,
        latency_ms=latency_ms,
    )
    return RecommendationResult(
        request_id=request_id,
        outfits=outfits,
        candidate_garments=candidate_garments,
        style_rules=style_rules,
        latency_ms=latency_ms,
        used_openai=used_openai,
        message=result_message,
    )


def retrieve_candidate_garments(
    query: str,
    *,
    available_garments: list[dict],
    top_k: int,
) -> list[RetrievalResult]:
    ranked = search_garments(query, garments=available_garments, top_k=top_k)
    seen = {result.garment["id"] for result in ranked}
    for garment in available_garments:
        if garment["id"] not in seen:
            ranked.append(RetrievalResult(garment=garment, score=0.0))
            seen.add(garment["id"])
        if len(ranked) >= top_k:
            break
    return ranked


def generate_openai_outfits(
    *,
    user_prompt: str,
    weather_text: str | None,
    occasion: str | None,
    candidate_garments: list[dict],
    style_rules: list[StyleRule],
) -> list[dict[str, Any]]:
    config = get_llm_config()
    from openai import OpenAI

    client_kwargs = {"api_key": config.api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url
    client = OpenAI(**client_kwargs)

    if config.api_mode == CHAT_COMPLETIONS_MODE:
        return generate_chat_completion_outfits(
            client=client,
            model=config.model,
            user_prompt=user_prompt,
            weather_text=weather_text,
            occasion=occasion,
            candidate_garments=candidate_garments,
            style_rules=style_rules,
        )

    response = client.responses.create(
        model=config.model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are an outfit recommendation engine for students. "
                            "Recommend complete outfits only from the provided garment IDs. "
                            "Use the style rules for reasoning, but write one natural "
                            "student-facing paragraph per outfit. Return JSON only."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_prompt(
                            user_prompt=user_prompt,
                            weather_text=weather_text,
                            occasion=occasion,
                            candidate_garments=candidate_garments,
                            style_rules=style_rules,
                        ),
                    }
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "outfit_recommendations",
                "schema": OUTFIT_SCHEMA,
                "strict": True,
            }
        },
    )
    parsed = json.loads(response.output_text)
    return parsed.get("outfits", [])


def generate_chat_completion_outfits(
    *,
    client,
    model: str,
    user_prompt: str,
    weather_text: str | None,
    occasion: str | None,
    candidate_garments: list[dict],
    style_rules: list[StyleRule],
) -> list[dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an outfit recommendation engine for students. "
                    "Recommend complete outfits only from the provided garment IDs. "
                    "Return only a valid JSON object. Do not wrap it in markdown. "
                    "Use one natural student-facing paragraph per outfit."
                ),
            },
            {
                "role": "user",
                "content": build_chat_completion_prompt(
                    user_prompt=user_prompt,
                    weather_text=weather_text,
                    occasion=occasion,
                    candidate_garments=candidate_garments,
                    style_rules=style_rules,
                ),
            },
        ],
        stream=False,
    )
    content = response.choices[0].message.content or ""
    parsed = parse_json_object(content)
    return parsed.get("outfits", [])


def build_chat_completion_prompt(
    *,
    user_prompt: str,
    weather_text: str | None,
    occasion: str | None,
    candidate_garments: list[dict],
    style_rules: list[StyleRule],
) -> str:
    return (
        "Return a JSON object matching this schema shape:\n"
        '{"outfits":[{"title":"string","items":[{"id":"garment_id","role":"top|bottom|dress|shoes|outerwear|accessory"}],"explanation":"string","confidence":0.0}]}\n\n'
        "Rules:\n"
        "- Return up to 3 outfits.\n"
        "- Use only garment IDs listed in available_garments.\n"
        "- Do not invent item IDs.\n"
        "- Do not use duplicate IDs inside an outfit.\n"
        "- A standard outfit needs top + bottom + shoes when shoes are available.\n"
        "- A dress outfit needs dress + shoes when shoes are available.\n"
        "- Each explanation must be one natural paragraph.\n\n"
        "Context:\n"
        f"{build_prompt(user_prompt=user_prompt, weather_text=weather_text, occasion=occasion, candidate_garments=candidate_garments, style_rules=style_rules)}"
    )


def parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def get_llm_config() -> LLMConfig:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    base_url = os.getenv("OPENAI_BASE_URL") or None
    default_model = DEFAULT_DARTMOUTH_MODEL if base_url else DEFAULT_OPENAI_MODEL
    model = os.getenv("OPENAI_MODEL") or default_model
    api_mode = os.getenv("OPENAI_API_MODE")
    if not api_mode:
        api_mode = CHAT_COMPLETIONS_MODE if base_url else RESPONSES_MODE
    api_mode = api_mode.lower().replace("-", "_")
    if api_mode not in {RESPONSES_MODE, CHAT_COMPLETIONS_MODE}:
        raise ValueError(
            f"Unsupported OPENAI_API_MODE `{api_mode}`. "
            f"Use `{RESPONSES_MODE}` or `{CHAT_COMPLETIONS_MODE}`."
        )
    return LLMConfig(
        api_key=api_key,
        model=model,
        api_mode=api_mode,
        base_url=base_url,
    )


def build_prompt(
    *,
    user_prompt: str,
    weather_text: str | None,
    occasion: str | None,
    candidate_garments: list[dict],
    style_rules: list[StyleRule],
) -> str:
    context = {
        "user_request": user_prompt,
        "weather_text": weather_text,
        "occasion": occasion,
        "available_garments": [
            {
                "id": garment["id"],
                "category": garment["category"],
                "subcategory": garment["subcategory"],
                "colors": garment["colors"],
                "style_tags": garment["style_tags"],
                "season_tags": garment["season_tags"],
                "formality": garment["formality"],
                "caption": garment["caption"],
            }
            for garment in candidate_garments
        ],
        "style_rules": [
            {
                "id": rule.id,
                "category": rule.category,
                "title": rule.title,
                "text": rule.text,
            }
            for rule in style_rules
        ],
        "constraints": [
            "Return exactly 3 outfits when enough garments exist.",
            "Only use IDs from available_garments.",
            "Do not use duplicate garment IDs inside one outfit.",
            "A standard outfit needs top + bottom + shoes when available.",
            "A dress outfit needs dress + shoes when available.",
            "Include outerwear when weather suggests cool, cold, rain, or layering and an outerwear candidate exists.",
            "Each explanation must be one natural paragraph.",
        ],
    }
    return json.dumps(context, indent=2)


def generate_fallback_outfits(
    *,
    user_prompt: str,
    candidate_garments: list[dict],
    style_rules: list[StyleRule],
) -> list[dict[str, Any]]:
    by_category = group_garments_by_category(candidate_garments)
    outfits: list[dict[str, Any]] = []
    templates = [
        ("Easy Class Fit", 0),
        ("Clean Casual Option", 1),
        ("Put-Together Backup", 2),
    ]
    for title, offset in templates:
        items = choose_fallback_items(by_category, offset)
        if not items:
            continue
        outfits.append(
            {
                "title": title,
                "items": [
                    {"id": garment["id"], "role": garment["category"]}
                    for garment in items
                ],
                "explanation": fallback_explanation(
                    items,
                    user_prompt=user_prompt,
                    style_rules=style_rules,
                ),
                "confidence": 0.62,
            }
        )
    return outfits


def choose_fallback_items(by_category: dict[str, list[dict]], offset: int) -> list[dict]:
    if by_category.get("dress"):
        items = [pick(by_category["dress"], offset)]
    elif by_category.get("top") and by_category.get("bottom"):
        items = [pick(by_category["top"], offset), pick(by_category["bottom"], offset)]
    else:
        return []

    if by_category.get("shoes"):
        items.append(pick(by_category["shoes"], offset))
    if by_category.get("outerwear"):
        items.append(pick(by_category["outerwear"], offset))
    if by_category.get("accessory"):
        items.append(pick(by_category["accessory"], offset))
    return dedupe_items(items)


def pick(items: list[dict], offset: int) -> dict:
    return items[offset % len(items)]


def dedupe_items(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in items:
        if item["id"] not in seen:
            deduped.append(item)
            seen.add(item["id"])
    return deduped


def fallback_explanation(
    items: list[dict],
    *,
    user_prompt: str,
    style_rules: list[StyleRule],
) -> str:
    item_text = ", ".join((item.get("caption") or item.get("subcategory") or item["category"]) for item in items)
    rule_hint = style_rules[0].text if style_rules else "The pieces are chosen to make a complete, wearable outfit from available closet items."
    return (
        f"This outfit uses {item_text}. It is built for your request, '{user_prompt}', "
        f"by keeping the pieces wearable together and using this styling idea: {rule_hint}"
    )


def validate_outfits(
    raw_outfits: list[dict[str, Any]],
    candidate_garments: list[dict],
) -> list[OutfitRecommendation]:
    candidate_by_id = {garment["id"]: garment for garment in candidate_garments}
    available_categories = {garment["category"] for garment in candidate_garments}
    valid: list[OutfitRecommendation] = []
    seen_combos: set[tuple[str, ...]] = set()

    for raw_outfit in raw_outfits:
        item_ids = [
            item.get("id")
            for item in raw_outfit.get("items", [])
            if isinstance(item, dict)
        ]
        item_ids = [item_id for item_id in item_ids if isinstance(item_id, str)]
        if len(item_ids) != len(set(item_ids)):
            continue
        if any(item_id not in candidate_by_id for item_id in item_ids):
            continue
        garments = [candidate_by_id[item_id] for item_id in item_ids]
        if not is_complete_outfit(garments, available_categories=available_categories):
            continue

        combo = tuple(sorted(item_ids))
        if combo in seen_combos:
            continue
        seen_combos.add(combo)

        title = str(raw_outfit.get("title") or "Outfit")
        explanation = str(raw_outfit.get("explanation") or "").strip()
        if not explanation:
            continue
        confidence = clamp_confidence(raw_outfit.get("confidence", 0.5))
        valid.append(
            OutfitRecommendation(
                title=title,
                item_ids=item_ids,
                explanation=explanation,
                confidence=confidence,
            )
        )
        if len(valid) == 3:
            break

    return valid


def is_complete_outfit(
    garments: list[dict],
    *,
    available_categories: set[str] | None = None,
) -> bool:
    categories = {garment["category"] for garment in garments}
    has_clothing_base = "dress" in categories or {"top", "bottom"} <= categories
    if not has_clothing_base:
        return False
    available_categories = available_categories or categories
    return "shoes" in categories or "shoes" not in available_categories


def clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, confidence))


def validate_minimum_closet(candidate_garments: list[dict]) -> str | None:
    categories = {garment["category"] for garment in candidate_garments}
    if "dress" not in categories and "top" not in categories:
        return "Add at least one top or dress before generating outfits."
    if "dress" not in categories and "bottom" not in categories:
        return "Add at least one bottom or dress before generating outfits."
    return None


def group_garments_by_category(garments: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for garment in garments:
        grouped.setdefault(garment["category"], []).append(garment)
    return grouped


def build_context_query(
    user_prompt: str,
    *,
    weather_text: str | None,
    occasion: str | None,
) -> str:
    return " ".join(part for part in [user_prompt, weather_text, occasion] if part)


def build_style_rule_query(query: str, candidate_garments: list[dict]) -> str:
    garment_terms: list[str] = []
    for garment in candidate_garments:
        garment_terms.extend(
            [
                garment.get("category") or "",
                garment.get("subcategory") or "",
                " ".join(garment.get("colors") or []),
                " ".join(garment.get("style_tags") or []),
                " ".join(garment.get("season_tags") or []),
            ]
        )
    return " ".join([query, *garment_terms])


def store_recommendation_result(
    *,
    user_prompt: str,
    weather_text: str | None,
    occasion: str | None,
    candidate_garments: list[dict],
    style_rules: list[StyleRule],
    outfits: list[OutfitRecommendation],
    latency_ms: int,
) -> str:
    request_id = create_recommendation_request(
        user_prompt=user_prompt,
        weather_text=weather_text,
        occasion=occasion,
        raw_context_json=json.dumps(
            {
                "candidate_garment_count": len(candidate_garments),
                "style_rule_count": len(style_rules),
            }
        ),
        retrieved_garment_ids=[garment["id"] for garment in candidate_garments],
        retrieved_rule_ids=[rule.id for rule in style_rules],
        latency_ms=latency_ms,
    )
    for outfit in outfits:
        create_outfit(
            request_id=request_id,
            title=outfit.title,
            item_ids=outfit.item_ids,
            explanation=outfit.explanation,
            score=outfit.confidence,
        )
    return request_id


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def parse_temperature(text: str) -> int | None:
    match = re.search(r"(-?\d{1,3})\s*(?:degrees|degree|deg|f)\b", text.lower())
    return int(match.group(1)) if match else None


def load_env_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
